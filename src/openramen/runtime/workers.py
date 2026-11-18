"""Running nodes in other processes.

A node whose spec says ``executor="process"`` is called in a worker process:
pure-Python work then really runs in parallel, a crash takes only the worker
down, and Stop can kill it.  Workers are started on first use and kept.

Large arrays do not travel through the pipe.  They are placed in shared
memory, which the worker maps: an array is copied at most once on its way to
a worker and once on its way back, and an array that came out of a worker
goes into the next one without being copied at all.  The parent process
creates and removes every shared block, so killing a worker leaks nothing.
"""

from __future__ import annotations

import atexit
import hashlib
import multiprocessing
import os
import pickle
import queue
import re
import sys
import threading
import traceback
import weakref
from multiprocessing import shared_memory
from typing import Any, Callable, Optional

import numpy as np

__all__ = ["call", "shutdown", "start", "RemoteError", "WorkerDied", "Unsendable", "Stopped",
           "SHARE_FROM"]

#: Arrays of at least this many bytes go through shared memory.
SHARE_FROM = 1 << 18
_READ_ONLY = re.compile(r"read[- ]?only|not writ(?:e)?able", re.I)
_TRACK = {"track": False} if sys.version_info >= (3, 13) else {}


class Stopped(Exception):
    """The call was abandoned because the run was cancelled."""


class WorkerDied(Exception):
    """The worker process ended while it was running a node."""


class Unsendable(Exception):
    """The node or its arguments cannot be sent to another process."""


class RemoteError(Exception):
    """The node failed in the worker.  ``original`` is the exception it raised
    when that could be brought back, else an equivalent ``RuntimeError``."""

    def __init__(self, original: BaseException, remote_traceback: str):
        super().__init__(str(original))
        self.original = original
        self.remote_traceback = remote_traceback


class _Shared:
    """Stands for an array that lives in a shared memory block."""

    __slots__ = ("name", "offset", "shape", "strides", "dtype", "private")

    def __init__(self, name: str, offset: int, shape: tuple, strides: Optional[tuple],
                 dtype: Any, private: bool):
        self.name, self.offset, self.shape = name, offset, shape
        self.strides, self.dtype, self.private = strides, dtype, private

    def __reduce__(self) -> tuple:
        return (_Shared, (self.name, self.offset, self.shape, self.strides, self.dtype,
                          self.private))


class _Need:
    """Stands for a result array that still has to be copied to shared memory."""

    __slots__ = ("index",)

    def __init__(self, index: int):
        self.index = index


def _address(array: np.ndarray) -> int:
    return int(array.__array_interface__["data"][0])


def _start_of(shm: shared_memory.SharedMemory) -> int:
    """Where a shared block is mapped in this process."""
    probe: np.ndarray = np.frombuffer(shm.buf, dtype=np.uint8)   # type: ignore[arg-type]
    return _address(probe)


def _bounds(array: np.ndarray) -> tuple[int, int]:
    """First byte and one past the last byte the array touches."""
    low = high = _address(array)
    for length, stride in zip(array.shape, array.strides):
        if length == 0:
            return low, low
        span = (length - 1) * stride
        if stride > 0:
            high += span
        else:
            low += span
    return low, high + array.itemsize


def _walk(value: Any, leaf: Callable[[Any], Any]) -> Any:
    """Apply ``leaf`` to every array and marker in plain containers."""
    if isinstance(value, (np.ndarray, _Shared, _Need)):
        return leaf(value)
    if type(value) is list:
        return [_walk(item, leaf) for item in value]
    if type(value) is tuple:
        return tuple(_walk(item, leaf) for item in value)
    if type(value) is dict:
        return {key: _walk(item, leaf) for key, item in value.items()}
    return value


# --------------------------------------------------------------------------
# Shared memory blocks (parent process)
# --------------------------------------------------------------------------

class _Segment:
    __slots__ = ("shm", "refs", "address", "size")

    def __init__(self, nbytes: int):
        self.shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
        self.refs = 0
        self.address = _start_of(self.shm)
        self.size = nbytes


_LOCK = threading.RLock()
_SEGMENTS: dict[str, _Segment] = {}                     # blocks this process owns
_ROOTS: dict[int, tuple[Any, _Segment]] = {}            # arrays laid over them
_GRAVE: list[shared_memory.SharedMemory] = []           # unlinked, not yet unmapped


def _sweep() -> None:
    """Unmap blocks whose last array has gone."""
    with _LOCK:
        for shm in list(_GRAVE):
            try:
                shm.close()
            except BufferError:
                continue                 # an array over it is still being torn down
            _GRAVE.remove(shm)


def _create(nbytes: int) -> _Segment:
    _sweep()
    segment = _Segment(nbytes)
    with _LOCK:
        _SEGMENTS[segment.shm.name] = segment
    return segment


def _release(segment: _Segment) -> None:
    with _LOCK:
        segment.refs -= 1
        if segment.refs > 0:
            return
        if _SEGMENTS.pop(segment.shm.name, None) is None:
            return
        try:
            segment.shm.unlink()
        except FileNotFoundError:
            pass
        _GRAVE.append(segment.shm)


def _forget(key: int, segment: _Segment) -> None:
    with _LOCK:
        _ROOTS.pop(key, None)
    _release(segment)


def _array_over(segment: _Segment, marker: _Shared) -> np.ndarray:
    """An array in the parent that looks at a shared block.  It is read-only:
    the block may be mapped by workers and reused for later calls."""
    array = np.ndarray(marker.shape, marker.dtype, buffer=segment.shm.buf,
                       offset=marker.offset, strides=marker.strides)
    array.flags.writeable = False
    with _LOCK:
        segment.refs += 1
        _ROOTS[id(array)] = (weakref.ref(array), segment)
    weakref.finalize(array, _forget, id(array), segment)
    return array


def _segment_of(array: np.ndarray) -> Optional[_Segment]:
    """The shared block an array (or the array it is a view of) lives in."""
    root: Any = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    entry = _ROOTS.get(id(root))
    if entry is not None and entry[0]() is root:
        return entry[1]
    return None


def shared_blocks() -> int:
    """How many shared memory blocks this process currently owns."""
    return len(_SEGMENTS)


# --------------------------------------------------------------------------
# Worker process
# --------------------------------------------------------------------------

def _serve(conn: Any) -> None:
    """Main loop of a worker: one request at a time until the pipe closes."""
    specs: dict[bytes, Any] = {}
    while True:
        try:
            message = conn.recv_bytes()
        except (EOFError, OSError):
            return
        request = pickle.loads(message)
        if request[0] == "stop":
            return
        try:
            if request[0] == "ping":
                conn.send_bytes(pickle.dumps("pong"))
                continue
            conn.send_bytes(_handle(conn, specs, *request[1:]))
        except (EOFError, OSError):
            return


def _handle(conn: Any, specs: dict[bytes, Any], key: bytes, spec_blob: Optional[bytes],
            payload: bytes) -> bytes:
    attached: list[tuple[shared_memory.SharedMemory, int, int, _Shared]] = []
    arguments: Any = None
    results: Any = None
    encoded: Any = None
    try:
        try:
            if spec_blob is not None:
                specs[key] = pickle.loads(spec_blob)
            spec = specs[key]
            arguments, config, share_from = pickle.loads(payload)
        except Exception as exc:
            return pickle.dumps(("unsendable", f"{type(exc).__name__}: {exc}"))

        def attach(marker: Any) -> Any:
            if not isinstance(marker, _Shared):
                return marker
            shm = shared_memory.SharedMemory(name=marker.name, **_TRACK)
            array = np.ndarray(marker.shape, marker.dtype, buffer=shm.buf,
                               offset=marker.offset, strides=marker.strides)
            array.flags.writeable = marker.private
            start = _start_of(shm)
            attached.append((shm, start, start + shm.size, marker))
            return array

        arguments = {pin: _walk(value, attach) for pin, value in arguments.items()}
        needed_writable = False
        try:
            results = spec.call(arguments, config)
        except (ValueError, TypeError, BufferError, RuntimeError) as exc:
            frozen = [pin for pin, value in arguments.items()
                      if isinstance(value, np.ndarray) and not value.flags.writeable]
            if not frozen or not _READ_ONLY.search(str(exc)):
                raise
            for pin in frozen:
                arguments[pin] = arguments[pin].copy()
            results = spec.call(arguments, config)
            needed_writable = True

        sizes: list[int] = []
        waiting: list[np.ndarray] = []

        def send(value: Any) -> Any:
            if not isinstance(value, np.ndarray) or value.dtype.hasobject \
                    or value.nbytes < share_from:
                return value
            low, high = _bounds(value)
            for _shm, start, end, marker in attached:
                if start <= low and high <= end:      # it lies in a block we were given
                    return _Shared(marker.name, _address(value) - start, value.shape,
                                   value.strides, value.dtype, marker.private)
            sizes.append(value.nbytes)
            waiting.append(value)
            return _Need(len(sizes) - 1)

        encoded = {pin: _walk(value, send) for pin, value in results.items()}
        if sizes:
            # ask the parent for blocks, then copy the results into them
            conn.send_bytes(pickle.dumps(("need", sizes)))
            names = pickle.loads(conn.recv_bytes())
            markers = []
            for name, value in zip(names, waiting):
                shm = shared_memory.SharedMemory(name=name, **_TRACK)
                target = np.ndarray(value.shape, value.dtype, buffer=shm.buf)
                np.copyto(target, value)
                del target
                shm.close()
                markers.append(_Shared(name, 0, value.shape, None, value.dtype, False))
            encoded = {pin: _walk(value, lambda v: markers[v.index] if isinstance(v, _Need)
                                  else v) for pin, value in encoded.items()}
        try:
            return pickle.dumps(("done", encoded, needed_writable))
        except Exception as exc:
            return pickle.dumps(("unsendable-result", f"{type(exc).__name__}: {exc}"))
    except BaseException as exc:
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        blob: Optional[bytes]
        try:
            blob = pickle.dumps(exc)
            pickle.loads(blob)
        except Exception:
            blob = None
        return pickle.dumps(("error", type(exc).__name__, str(exc), text, blob))
    finally:
        arguments = results = encoded = None
        for shm, _, _, _ in attached:
            try:
                shm.close()
            except BufferError:
                pass                     # the function kept a reference; leave it mapped


# --------------------------------------------------------------------------
# Workers (parent process)
# --------------------------------------------------------------------------

class _Lane:
    """One worker process and the pipe to it."""

    def __init__(self) -> None:
        context = multiprocessing.get_context("spawn")
        self.conn, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child,), daemon=True,
                                       name="openramen-worker")
        self.process.start()
        child.close()
        self.known: set[bytes] = set()

    def kill(self) -> None:
        try:
            self.process.kill()
            self.process.join(2.0)
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

    def exchange(self, message: bytes, cancel: Optional[Callable[[], bool]]) -> Any:
        """Send a message and wait for the answer, watching for Stop."""
        try:
            self.conn.send_bytes(message)
            while not self.conn.poll(0.05):
                if cancel is not None and cancel():
                    raise Stopped()
                if not self.process.is_alive() and not self.conn.poll(0):
                    raise WorkerDied(self._obituary())
            return pickle.loads(self.conn.recv_bytes())
        except (EOFError, OSError, BrokenPipeError) as exc:
            raise WorkerDied(self._obituary()) from exc

    def _obituary(self) -> str:
        self.process.join(0.5)
        code = self.process.exitcode
        return ("the worker process died while running this node"
                + (f" (exit code {code})" if code is not None else "")
                + ". If this happens as soon as a script starts, put the code that runs "
                  "the graph under  if __name__ == \"__main__\":")


_LANES: list[_Lane] = []
_IDLE: "queue.LifoQueue[_Lane]" = queue.LifoQueue()
_BLOBS: dict[int, tuple[Any, bytes, bytes]] = {}


def limit() -> int:
    """How many worker processes may exist."""
    return max(1, min(8, os.cpu_count() or 2))


def _acquire(cancel: Optional[Callable[[], bool]]) -> _Lane:
    while True:
        try:
            return _IDLE.get_nowait()
        except queue.Empty:
            pass
        with _LOCK:
            if len(_LANES) < limit():
                lane = _Lane()
                _LANES.append(lane)
                return lane
        try:
            return _IDLE.get(timeout=0.05)
        except queue.Empty:
            if cancel is not None and cancel():
                raise Stopped()


def _retire(lane: _Lane) -> None:
    lane.kill()
    with _LOCK:
        if lane in _LANES:
            _LANES.remove(lane)


def start(count: Optional[int] = None) -> int:
    """Start worker processes ahead of time and wait until they are ready
    (they otherwise start on first use, which takes a moment).  Returns how
    many are running."""
    wanted = limit() if count is None else max(0, min(count, limit()))
    fresh: list[_Lane] = []
    with _LOCK:
        while len(_LANES) < wanted:
            lane = _Lane()
            _LANES.append(lane)
            fresh.append(lane)
    for lane in fresh:
        try:
            lane.exchange(pickle.dumps(("ping",)), None)
        except WorkerDied:
            _retire(lane)
        else:
            _IDLE.put(lane)
    return len(_LANES)


def shutdown() -> None:
    """Stop every worker process.  New ones start when next needed."""
    with _LOCK:
        lanes = list(_LANES)
        _LANES.clear()
        while True:
            try:
                _IDLE.get_nowait()
            except queue.Empty:
                break
    for lane in lanes:
        lane.kill()


def _after_fork() -> None:
    """A forked child must not talk to its parent's workers."""
    global _LOCK, _IDLE
    _LOCK = threading.RLock()
    _IDLE = queue.LifoQueue()
    _LANES.clear()
    _SEGMENTS.clear()
    _ROOTS.clear()
    _GRAVE.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(shutdown)


def _spec_blob(spec: Any) -> tuple[bytes, bytes]:
    """The pickled spec and a short key for it."""
    known = _BLOBS.get(id(spec))
    if known is not None and known[0] is spec:
        return known[1], known[2]
    import dataclasses
    try:
        blob = pickle.dumps(dataclasses.replace(spec, infer=None), protocol=5)
    except Exception as exc:
        raise Unsendable(f"the node cannot be sent to a worker process "
                         f"({type(exc).__name__}: {exc})") from exc
    key = hashlib.blake2b(blob, digest_size=16).digest()
    _BLOBS[id(spec)] = (spec, key, blob)
    return key, blob


def call(spec: Any, arguments: dict[str, Any], config: dict[str, Any],
         cancel: Optional[Callable[[], bool]] = None) -> tuple[dict[str, Any], bool]:
    """Call a node in a worker process.

    Returns its outputs and whether the function needed writable copies of
    arrays it had only declared to read.  Raises :class:`RemoteError` if the
    function failed, :class:`WorkerDied`, :class:`Stopped`, or
    :class:`Unsendable` (before anything ran) if the call cannot be sent.
    """
    key, blob = _spec_blob(spec)
    held: list[_Segment] = []

    def hold(segment: _Segment) -> None:
        with _LOCK:
            segment.refs += 1
        held.append(segment)

    def export(value: Any) -> Any:
        if not isinstance(value, np.ndarray) or value.dtype.hasobject \
                or value.nbytes < SHARE_FROM:
            return value
        segment = _segment_of(value)
        if segment is not None:                       # already shared: nothing to copy
            hold(segment)
            return _Shared(segment.shm.name, _address(value) - segment.address, value.shape,
                           value.strides, value.dtype, False)
        segment = _create(value.nbytes)
        hold(segment)
        target = np.ndarray(value.shape, value.dtype, buffer=segment.shm.buf)
        np.copyto(target, value)
        del target
        return _Shared(segment.shm.name, 0, value.shape, None, value.dtype, True)

    lane: Optional[_Lane] = None
    healthy = False
    try:
        try:
            payload = pickle.dumps(
                ({pin: _walk(value, export) for pin, value in arguments.items()},
                 config, SHARE_FROM), protocol=5)
        except Exception as exc:
            raise Unsendable(f"its arguments cannot be sent to a worker process "
                             f"({type(exc).__name__}: {exc})") from exc
        lane = _acquire(cancel)
        first = key not in lane.known
        reply = lane.exchange(pickle.dumps(("call", key, blob if first else None, payload),
                                           protocol=5), cancel)
        if reply[0] == "need":
            blocks = [_create(size) for size in reply[1]]
            for block in blocks:
                hold(block)
            reply = lane.exchange(pickle.dumps([block.shm.name for block in blocks]), cancel)
        healthy = True
        if reply[0] == "unsendable":
            raise Unsendable(f"it cannot be loaded in a worker process ({reply[1]})")
        lane.known.add(key)
        if reply[0] == "unsendable-result":
            raise RemoteError(RuntimeError(
                f"the result cannot be sent back from the worker process ({reply[1]}); "
                "run this node with executor 'thread'"), "")
        if reply[0] == "error":
            _, kind, message, text, pickled = reply
            original: Any = None
            if pickled is not None:
                try:
                    original = pickle.loads(pickled)
                except Exception:
                    original = None
            if not isinstance(original, BaseException):
                original = RuntimeError(f"{kind}: {message}")
            raise RemoteError(original, text)

        def adopt(marker: Any) -> Any:
            if not isinstance(marker, _Shared):
                return marker
            return _array_over(_SEGMENTS[marker.name], marker)

        _, encoded, needed_writable = reply
        return {pin: _walk(value, adopt) for pin, value in encoded.items()}, needed_writable
    finally:
        if lane is not None:
            if healthy:
                _IDLE.put(lane)
            else:
                _retire(lane)            # killed, died, or in an unknown state
        for segment in held:
            _release(segment)
