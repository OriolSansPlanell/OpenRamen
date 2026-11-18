"""Headless execution of graphs.

Nodes run one at a time, in dependency order, in the calling thread.  The
same engine runs the frames of structures (loops, cases, subgraphs), so it is
re-entrant.  Parallel executors and caching build on the same :class:`Plan`.

Values on wires have value semantics, enforced here:

* a node that declares it writes to an input (``mutate``) gets a private copy;
* arrays reach every other input as read-only views, so an undeclared write
  cannot change what another node sees;
* a shift register that is only ever updated in place keeps one buffer for
  the whole loop.
"""

from __future__ import annotations

import copy
import enum
import heapq
import itertools
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import numpy as np

from .. import types as T
from ..check import Analysis, GraphCheckError, analyze
from ..library import NodeSpec, Registry
from ..model import FRAME_IN, FRAME_OUT, INPUTS, UNSET, Graph, Structure, _structure_to_dict
from . import cache as _cache
from . import workers as _workers
from .cache import Cache

__all__ = ["Plan", "Step", "StructurePlan", "Event", "RunResult", "NodeError", "Cancelled",
           "GraphCheckError", "Run", "Cache", "compile_graph", "run", "start", "default_workers"]

_PASS, _CONVERT, _ACCEPT = 0, 1, 2           # what a wire does to its value
_READ, _MUTATE, _IN_PLACE = 0, 1, 2          # how the receiving pin uses it
_IMMUTABLE = (type(None), bool, int, float, complex, str, bytes, PurePath)
_READ_ONLY = re.compile(r"read[- ]?only|not writ(?:e)?able", re.I)

#: Specs found to need writable arrays although they do not declare that they
#: write to them.  They are given private copies from then on.
_NEEDS_WRITABLE: set[str] = set()
#: Specs the mutation detector has seen leave their inputs alone.
_VERIFIED: set[tuple[str, str]] = set()
#: Specs that asked for a worker process but turned out not to be sendable.
_LOCAL_ONLY: set[str] = set()


class Cancelled(Exception):
    """The run was stopped through its ``cancel`` callback."""


class NodeError(Exception):
    """A node failed while a graph was running.

    ``original`` is the exception raised inside the node (also available as
    ``__cause__``); ``pin`` is set when a value did not fit a pin.  Inside
    structures ``node`` is a path such as ``"loop/add_1"``, and ``context``
    says which iteration or frame was running.
    """

    def __init__(self, node: str, spec: str, original: BaseException,
                 pin: Optional[str] = None, detail: str = ""):
        self.node = node
        self.spec = spec
        self.pin = pin
        self.original = original
        self.context: list[str] = []
        what = detail or f"{type(original).__name__}: {original}"
        where = f"node {node!r} ({spec})"
        if pin:
            where += f", pin {pin!r}"
        self._message = f"{where}: {what}"
        super().__init__(self._message)

    def __str__(self) -> str:
        if not self.context:
            return self._message
        return f"{self._message} [{', '.join(self.context)}]"


@dataclass(frozen=True)
class Event:
    """Progress notification sent to a listener during a run."""

    kind: str                   # graph-started, node-started, node-finished, node-failed,
    node: Optional[str] = None  # node-warning, graph-finished, graph-failed
    duration: Optional[float] = None
    error: Optional[BaseException] = None
    message: str = ""


Listener = Callable[[Event], None]


@dataclass
class StructurePlan:
    """A structure, ready to run: its frames as plans, and what the checker
    found out about its tunnels and shift registers."""

    node: Structure
    frames: list["Plan"]
    data: dict[str, Any]
    #: Shift registers whose buffer is updated in place, one buffer per loop.
    in_place: set[str] = field(default_factory=set)


@dataclass
class Step:
    """Everything needed to execute one node."""

    node: str
    spec: NodeSpec
    #: pin, source, what the wire does, its payload, how the pin uses the value
    wired: list[tuple[str, tuple[str, str], int, Any, int]]
    literals: dict[str, Any]
    config: dict[str, Any]
    out_types: dict[str, T.Type]
    needs: frozenset[str] = field(default_factory=frozenset)  # upstream node ids
    structure: Optional[StructurePlan] = None
    boundary: str = ""           # "in" or "out" for the boundary nodes of a frame
    #: Same inputs, same outputs, no side effects (for a structure: everything in it).
    pure: bool = True
    #: Worth another thread: a library call, or a structure that contains one.
    heavy: bool = False
    #: Measured run time in seconds, smoothed; negative while unknown.
    cost: float = -1.0
    #: For the cache: a fingerprint of everything fixed about this node (its
    #: function, configuration and literals).  Empty: cannot be cached.
    #: ``None``: not worked out yet.
    identity: Optional[bytes] = field(default=None, repr=False)
    #: Runs in a worker process.
    remote: bool = False
    #: May receive an object with reference semantics, which only one node
    #: may use at a time.
    guarded: bool = False


@dataclass
class _Schedule:
    """What the parallel scheduler needs to know about a diagram."""

    pending: list[int]                    # per step: how many steps it waits for
    consumers: list[tuple[int, ...]]      # per step: the steps waiting for it


@dataclass
class Plan:
    """A checked graph, ready to run any number of times."""

    name: str
    steps: list[Step]
    analysis: Analysis
    #: Set when two heavy nodes could run at the same time; ``None`` means
    #: running the steps in order loses nothing.
    schedule: Optional[_Schedule] = None

    @property
    def pure(self) -> bool:
        return all(step.pure for step in self.steps)

    @property
    def heavy(self) -> bool:
        return any(step.heavy for step in self.steps)


class RunResult:
    """Values produced by a run, addressed by node id and output pin."""

    def __init__(self, values: dict[tuple[str, str], Any],
                 outputs: dict[str, tuple[str, ...]], durations: dict[str, float]):
        self.values = values
        self.durations = durations
        self._outputs = outputs

    def __getitem__(self, key: Union[str, tuple[str, str]]) -> Any:
        """``result["node"]`` (single output) or ``result["node", "pin"]``."""
        if isinstance(key, tuple):
            return self.values[key]
        pins = self._outputs[key]
        if len(pins) == 1:
            return self.values[(key, pins[0])]
        return {pin: self.values[(key, pin)] for pin in pins}

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple):
            return key in self.values
        return key in self._outputs

    def outputs(self, node: str) -> dict[str, Any]:
        """All outputs of one node, by pin name."""
        return {pin: self.values[(node, pin)] for pin in self._outputs[node]}

    def observed(self, node: str) -> dict[str, T.Type]:
        """The types the outputs of a node actually had in this run."""
        return {pin: T.type_of(self.values[(node, pin)]) for pin in self._outputs[node]}

    def __repr__(self) -> str:
        return f"<RunResult: {len(self._outputs)} nodes>"


# --------------------------------------------------------------------------
# Compiling
# --------------------------------------------------------------------------

def compile_graph(graph: Graph, registry: Optional[Registry] = None) -> Plan:
    """Check a graph and turn it into an execution plan.

    Raises :class:`~openramen.check.GraphCheckError` if the graph has errors.
    """
    analysis = analyze(graph, registry)
    analysis.raise_for_errors()
    return _plan(graph, analysis)


def _plan(graph: Graph, analysis: Analysis) -> Plan:
    steps: list[Step] = []
    needs: dict[str, frozenset[str]] = {}
    for nid in analysis.order:
        info = analysis.nodes[nid]
        node = graph.nodes[nid]
        assert info.spec is not None
        wired: list[tuple[str, tuple[str, str], int, Any, int]] = []
        upstream: set[str] = set()
        for pin, (wire, compat) in info.wires.items():
            mode, payload = _PASS, None  # type: tuple[int, Any]
            if compat.verdict is T.Verdict.COERCION:
                mode, payload = _CONVERT, compat.convert
            elif compat.verdict is T.Verdict.DYNAMIC:
                mode, payload = _ACCEPT, info.in_types[pin]
            governing = info.spec.pin_for(pin)
            use = _MUTATE if governing is not None and governing.access == "mutate" else _READ
            wired.append((pin, (wire.src, wire.src_pin), mode, payload, use))
            upstream.add(wire.src)
            upstream |= needs[wire.src]
        needs[nid] = frozenset(upstream)
        structure = None
        if isinstance(node, Structure):
            frames = [_plan(frame, frame_analysis)
                      for frame, frame_analysis in zip(node.frames, info.frames)]
            structure = StructurePlan(node, frames, info.data)
            if node.kind in ("for", "while"):
                structure.in_place = _in_place_registers(frames[0], node.registers)
        boundary = {FRAME_IN: "in", FRAME_OUT: "out"}.get(info.spec.id, "")
        if structure is None:
            pure = info.spec.pure
            heavy = (info.spec.executor != "inline" or _EVERYTHING_HEAVY) and not boundary
        else:
            # a With structure opens and closes a resource: that is a side effect
            pure = structure.node.kind != "with" and all(f.pure for f in structure.frames)
            heavy = not _GIL or _EVERYTHING_HEAVY or any(f.heavy for f in structure.frames)
        guarded = structure is None and not boundary and (
            any(_may_be_object(analysis.nodes[wire.src].out_types.get(wire.src_pin, T.ANY))
                for wire, _ in info.wires.values())
            or any(not isinstance(value, _VALUES) for value in info.literals.values()))
        steps.append(Step(nid, info.spec, wired, dict(info.literals), dict(info.config),
                          dict(info.out_types), needs[nid], structure, boundary, pure, heavy,
                          remote=structure is None and _is_remote(info.spec), guarded=guarded))
    return Plan(graph.name, steps, analysis, _schedule(steps))


def _may_be_object(t: T.Type) -> bool:
    """Whether a wire of this type can carry an object with reference
    semantics (as opposed to a number, text, an array or a container)."""
    if isinstance(t, (T.ScalarT, T.ArrayT, T.EnumT)):
        return False
    if isinstance(t, (T.ListT, T.IteratorT)):
        return _may_be_object(t.elem)
    if isinstance(t, T.TupleT):
        return any(_may_be_object(e) for e in t.elems)
    if isinstance(t, T.DictT):
        return _may_be_object(t.key) or _may_be_object(t.value)
    if isinstance(t, T.UnionT):
        return any(_may_be_object(o) for o in t.options)
    return True                 # Any, type variables, records, objects, callables


def _is_remote(spec: NodeSpec) -> bool:
    """Whether a node runs in a worker process.  Objects with reference
    semantics cannot leave this process, so a node with such a pin stays."""
    if spec.executor != "process":
        return False
    return not any(isinstance(pin.type, (T.ObjectT, T.CallableT, T.IteratorT))
                   for pin in spec.inputs + spec.outputs + spec.config)


def _schedule(steps: list[Step]) -> Optional[_Schedule]:
    """Dependency counts for running ``steps`` in parallel, or ``None`` if no
    two heavy steps could ever run at the same time.

    Nodes with side effects are chained in their order, so they never
    overtake each other.
    """
    if sum(step.heavy for step in steps) < 2:
        return None
    index = {step.node: i for i, step in enumerate(steps)}
    depends: list[set[int]] = []
    upstream: list[set[int]] = []
    impure: Optional[int] = None
    for i, step in enumerate(steps):
        direct = {index[source[0]] for _, source, _, _, _ in step.wired if source[0] in index}
        if not step.pure:
            if impure is not None:
                direct.add(impure)
            impure = i
        depends.append(direct)
        closure = set(direct)
        for j in direct:
            closure |= upstream[j]
        upstream.append(closure)
    heavy = [i for i, step in enumerate(steps) if step.heavy]
    if all(a in upstream[b] for k, b in enumerate(heavy) for a in heavy[:k]):
        return None
    consumers: list[list[int]] = [[] for _ in steps]
    for i, direct in enumerate(depends):
        for j in direct:
            consumers[j].append(i)
    return _Schedule([len(direct) for direct in depends], [tuple(c) for c in consumers])


def _in_place_registers(frame: Plan, registers: list[str]) -> set[str]:
    """Shift registers that follow the accumulate-in-a-buffer pattern.

    That is: the register's value goes to exactly one pin, that pin belongs to
    a node which writes to it in place, and the node's output for that very
    object goes nowhere but back into the register.  Then no other node can
    ever see the buffer, and the copy per iteration can be skipped.
    """
    consumers: dict[tuple[str, str], list[tuple[Step, int]]] = {}
    for step in frame.steps:
        for index, (_pin, source, _mode, _payload, _use) in enumerate(step.wired):
            consumers.setdefault(source, []).append((step, index))
    found: set[str] = set()
    for register in registers:
        readers = consumers.get((INPUTS, register), [])
        if len(readers) != 1:
            continue
        step, index = readers[0]
        pin, source, mode, payload, use = step.wired[index]
        if use != _MUTATE or mode != _PASS or step.structure is not None or step.boundary:
            continue
        handed_on = [p.name for p in step.spec.outputs if p.aliases == pin]
        if len(handed_on) != 1:
            continue
        back = consumers.get((step.node, handed_on[0]), [])
        if len(back) != 1 or back[0][0].boundary != "out":
            continue
        back_pin, _, back_mode, _, _ = back[0][0].wired[back[0][1]]
        if back_pin != register or back_mode != _PASS:
            continue
        step.wired[index] = (pin, source, mode, payload, _IN_PLACE)
        found.add(register)
    return found


# --------------------------------------------------------------------------
# Value semantics
# --------------------------------------------------------------------------

def _fresh(value: Any) -> Any:
    """A private copy of a literal, so no node can alter the graph's own."""
    if isinstance(value, _IMMUTABLE):
        return value
    try:
        return copy.deepcopy(value)
    except Exception:
        return value      # an object that cannot be copied has reference semantics anyway


def _private(value: Any) -> Any:
    """A copy the receiving node may freely modify."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (list, dict, set, bytearray, tuple)):
        return copy.deepcopy(value)
    return value      # immutable, or an object with reference semantics


def _frozen(value: np.ndarray) -> np.ndarray:
    """A read-only view: the node can look, but a write raises."""
    view = value.view()
    view.flags.writeable = False
    return view


def _same(a: Any, b: Any) -> bool:
    """Deep equality for the mutation detector; unknown objects count as equal."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.shape == b.shape and a.dtype == b.dtype
                and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")))
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (set, bytearray, bool, int, str, bytes, complex)):
        return bool(a == b)
    if isinstance(a, float):
        return a == b or (a != a and b != b)
    return True


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

#: A step that has proved cheaper than this (seconds) is run in the
#: scheduling thread: handing it to a pool would cost more than it saves.
_CHEAP = 200e-6
_GIL = getattr(sys, "_is_gil_enabled", lambda: True)()
#: For tests: plan every node as if it were worth a thread of its own.
_EVERYTHING_HEAVY = False

_POOLS: dict[int, ThreadPoolExecutor] = {}
_COORDINATORS: list[ThreadPoolExecutor] = []
_POOL_LOCK = threading.Lock()
#: Threads that run the frames of structures.  They are taken only when one
#: is free and never waited for, so nested structures cannot starve each other.
_SLOTS = threading.BoundedSemaphore(64)
#: One lock per library whose nodes are not thread-safe.
_LIBRARY_LOCKS: dict[str, threading.RLock] = {}
#: Locks for objects with reference semantics: two nodes never use the same
#: object at the same time.  Striped by address, so nothing is stored per object.
_STRIPES = [threading.Lock() for _ in range(64)]
_VALUES = _IMMUTABLE + (np.ndarray, np.generic, list, dict, tuple, set, frozenset,
                        bytearray, enum.Enum)


def default_workers() -> int:
    """How many nodes run at the same time unless ``run`` is told otherwise."""
    return max(2, min(8, os.cpu_count() or 2))


def _pool(workers: int) -> ThreadPoolExecutor:
    pool = _POOLS.get(workers)
    if pool is None:
        with _POOL_LOCK:
            pool = _POOLS.get(workers)
            if pool is None:
                pool = _POOLS[workers] = ThreadPoolExecutor(
                    workers, thread_name_prefix="openramen-node")
    return pool


def _coordinators() -> ThreadPoolExecutor:
    if not _COORDINATORS:
        with _POOL_LOCK:
            if not _COORDINATORS:
                _COORDINATORS.append(ThreadPoolExecutor(64, thread_name_prefix="openramen-frame"))
    return _COORDINATORS[0]


def _forget_pools() -> None:
    """Threads do not survive a fork: a child process starts its pools afresh."""
    global _POOL_LOCK, _SLOTS, _STRIPES
    _POOLS.clear()
    _COORDINATORS.clear()
    _LIBRARY_LOCKS.clear()
    _POOL_LOCK = threading.Lock()
    _SLOTS = threading.BoundedSemaphore(64)
    _STRIPES = [threading.Lock() for _ in range(64)]


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_pools)


def _library_lock(spec_id: str) -> threading.RLock:
    library = spec_id.split(".", 1)[0]
    lock = _LIBRARY_LOCKS.get(library)
    if lock is None:
        with _POOL_LOCK:
            lock = _LIBRARY_LOCKS.setdefault(library, threading.RLock())
    return lock


class _Aborted(BaseException):
    """Raised inside a frame that is being abandoned because a sibling failed.
    Never reaches the caller of :func:`run`."""


class _Scope:
    """One running frame.  Marked dead when one of its nodes fails, so that
    whatever is still running inside it stops at the next node."""

    __slots__ = ("parent", "dead")

    def __init__(self, parent: Optional["_Scope"]):
        self.parent = parent
        self.dead = False

    def stopped(self) -> bool:
        scope: Optional[_Scope] = self
        while scope is not None:
            if scope.dead:
                return True
            scope = scope.parent
        return False


def _first_error(failures: dict[int, BaseException]) -> BaseException:
    """The error to report when several things failed at once."""
    real = {i: exc for i, exc in failures.items() if not isinstance(exc, _Aborted)}
    if not real:
        return _Aborted()
    for exc in real.values():
        if not isinstance(exc, (NodeError, Cancelled)):
            return exc                      # KeyboardInterrupt and the like
    for exc in real.values():
        if isinstance(exc, Cancelled):
            return exc
    return real[min(real)]


@dataclass
class _Context:
    emit: Listener
    check_outputs: Optional[bool]
    cancel: Optional[Callable[[], bool]]
    detect_mutation: bool
    workers: int = 1
    cache: Optional[Cache] = None
    #: With a cache: the token of every value on a top-level wire.
    tokens: dict[tuple[str, str], bytes] = field(default_factory=dict)


def run(graph: Union[Graph, Plan], registry: Optional[Registry] = None, *,
        targets: Optional[Iterable[str]] = None,
        listener: Optional[Listener] = None,
        check_outputs: Optional[bool] = None,
        cancel: Optional[Callable[[], bool]] = None,
        detect_mutation: bool = False,
        workers: Optional[int] = None,
        cache: Optional[Cache] = None) -> RunResult:
    """Run a graph (or a compiled :class:`Plan`) and return its values.

    * ``targets``: only run these nodes and what they depend on.
    * ``listener``: called with an :class:`Event` as the run progresses, one
      call at a time.
    * ``check_outputs``: verify each output against its declared type.  By
      default this is done for every node that is not hand-curated.
    * ``cancel``: called before every node and loop iteration, possibly from
      several threads; when it returns true the run stops with
      :class:`Cancelled`.
    * ``detect_mutation``: debugging aid.  Compare the lists and dicts given
      to each node before and after the call, and fail if a node changed one
      without declaring it.  (Arrays are always protected.)
    * ``workers``: how many nodes may run at the same time.  Independent
      branches run in parallel by default; ``workers=1`` runs one node at a
      time, in order.
    * ``cache``: a :class:`~openramen.runtime.cache.Cache`.  Pure nodes whose
      inputs have not changed since an earlier run with the same cache are
      not run again.
    """
    plan = graph if isinstance(graph, Plan) else compile_graph(graph, registry)
    steps = plan.steps
    schedule = plan.schedule
    if targets is not None:
        wanted = set(targets)
        known = {s.node for s in steps}
        missing = wanted - known
        if missing:
            raise KeyError(f"no node {sorted(missing)[0]!r} in graph")
        keep = set(wanted)
        for step in steps:
            if step.node in wanted:
                keep |= step.needs
        steps = [s for s in steps if s.node in keep]
        schedule = _schedule(steps)

    workers = default_workers() if workers is None else max(1, int(workers))
    emit: Listener = listener or (lambda event: None)
    if listener is not None and workers > 1:
        lock = threading.Lock()

        def emit(event: Event, _send: Listener = listener) -> None:
            with lock:
                _send(event)

    context = _Context(emit, check_outputs, cancel, detect_mutation, workers, cache)
    durations: dict[str, float] = {}
    emit(Event("graph-started"))
    started = time.perf_counter()
    try:
        values, _ = _run_steps(steps, schedule, context, "", None, durations)
    except BaseException as exc:
        emit(Event("graph-failed", duration=time.perf_counter() - started, error=exc))
        raise
    emit(Event("graph-finished", duration=time.perf_counter() - started))
    outputs = {s.node: tuple(p.name for p in s.spec.outputs) for s in steps}
    return RunResult(values, outputs, durations)


class Run:
    """A graph running in the background, as returned by :func:`start`."""

    def __init__(self, plan: Plan, options: dict[str, Any]):
        self.plan = plan
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._result: Optional[RunResult] = None
        self._error: Optional[BaseException] = None
        also = options.pop("cancel", None)
        stop = self._stop.is_set
        options["cancel"] = stop if also is None else (lambda: stop() or bool(also()))
        self._thread = threading.Thread(target=self._work, args=(options,), daemon=True,
                                        name=f"openramen-run-{plan.name or 'graph'}")
        self._thread.start()

    def _work(self, options: dict[str, Any]) -> None:
        try:
            self._result = run(self.plan, **options)
        except BaseException as exc:
            self._error = exc
        finally:
            self._finished.set()

    def cancel(self) -> None:
        """Ask the run to stop.  It does so before the next node or loop
        iteration; a node that is already running finishes first."""
        self._stop.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until the run is over; false if ``timeout`` seconds passed first."""
        return self._finished.wait(timeout)

    @property
    def done(self) -> bool:
        return self._finished.is_set()

    @property
    def cancelled(self) -> bool:
        """True once the run has stopped because it was cancelled."""
        return isinstance(self._error, Cancelled)

    @property
    def error(self) -> Optional[BaseException]:
        """What stopped the run, if it did not finish; ``None`` while it runs."""
        return self._error

    def result(self, timeout: Optional[float] = None) -> RunResult:
        """The values of the run.  Waits for it to finish; raises what stopped
        it (:class:`NodeError`, :class:`Cancelled`), or ``TimeoutError``."""
        if not self._finished.wait(timeout):
            raise TimeoutError("the run has not finished yet")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def __repr__(self) -> str:
        state = "running" if not self.done else "cancelled" if self.cancelled else (
            "failed" if self._error is not None else "finished")
        return f"<Run of {self.plan.name or 'graph'!r}: {state}>"


def start(graph: Union[Graph, Plan], registry: Optional[Registry] = None,
          **options: Any) -> Run:
    """Run a graph in the background and return at once.

    The graph is checked first, so a graph with errors raises here.  Options
    are those of :func:`run`.  The returned :class:`Run` can be cancelled,
    waited for, and asked for its result.
    """
    plan = graph if isinstance(graph, Plan) else compile_graph(graph, registry)
    return Run(plan, options)


def _run_steps(steps: list[Step], schedule: Optional[_Schedule], context: _Context, path: str,
               boundary: Optional[dict[str, Any]],
               durations: Optional[dict[str, float]] = None,
               scope: Optional[_Scope] = None
               ) -> tuple[dict[tuple[str, str], Any], dict[str, Any]]:
    """Run one diagram.  For a frame, ``boundary`` feeds its ``inputs`` node
    and the second result is what arrived at its ``outputs`` node."""
    if schedule is not None and context.workers > 1:
        return _run_parallel(steps, schedule, context, path, boundary, durations, scope)
    values: dict[tuple[str, str], Any] = {}
    result: dict[str, Any] = {}
    emit = context.emit
    for step in steps:
        if context.cancel is not None and context.cancel():
            raise Cancelled("the run was cancelled")
        if scope is not None and scope.stopped():
            raise _Aborted()
        if step.boundary == "in":
            for pin, value in (boundary or {}).items():
                values[(step.node, pin)] = value
            continue
        if step.boundary == "out":
            result = _gather(step, values, path, protect=False)
            continue
        name = path + step.node
        emit(Event("node-started", name))
        t0 = time.perf_counter()
        try:
            results = _compute(step, values, context, path, scope)
        except NodeError as exc:
            emit(Event("node-failed", name, time.perf_counter() - t0, exc))
            raise
        for pin, value in results.items():
            values[(step.node, pin)] = value
        elapsed = time.perf_counter() - t0
        if durations is not None:
            durations[step.node] = elapsed
        emit(Event("node-finished", name, elapsed))
    return values, result


def _task(step: Step, values: dict[tuple[str, str], Any], context: _Context, path: str,
          scope: _Scope) -> tuple[dict[str, Any], float]:
    """Run one node of a diagram that is being run in parallel."""
    if context.cancel is not None and context.cancel():
        raise Cancelled("the run was cancelled")
    if scope.stopped():
        raise _Aborted()
    name = path + step.node
    context.emit(Event("node-started", name))
    t0 = time.perf_counter()
    try:
        results = _compute(step, values, context, path, scope)
    except NodeError as exc:
        context.emit(Event("node-failed", name, time.perf_counter() - t0, exc))
        raise
    elapsed = time.perf_counter() - t0
    context.emit(Event("node-finished", name, elapsed))
    return results, elapsed


def _in_slot(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    finally:
        _SLOTS.release()


def _dispatch(step: Step, values: dict[tuple[str, str], Any], context: _Context, path: str,
              scope: _Scope) -> "Future[tuple[dict[str, Any], float]]":
    """Start a node on another thread."""
    if step.structure is None:
        return _pool(context.workers).submit(_task, step, values, context, path, scope)
    if _SLOTS.acquire(blocking=False):
        return _coordinators().submit(_in_slot, _task, step, values, context, path, scope)
    future: Future[tuple[dict[str, Any], float]] = Future()   # no thread to spare: run it here
    try:
        future.set_result(_task(step, values, context, path, scope))
    except BaseException as exc:
        future.set_exception(exc)
    return future


def _run_parallel(steps: list[Step], schedule: _Schedule, context: _Context, path: str,
                  boundary: Optional[dict[str, Any]], durations: Optional[dict[str, float]],
                  scope: Optional[_Scope]
                  ) -> tuple[dict[tuple[str, str], Any], dict[str, Any]]:
    """Run one diagram, starting every node as soon as its inputs are there.

    Cheap nodes run right here; the others go to the thread pool whenever
    more than one thing could run at the same time.
    """
    mine = _Scope(scope)
    values: dict[tuple[str, str], Any] = {}
    result: dict[str, Any] = {}
    pending = list(schedule.pending)
    consumers = schedule.consumers
    ready = [i for i, count in enumerate(pending) if not count]
    in_flight: dict[Future[tuple[dict[str, Any], float]], int] = {}
    failures: dict[int, BaseException] = {}
    current = -1

    def release(i: int) -> None:
        for consumer in consumers[i]:
            pending[consumer] -= 1
            if not pending[consumer]:
                heapq.heappush(ready, consumer)

    def done(i: int, outcome: tuple[dict[str, Any], float]) -> None:
        results, elapsed = outcome
        step = steps[i]
        for pin, value in results.items():
            values[(step.node, pin)] = value
        step.cost = elapsed if step.cost < 0 else 0.7 * step.cost + 0.3 * elapsed
        if durations is not None:
            durations[step.node] = elapsed
        release(i)

    try:
        while ready or in_flight:
            heavy: list[int] = []
            while ready:
                current = i = heapq.heappop(ready)
                step = steps[i]
                if step.boundary == "in":
                    for pin, value in (boundary or {}).items():
                        values[(step.node, pin)] = value
                    release(i)
                elif step.boundary == "out":
                    result = _gather(step, values, path, protect=False)
                    release(i)
                elif step.heavy and not 0 <= step.cost < _CHEAP:
                    heavy.append(i)
                else:
                    done(i, _task(step, values, context, path, mine))
            if len(heavy) == 1 and not in_flight:
                # nothing else could run meanwhile: no need for another thread
                current = heavy[0]
                done(current, _task(steps[current], values, context, path, mine))
                continue
            for i in heavy:
                current = i
                in_flight[_dispatch(steps[i], values, context, path, mine)] = i
            if in_flight:
                finished, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for future in sorted(finished, key=in_flight.__getitem__):
                    i = in_flight.pop(future)
                    try:
                        outcome = future.result()
                    except BaseException as exc:
                        failures[i] = exc
                    else:
                        done(i, outcome)
                if failures:
                    break
    except BaseException as exc:
        failures[current] = exc
    if failures:
        mine.dead = True
        for future in in_flight:
            future.cancel()
        if in_flight:
            wait(list(in_flight))
            for future, i in in_flight.items():
                if not future.cancelled():
                    error = future.exception()
                    if error is not None:
                        failures.setdefault(i, error)
        raise _first_error(failures)
    return values, result


def _gather(step: Step, values: dict[tuple[str, str], Any], path: str,
            protect: bool = True) -> dict[str, Any]:
    """The arguments of a node: literals, and what its wires deliver."""
    spec = step.spec
    arguments = {pin: _fresh(value) for pin, value in step.literals.items()}
    copies = protect and spec.id in _NEEDS_WRITABLE
    for pin, source, mode, payload, use in step.wired:
        value = values[source]
        if mode == _CONVERT:
            try:
                value = payload(value)
            except Exception as exc:
                raise NodeError(path + step.node, spec.id, exc, pin) from exc
        elif mode == _ACCEPT:
            try:
                value = T.accept(value, payload)
            except T.TypeMismatch as exc:
                raise NodeError(path + step.node, spec.id, exc, pin, str(exc)) from exc
        if protect:
            if use == _MUTATE:
                value = _private(value)
            elif use == _READ and isinstance(value, np.ndarray):
                if copies:
                    value = value.copy()
                elif value.flags.writeable:
                    value = _frozen(value)
        arguments[pin] = value
    return arguments


def _identity(step: Step) -> bytes:
    """Fingerprint of what is fixed about a step, or ``b""`` if it has none."""
    if step.identity is not None:
        return step.identity
    spec = step.spec
    parts = [spec.id.encode(), spec.version.encode()]
    known = True
    for label, fixed in (("literal", step.literals), ("config", step.config)):
        for name in sorted(fixed):
            printed = _cache.fingerprint(fixed[name])
            if printed is None:
                known = False
                break
            parts += [f"{label}:{name}".encode(), printed]
    parts += [f"{pin}:{mode}".encode() for pin, _, mode, _, _ in step.wired]
    if known and step.structure is not None:
        try:
            parts.append(repr(_structure_to_dict(step.structure.node)).encode("utf-8", "replace"))
        except Exception:
            known = False                  # holds a value that cannot be written down
        for frame in step.structure.frames:
            for inner in frame.steps:
                if not inner.boundary:
                    known = known and bool(_identity(inner))
                    parts.append(inner.identity or b"")
    elif known:
        try:
            parts.append(_cache.function_identity(spec.resolve()))
        except Exception:
            known = False
    step.identity = _cache.digest(*parts) if known else b""
    return step.identity


def _compute(step: Step, values: dict[tuple[str, str], Any], context: _Context,
             path: str, scope: Optional[_Scope]) -> dict[str, Any]:
    """Run one node and return its outputs, normalised and verified.  At the
    top level of a run with a cache, pure nodes are looked up first."""
    cache = context.cache
    if cache is None or path:
        return _compute_now(step, values, context, path, scope)
    tokens = context.tokens
    key = b""
    if step.pure and _identity(step):
        try:
            key = _cache.digest(step.identity or b"",
                                *(tokens[source] for _, source, _, _, _ in step.wired))
        except KeyError:
            key = b""
    if key:
        found = cache.get(key)
        if found is not None:
            results, kept = found
            for pin, token in kept.items():
                tokens[(step.node, pin)] = token
            context.emit(Event("node-cached", step.node))
            return results
    results = _compute_now(step, values, context, path, scope)
    made: dict[str, bytes] = {}
    for pin, value in results.items():
        token = _cache.content_token(value) or (
            _cache.digest(key, pin.encode()) if key else _cache.unique_token())
        made[pin] = tokens[(step.node, pin)] = token
    if key:
        cache.put(key, results, made)
    return results


def _compute_now(step: Step, values: dict[tuple[str, str], Any], context: _Context,
                 path: str, scope: Optional[_Scope]) -> dict[str, Any]:
    spec = step.spec
    name = path + step.node
    arguments = _gather(step, values, path)
    if step.structure is not None:
        results = _run_structure(step, arguments, context, path, scope)
    elif spec.thread_safe and not (step.guarded and context.workers > 1):
        results = _call_unlocked(step, arguments, context, name)
    else:
        results = _call(step, arguments, context, name)
    verify = spec.tier != "curated" if context.check_outputs is None else context.check_outputs
    for pin, value in results.items():
        results[pin] = value = T.plain_scalar(value)
        if verify and not T.conforms(value, step.out_types[pin]):
            declared = spec.output(pin)
            origin = f" (a type taken from its {declared.source})" if (
                declared is not None and declared.source in ("docstring", "annotation")) else ""
            mismatch = T.TypeMismatch(
                f"output is declared {step.out_types[pin]}{origin} but the node returned "
                f"{T.type_of(value)}")
            raise NodeError(name, spec.id, mismatch, pin, str(mismatch))
    return results


def _call(step: Step, arguments: dict[str, Any], context: _Context,
          name: str) -> dict[str, Any]:
    """Call the function of a node, holding the locks it needs."""
    spec = step.spec
    held: list[Any] = []
    if not spec.thread_safe:
        held.append(_library_lock(spec.id))
    if step.guarded and context.workers > 1:
        stripes = {(id(value) >> 4) % 64 for value in arguments.values()
                   if not isinstance(value, _VALUES)}
        held.extend(_STRIPES[index] for index in sorted(stripes))
    if not held:
        return _call_unlocked(step, arguments, context, name)
    for lock in held:
        lock.acquire()
    try:
        return _call_unlocked(step, arguments, context, name)
    finally:
        for lock in reversed(held):
            lock.release()


def _call_remote(step: Step, arguments: dict[str, Any], config: dict[str, Any],
                 context: _Context, name: str) -> Optional[dict[str, Any]]:
    """Call a node in a worker process.  ``None``: it has to run here instead."""
    spec = step.spec
    try:
        results, needed_writable = _workers.call(spec, arguments, config, context.cancel)
    except _workers.Unsendable as exc:
        _LOCAL_ONLY.add(spec.id)
        context.emit(Event("node-warning", name,
                           message=f"{spec.id} runs in this process, not in a worker: {exc}"))
        return None
    except _workers.Stopped:
        raise Cancelled("the run was cancelled") from None
    except _workers.RemoteError as exc:
        raise NodeError(name, spec.id, exc.original) from exc
    except _workers.WorkerDied as exc:
        raise NodeError(name, spec.id, exc) from exc
    if needed_writable:
        _NEEDS_WRITABLE.add(spec.id)
        context.emit(Event("node-warning", name,
                           message=f"{spec.id} needs writable arrays; its array inputs "
                                   "are given private copies"))
    return results


def _call_unlocked(step: Step, arguments: dict[str, Any], context: _Context,
                   name: str) -> dict[str, Any]:
    spec = step.spec
    config = {key: _fresh(value) for key, value in step.config.items()}
    if step.remote and spec.id not in _LOCAL_ONLY:
        remote = _call_remote(step, arguments, config, context, name)
        if remote is not None:
            return remote
    watched: dict[str, Any] = {}
    if context.detect_mutation and (spec.id, spec.version) not in _VERIFIED:
        declared = {pin for pin, _, _, _, use in step.wired if use != _READ}
        for pin, value in arguments.items():
            governing = spec.pin_for(pin)
            writes = governing is not None and governing.access == "mutate"
            if pin not in declared and not writes and isinstance(
                    value, (list, dict, set, bytearray)):
                try:
                    watched[pin] = copy.deepcopy(value)
                except Exception:
                    pass          # cannot be snapshotted: cannot be watched
    try:
        results = spec.call(arguments, config)
    except (ValueError, TypeError, BufferError, RuntimeError) as exc:
        frozen = [pin for pin, value in arguments.items()
                  if isinstance(value, np.ndarray) and not value.flags.writeable]
        if not frozen or not _READ_ONLY.search(str(exc)):
            raise NodeError(name, spec.id, exc) from exc
        # The node needs writable arrays: either it writes to an input without
        # declaring it, or it merely refuses read-only buffers.  Private
        # copies are right in both cases.
        retry = dict(arguments)
        for pin in frozen:
            retry[pin] = arguments[pin].copy()
        try:
            results = spec.call(retry, config)
        except Exception as again:
            raise NodeError(name, spec.id, again) from again
        _NEEDS_WRITABLE.add(spec.id)
        context.emit(Event(
            "node-warning", name,
            message=f"{spec.id} needs writable arrays ({exc}); its inputs "
                    f"{', '.join(repr(p) for p in sorted(frozen))} are given private copies"))
    except Exception as exc:
        raise NodeError(name, spec.id, exc) from exc
    for pin, before in watched.items():
        if not _same(before, arguments[pin]):
            problem = RuntimeError(
                f"the node modified its input {pin!r} in place without declaring it; "
                "mark the input as mutated (in an override: mutates or passthrough)")
            raise NodeError(name, spec.id, problem, pin, str(problem))
    if context.detect_mutation:
        _VERIFIED.add((spec.id, spec.version))
    return results


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------

def _fail(name: str, spec: str, message: str, pin: Optional[str] = None) -> NodeError:
    problem = RuntimeError(message)
    return NodeError(name, spec, problem, pin, message)


def _elements(value: Any) -> Iterator[Any]:
    """What an indexing input tunnel hands out, one item per iteration."""
    if isinstance(value, (dict, str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"a {T.type_of(value)} cannot be indexed by a loop")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        raise TypeError("a 0-d array cannot be indexed by a loop")
    return iter(value)


_Frame = Callable[..., dict[str, Any]]


def _run_structure(step: Step, arguments: dict[str, Any], context: _Context,
                   path: str, scope: Optional[_Scope]) -> dict[str, Any]:
    plan = step.structure
    assert plan is not None
    node = plan.node
    name = path + step.node
    spec_id = step.spec.id
    converts: dict[str, T.Type] = plan.data.get("convert", {})
    numbered = node.kind in ("case", "try")

    def frame(index: int, boundary: dict[str, Any], what: str,
              within: Optional[_Scope] = scope) -> dict[str, Any]:
        inner = f"{name}[{index}]/" if numbered else name + "/"
        chosen = plan.frames[index]
        try:
            _, result = _run_steps(chosen.steps, chosen.schedule, context, inner, boundary,
                                   None, within)
        except NodeError as exc:
            exc.context.append(what)
            raise
        return result

    def finish(pin: str, value: Any) -> Any:
        target = converts.get(pin)
        if target is None:
            return value
        try:
            return T.accept(value, target)
        except T.TypeMismatch as exc:
            raise NodeError(name, spec_id, exc, pin, str(exc)) from exc

    def leave(result: dict[str, Any]) -> dict[str, Any]:
        return {t.name: finish(t.name, _tunnel_value(t, result, name, spec_id))
                for t in node.outputs}

    if node.kind in ("for", "while"):
        return _run_loop(step, arguments, context, name, frame, finish, scope)

    boundary = {t.name: arguments[t.name] for t in node.inputs}
    if node.kind == "subgraph":
        return leave(frame(0, boundary, f"in {name!r}"))

    if node.kind == "case":
        selector = arguments["selector"]
        key = selector.name if isinstance(selector, enum.Enum) else selector
        index = _matching_frame(node.cases, key)
        if index is None:
            raise _fail(name, spec_id, f"no frame handles the selector value {key!r}", "selector")
        boundary["selector"] = selector
        return leave(frame(index, boundary, f"frame {index} of {name!r}"))

    if node.kind == "try":
        message = ""
        try:
            result = frame(0, boundary, f"in the body of {name!r}")
        except NodeError as exc:
            message = str(exc)
            handler = dict(boundary, error=message, error_type=type(exc.original).__name__)
            result = frame(1, handler, f"in the error handler of {name!r}")
        outputs = leave(result)
        outputs.update(failed=bool(message), error=message)
        return outputs

    # with: enter the context manager, run the frame, always leave again
    resource = arguments["resource"]
    enter = getattr(type(resource), "__enter__", None)
    leave_resource = getattr(type(resource), "__exit__", None)
    if enter is None or leave_resource is None:
        raise _fail(name, spec_id,
                    f"'resource' is a {T.type_of(resource)}, not a context manager", "resource")
    try:
        boundary["resource"] = T.plain_scalar(enter(resource))
    except Exception as exc:
        raise NodeError(name, spec_id, exc, "resource") from exc
    try:
        result = frame(0, boundary, f"in {name!r}")
    except BaseException as exc:
        cause = exc.original if isinstance(exc, NodeError) else exc
        try:
            suppressed = leave_resource(resource, type(cause), cause, cause.__traceback__)
        except Exception as closing:
            raise NodeError(name, spec_id, closing, "resource") from exc
        if not suppressed or not isinstance(exc, NodeError):
            raise
        result = {}             # the context manager swallowed the error
    else:
        try:
            leave_resource(resource, None, None, None)
        except Exception as exc:
            raise NodeError(name, spec_id, exc, "resource") from exc
    return leave(result)


def _matching_frame(cases: list, key: Any) -> Optional[int]:
    default = None
    is_bool = isinstance(key, bool)
    for index, match in enumerate(cases):
        if match is None:
            default = index if default is None else default
            continue
        for entry in match:
            if isinstance(entry, tuple):
                if isinstance(key, int) and not is_bool and entry[0] <= key <= entry[1]:
                    return index
            elif isinstance(entry, bool) == is_bool and type(entry) is type(key) and entry == key:
                return index
    return default


def _tunnel_value(tunnel: Any, result: dict[str, Any], name: str, spec_id: str) -> Any:
    if tunnel.name in result:
        return result[tunnel.name]
    if tunnel.fallback is not UNSET:
        return _fresh(tunnel.fallback)
    raise _fail(name, spec_id, f"output {tunnel.name!r} received no value", tunnel.name)


def _spread(count: int, body: Callable[[int], Any], context: _Context,
            mine: _Scope) -> list[Any]:
    """Run ``body(0) ... body(count - 1)`` on several threads, this one included."""
    results: list[Any] = [None] * count
    failures: dict[int, BaseException] = {}
    numbers = iter(range(count))
    lock = threading.Lock()

    def runner() -> None:
        while not mine.dead:
            with lock:
                i = next(numbers, None)
            if i is None:
                return
            try:
                results[i] = body(i)
            except BaseException as exc:
                failures[i] = exc
                mine.dead = True

    helpers: list[Future[Any]] = []
    for _ in range(min(context.workers, count) - 1):
        if not _SLOTS.acquire(blocking=False):
            break
        helpers.append(_coordinators().submit(_in_slot, runner))
    runner()
    if helpers:
        wait(helpers)
    if failures:
        raise _first_error(failures)
    return results


def _run_loop(step: Step, arguments: dict[str, Any], context: _Context, name: str,
              frame: _Frame, finish: Callable[[str, Any], Any],
              scope: Optional[_Scope]) -> dict[str, Any]:
    plan = step.structure
    assert plan is not None
    node = plan.node
    spec_id = step.spec.id
    register_types: dict[str, T.Type] = plan.data.get("registers", {})
    is_for = node.kind == "for"

    def into_register(register: str, value: Any) -> Any:
        wanted = register_types.get(register, T.ANY)
        if isinstance(wanted, T.AnyT) or T.conforms(value, wanted):
            return value
        try:
            return T.accept(value, wanted)
        except T.TypeMismatch as exc:
            raise NodeError(name, spec_id, exc, register,
                            f"shift register {register!r}: {exc}") from exc

    plain = {t.name: arguments[t.name] for t in node.inputs if t.mode == "plain"}
    indexed: dict[str, Iterator[Any]] = {}
    for tunnel in node.inputs:
        if tunnel.mode == "index":
            try:
                indexed[tunnel.name] = _elements(arguments[tunnel.name])
            except TypeError as exc:
                raise NodeError(name, spec_id, exc, tunnel.name) from exc
    count = arguments.get("count") if is_for else None
    if is_for and count is None and not indexed:
        raise _fail(name, spec_id, "a For loop needs a count or an indexing input", "count")

    last: dict[str, Any] = {}
    gathered: dict[str, list[Any]] = {t.name: [] for t in node.outputs if t.mode != "last"}

    def keep(result: dict[str, Any]) -> None:
        for tunnel in node.outputs:
            if tunnel.name not in result:
                continue
            if tunnel.mode == "last":
                last[tunnel.name] = result[tunnel.name]
            else:
                gathered[tunnel.name].append(result[tunnel.name])

    registers: dict[str, Any] = {}
    if node.parallel and context.workers > 1 and not node.registers:
        # every iteration is independent: work out what each one receives,
        # then run them on several threads and keep the results in order
        rows: list[tuple[Any, ...]]
        if indexed:
            columns = zip(*indexed.values())
            rows = list(columns if count is None else itertools.islice(columns, max(count, 0)))
        else:
            rows = [()] * max(count or 0, 0)
        pins = list(indexed)
        mine = _Scope(scope)

        def iteration(i: int) -> dict[str, Any]:
            if context.cancel is not None and context.cancel():
                raise Cancelled("the run was cancelled")
            boundary = dict(plain)
            for pin, value in zip(pins, rows[i]):
                boundary[pin] = T.plain_scalar(value)
            boundary["i"] = i
            return frame(0, boundary, f"iteration {i} of {name!r}", mine)

        for result in _spread(len(rows), iteration, context, mine):
            keep(result)
    else:
        owned: dict[str, Any] = {}        # the buffers of in-place registers
        for register in node.registers:
            value = into_register(register, arguments[register])
            if register in plan.in_place:
                value = owned[register] = _private(value)
            registers[register] = value
        limit = node.max_iterations
        i = 0
        while count is None or i < count:
            if context.cancel is not None and context.cancel():
                raise Cancelled("the run was cancelled")
            boundary = dict(plain)
            try:
                for pin, iterator in indexed.items():
                    boundary[pin] = T.plain_scalar(next(iterator))
            except StopIteration:
                break
            for register in plan.in_place:
                if registers[register] is not owned.get(register):
                    # the loop fed back some other object: take a buffer of our own again
                    registers[register] = owned[register] = _private(registers[register])
            boundary.update(registers)
            boundary["i"] = i
            result = frame(0, boundary, f"iteration {i} of {name!r}")
            for register in node.registers:
                registers[register] = into_register(register, result[register])
            keep(result)
            i += 1
            if not is_for:
                stop = result.get("stop")
                if not isinstance(stop, (bool, np.bool_)):
                    raise _fail(name, spec_id,
                                f"'stop' must be a Bool, got {T.type_of(stop)}", "stop")
                if stop:
                    break
                if limit is not None and i >= limit:
                    raise _fail(name, spec_id,
                                f"the loop did not stop within {limit} iterations", "stop")

    outputs: dict[str, Any] = {}
    for tunnel in node.outputs:
        mode, element, _out_type = plan.data["outputs"][tunnel.name]
        if mode == "last":
            if tunnel.name in last:
                value = last[tunnel.name]
            elif tunnel.fallback is not UNSET:
                value = _fresh(tunnel.fallback)
            else:
                raise _fail(
                    name, spec_id,
                    f"the loop ran zero times, so output {tunnel.name!r} has no value; "
                    "give it a fallback or use a shift register", tunnel.name)
        else:
            try:
                value = _combine(mode, gathered[tunnel.name], element, _out_type)
            except Exception as exc:
                raise NodeError(name, spec_id, exc, tunnel.name) from exc
        outputs[tunnel.name] = finish(tunnel.name, value)
    outputs.update(registers)
    return outputs


_STACK_DTYPE = {T.BOOL: "bool", T.INT: "int64", T.FLOAT: "float64", T.COMPLEX: "complex128"}


def _empty(element: T.Type, extra_dimensions: int) -> np.ndarray:
    """The array a collecting tunnel gives when the loop ran zero times."""
    dtype, ndim = "float64", 1
    if isinstance(element, T.ArrayT):
        dtype = element.dtype if isinstance(element.dtype, str) else "float64"
        ndim = (element.ndim if isinstance(element.ndim, int) else 0) + extra_dimensions
    return np.empty((0,) * max(ndim, 1), dtype=dtype)


def _combine(mode: str, items: list[Any], element: T.Type, built: T.Type) -> Any:
    """Build the value of a collecting output tunnel from one item per
    iteration.  ``built`` is the type the checker promised."""
    if mode == "collect":
        return list(items)
    if mode == "stack":
        if element in _STACK_DTYPE:
            return np.array(items, dtype=_STACK_DTYPE[element])
        if not items:
            return _empty(element, 1)
        stacked = np.stack([np.asarray(item) for item in items])
        if stacked.dtype == object:
            raise TypeError("these values cannot be stacked into an array")
        return stacked
    # concat: lists or arrays, end to end
    as_list = isinstance(built, T.ListT) or (
        not isinstance(built, T.ArrayT) and all(isinstance(i, list) for i in items))
    if as_list:
        joined: list[Any] = []
        for item in items:
            if not isinstance(item, list):
                raise TypeError(f"cannot concatenate a {T.type_of(item)} to a list")
            joined.extend(item)
        return joined
    if not items:
        return _empty(built, 0)
    return np.concatenate(items)
