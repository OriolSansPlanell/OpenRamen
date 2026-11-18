"""Remembering what pure nodes computed.

A pure node gives the same outputs for the same inputs, so its outputs can be
kept and handed out again.  Every value on a wire gets a *token*, a short
fingerprint: of its content when that is cheap to compute (numbers, short
text), and otherwise of where it came from (which node, with which inputs).
A node's key is made of its own identity and the tokens of its inputs; the
same key means the same outputs.

Because the tokens of computed values depend only on how they were computed,
a value that was evicted and is computed again gets the same token, and
everything downstream of it is still found in the cache.
"""

from __future__ import annotations

import enum
import hashlib
import itertools
import struct
import threading
from collections import OrderedDict
from pathlib import PurePath
from typing import Any, Optional

import numpy as np

from .. import types as T

__all__ = ["Cache", "fingerprint"]

_SMALL = 256            # text and bytes up to this length are fingerprinted by content
_unique = itertools.count(1)


def _feed(h: Any, value: Any) -> bool:
    """Add a value to a hash.  False if the value has no stable fingerprint."""
    if value is None:
        h.update(b"N")
    elif isinstance(value, (bool, np.bool_)):
        h.update(b"T" if value else b"F")
    elif isinstance(value, enum.Enum):
        cls = type(value)
        h.update(f"e{cls.__module__}:{cls.__qualname__}.{value.name};".encode())
    elif isinstance(value, int):
        h.update(f"i{value};".encode())
    elif isinstance(value, float):
        h.update(b"f" + struct.pack("<d", value))
    elif isinstance(value, complex):
        h.update(b"c" + struct.pack("<dd", value.real, value.imag))
    elif isinstance(value, str):
        data = value.encode("utf-8", "surrogatepass")
        h.update(f"s{len(data)};".encode() + data)
    elif isinstance(value, (bytes, bytearray)):
        h.update(f"b{len(value)};".encode() + bytes(value))
    elif isinstance(value, PurePath):
        h.update(f"p{value};".encode())
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return False
        h.update(f"a{value.dtype.str}{value.shape};".encode())
        h.update(np.ascontiguousarray(value).data)
    elif isinstance(value, np.generic):
        h.update(f"g{value.dtype.str};".encode() + value.tobytes())
    elif isinstance(value, (list, tuple)):
        h.update(f"{'l' if isinstance(value, list) else 't'}{len(value)};".encode())
        return all(_feed(h, item) for item in value)
    elif isinstance(value, dict):
        h.update(f"d{len(value)};".encode())
        return all(_feed(h, key) and _feed(h, item) for key, item in value.items())
    elif isinstance(value, (set, frozenset)):
        prints: list[bytes] = []
        for item in value:
            printed = fingerprint(item)
            if printed is None:
                return False
            prints.append(printed)
        h.update(f"S{len(prints)};".encode() + b"".join(sorted(prints)))
    elif isinstance(value, T.Type):
        h.update(f"y{value};".encode())
    else:
        return False
    return True


def fingerprint(value: Any) -> Optional[bytes]:
    """A 16-byte fingerprint of a value's content, or ``None`` for values
    that cannot be fingerprinted (objects with reference semantics)."""
    h = hashlib.blake2b(digest_size=16)
    return h.digest() if _feed(h, value) else None


def digest(*parts: bytes) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(len(part).to_bytes(4, "little"))
        h.update(part)
    return h.digest()


def unique_token() -> bytes:
    """A token no other value has: for values nothing is known about."""
    return b"u" + next(_unique).to_bytes(15, "little")


def content_token(value: Any) -> Optional[bytes]:
    """The fingerprint of a value if it is cheap to compute, else ``None``."""
    if value is None or isinstance(value, (bool, int, float, complex, PurePath, enum.Enum)):
        return fingerprint(value)
    if isinstance(value, (str, bytes)) and len(value) <= _SMALL:
        return fingerprint(value)
    return None


def function_identity(fn: Any) -> bytes:
    """Changes when a function is replaced by one with different code."""
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{getattr(fn, '__module__', '')}:{getattr(fn, '__qualname__', '')};".encode())
    code = getattr(fn, "__code__", None)
    if code is not None:
        h.update(code.co_code)
        h.update(repr(code.co_consts).encode("utf-8", "replace"))
    return h.digest()


def size_of(value: Any) -> int:
    """Roughly how many bytes a value keeps alive."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes) + 96
    if isinstance(value, (str, bytes, bytearray)):
        return len(value) + 56
    if isinstance(value, (list, tuple, set, frozenset)):
        return 64 + sum(size_of(item) for item in value)
    if isinstance(value, dict):
        return 64 + sum(size_of(key) + size_of(item) for key, item in value.items())
    return 32


class Cache:
    """Keeps the outputs of pure nodes between runs.

    ``max_bytes`` is the memory budget for the kept values; the values used
    least recently are dropped first, and computed again when needed.

    Arrays that come out of a cache are read-only: they are shared between
    runs.  Copy one before changing it.
    """

    def __init__(self, max_bytes: int = 1 << 30, max_entries: int = 100_000):
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("the limits of a cache cannot be negative")
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self.hits = 0
        self.misses = 0
        self._bytes = 0
        self._lock = threading.Lock()
        self._entries: OrderedDict[bytes, tuple[dict[str, Any], dict[str, bytes], int]] = \
            OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        """Approximate size of everything the cache holds."""
        return self._bytes

    def clear(self) -> None:
        """Forget everything.  The counters are kept."""
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def get(self, key: bytes) -> Optional[tuple[dict[str, Any], dict[str, bytes]]]:
        """The outputs and tokens stored under ``key``, or ``None``."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return dict(entry[0]), entry[1]

    def put(self, key: bytes, outputs: dict[str, Any], tokens: dict[str, bytes]) -> None:
        """Keep the outputs of a node.  Arrays among them become read-only."""
        size = 64
        for value in outputs.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            size += size_of(value)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old[2]
            if size > self.max_bytes or self.max_entries == 0:
                return                       # would not fit even alone
            self._entries[key] = (dict(outputs), dict(tokens), size)
            self._bytes += size
            while self._bytes > self.max_bytes or len(self._entries) > self.max_entries:
                _, dropped = self._entries.popitem(last=False)
                self._bytes -= dropped[2]

    def __repr__(self) -> str:
        return (f"<Cache: {len(self)} entries, {self._bytes / 1e6:.1f} MB of "
                f"{self.max_bytes / 1e6:.0f} MB, {self.hits} hits, {self.misses} misses>")
