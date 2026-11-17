"""Type inference by sampling.

The type of ``fn(a, b)`` is found by calling ``fn`` on tiny sample values of
the input types.  That makes the static type agree with what Python and NumPy
really do, promotion rules and all, without re-implementing those rules.
"""

from __future__ import annotations

import functools
import itertools
from typing import Any, Callable, Optional

import numpy as np

from .. import types as T
from ..types import BOOL, COMPLEX, FLOAT, INT, STR, ArrayT

__all__ = ["SAMPLES", "sample_value", "result_type", "scalar_of"]

# Sample values avoid 0, 1, 2 and 0.5, which some operations treat specially.
SAMPLES: dict[T.Type, Any] = {INT: 3, FLOAT: 3.0, COMPLEX: 3j, BOOL: True, STR: "a"}


def sample_value(t: T.Type) -> Any:
    """A tiny value of type ``t``, or ``None`` if the type is not specific."""
    if t in SAMPLES:
        return SAMPLES[t]
    if isinstance(t, ArrayT) and isinstance(t.dtype, str):
        try:
            dtype = np.dtype(t.dtype)
        except TypeError:
            return None
        if dtype.kind not in "biufc":
            return None
        ndim = t.ndim if isinstance(t.ndim, int) else 1
        return np.full((1,) * ndim, 3, dtype=dtype)
    return None


def scalar_of(dtype: Any) -> T.Type:
    """The scalar type an element of an array of this dtype has."""
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        return T.ANY
    return {"b": BOOL, "i": INT, "u": INT, "f": FLOAT, "c": COMPLEX}.get(kind, T.ANY)


def _options(t: Optional[T.Type]) -> "list[T.Type] | None":
    if t is None:
        return None
    return list(t.options) if isinstance(t, T.UnionT) else [t]


def result_type(fn: Callable[..., Any], *in_types: Optional[T.Type],
                raw: bool = False) -> Optional[T.Type]:
    """Type of ``fn`` applied to values of ``in_types``; ``None`` if unknown.

    ``raw`` is for plain NumPy functions, which return a scalar when every
    array involved is 0-d.  With arrays of unknown rank the answer is then
    "array or scalar".  The base palette keeps 0-d arrays as arrays and
    leaves ``raw`` off.
    """
    per_input = [_options(t) for t in in_types]
    if any(opts is None for opts in per_input):
        return None
    combos = list(itertools.islice(itertools.product(*per_input), 17))  # type: ignore[arg-type]
    if len(combos) > 16:
        return None
    results = []
    for combo in combos:
        one = _result_type_one(fn, combo, raw)
        if one is None:
            return None
        results.append(one)
    return T.union(*results)


@functools.lru_cache(maxsize=4096)
def _result_type_one(fn: Callable[..., Any], in_types: "tuple[T.Type, ...]",
                     raw: bool) -> Optional[T.Type]:
    samples = [sample_value(t) for t in in_types]
    if any(s is None for s in samples):
        return None
    try:
        with np.errstate(all="ignore"):
            out = fn(*samples)
    except Exception:
        return None
    out_type = T.type_of(T.plain_scalar(out))
    if isinstance(out_type, ArrayT):
        # The samples of arrays with unknown rank are 1-d: do not trust ndim.
        if any(isinstance(t, ArrayT) and not isinstance(t.ndim, int) for t in in_types):
            unknown_rank = ArrayT(out_type.dtype, None)
            if raw and isinstance(out_type.dtype, str):
                return T.union(unknown_rank, scalar_of(out_type.dtype))
            return unknown_rank
    return out_type
