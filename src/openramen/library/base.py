"""The base palette: primitives that are not functions of any library.

Arithmetic, comparison, logic, conversion, text, list and array basics.
All of them are hand-written ("curated") and cheap enough to run inline.

Polymorphic nodes such as ``base.add`` work on scalars and arrays alike.
Their output type is found by an ``infer`` hook that applies the real
operation to tiny sample values, so the static type always agrees with what
Python and NumPy do at run time.
"""

from __future__ import annotations

import functools
import itertools
import math
import operator
import random as _random
import time as _time
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .. import types as T
from ..types import ANY, BOOL, COMPLEX, FLOAT, INT, STR, ArrayT, ListT, TupleT
from .registry import Registry
from .spec import spec_from_function

__all__ = ["register", "NUMBER", "REAL", "ORDERABLE"]

NUMBER = T.union(INT, FLOAT, COMPLEX, ArrayT())
REAL = T.union(INT, FLOAT, ArrayT())
ORDERABLE = T.union(INT, FLOAT, STR, ArrayT())

_BASE = Registry()


def register(registry: Registry, *, replace: bool = False) -> None:
    """Add the base palette to ``registry``."""
    registry.update(_BASE, replace=replace)


def _add(fn: Callable[..., Any], name: str, category: str, **options: Any) -> None:
    options.setdefault("executor", "inline")
    _BASE.register(spec_from_function(
        fn, f"base.{name}", category=category, tier="curated", **options))


# --------------------------------------------------------------------------
# Type inference by sampling
# --------------------------------------------------------------------------

_SAMPLES: dict[T.Type, Any] = {INT: 1, FLOAT: 1.0, COMPLEX: 1j, BOOL: True, STR: "a"}


def _sample(t: T.Type) -> Any:
    """A tiny value of type ``t``, or ``None`` if the type is not specific."""
    if t in _SAMPLES:
        return _SAMPLES[t]
    if isinstance(t, ArrayT) and isinstance(t.dtype, str):
        try:
            dtype = np.dtype(t.dtype)
        except TypeError:
            return None
        if dtype.kind not in "biufc":
            return None
        ndim = t.ndim if isinstance(t.ndim, int) else 1
        return np.ones((1,) * ndim, dtype=dtype)
    return None


def _options(t: Optional[T.Type]) -> "list[T.Type] | None":
    if t is None:
        return None
    return list(t.options) if isinstance(t, T.UnionT) else [t]


def _result_type(fn: Callable[..., Any], *in_types: Optional[T.Type]) -> Optional[T.Type]:
    """Type of ``fn`` applied to values of ``in_types``; ``None`` if unknown."""
    per_input = [_options(t) for t in in_types]
    if any(opts is None for opts in per_input):
        return None
    combos = list(itertools.islice(itertools.product(*per_input), 17))  # type: ignore[arg-type]
    if len(combos) > 16:
        return None
    results = []
    for combo in combos:
        one = _result_type_one(fn, combo)
        if one is None:
            return None
        results.append(one)
    return T.union(*results)


@functools.lru_cache(maxsize=4096)
def _result_type_one(fn: Callable[..., Any], in_types: "tuple[T.Type, ...]") -> Optional[T.Type]:
    samples = [_sample(t) for t in in_types]
    if any(s is None for s in samples):
        return None
    try:
        with np.errstate(all="ignore"):
            out = fn(*samples)
    except Exception:
        return None
    out_type = T.type_of(out)
    if isinstance(out_type, ArrayT):
        # The samples of arrays with unknown rank are 1-d: do not trust ndim.
        if any(isinstance(t, ArrayT) and not isinstance(t.ndim, int) for t in in_types):
            return ArrayT(out_type.dtype, None)
    return out_type


def _infer_with(fn: Callable[..., Any], *pins: str) -> Callable[..., Mapping[str, T.Type]]:
    def infer(inputs: Mapping[str, Optional[T.Type]], config: Mapping[str, Any]
              ) -> Mapping[str, T.Type]:
        result = _result_type(fn, *(inputs.get(p) for p in pins))
        return {} if result is None else {"out": result}
    return infer


def _scalar_of(dtype: Any) -> T.Type:
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        return ANY
    return {"b": BOOL, "i": INT, "u": INT, "f": FLOAT, "c": COMPLEX}.get(kind, ANY)


# --------------------------------------------------------------------------
# Numeric
# --------------------------------------------------------------------------

def _keep_arrays(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Make "array in, array out" hold for 0-d arrays too.

    NumPy turns the result of an operation on 0-d arrays into a scalar.  On a
    wire that would make the type depend on the shape, so the 0-d array is
    restored here.
    """
    @functools.wraps(fn)
    def run(*args: Any) -> Any:
        out = fn(*args)
        if isinstance(out, np.generic) and any(isinstance(a, np.ndarray) for a in args):
            return np.asarray(out)
        return out
    return run


def _binary(name: str, fn: Callable[[Any, Any], Any], pin_type: T.Type, out_type: T.Type,
            category: str, summary: str,
            infer: Optional[Callable[..., Mapping[str, T.Type]]] = None) -> None:
    fn = _keep_arrays(fn)

    def run(a: Any, b: Any) -> Any:
        return fn(a, b)
    run.__name__ = name
    run.__doc__ = summary
    _add(run, name, category, types={"a": pin_type, "b": pin_type, "out": out_type},
         infer=infer or _infer_with(fn, "a", "b"))


def _unary(name: str, fn: Callable[[Any], Any], pin_type: T.Type, out_type: T.Type,
           category: str, summary: str) -> None:
    fn = _keep_arrays(fn)

    def run(x: Any) -> Any:
        return fn(x)
    run.__name__ = name
    run.__doc__ = summary
    _add(run, name, category, types={"x": pin_type, "out": out_type},
         infer=_infer_with(fn, "x"))


def _power(a: Any, b: Any) -> Any:
    """``a ** b`` whose result type depends only on the input types.

    Plain Python lets the *values* decide: ``2 ** -1`` is a float and
    ``(-8.0) ** 0.5`` is complex.  A wire needs one type, so this follows
    NumPy: a negative integer exponent is an error and a real power with no
    real result is NaN.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return operator.pow(a, b)
    if isinstance(a, int) and isinstance(b, int):
        if b < 0:
            raise ValueError(
                "an integer cannot be raised to a negative integer power; "
                "convert the base to Float first")
        return a ** b
    result = a ** b
    if isinstance(result, complex) and not isinstance(a, complex) and not isinstance(b, complex):
        return math.nan
    return result


def _extreme(np_fn: Callable[[Any, Any], Any], py_fn: Callable[[Any, Any], Any]
             ) -> Callable[[Any, Any], Any]:
    """Minimum or maximum that is exact for big integers and type-stable."""
    def run(a: Any, b: Any) -> Any:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np_fn(a, b)
        mixed = isinstance(a, float) or isinstance(b, float)
        if mixed and (a != a or b != b):
            return math.nan  # NaN wins, as in NumPy, whichever side it is on
        out = py_fn(a, b)
        return float(out) if mixed else out
    return run


def _float_if_int(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Let a NumPy function take Python integers of any size."""
    def run(x: Any) -> Any:
        return fn(float(x) if type(x) is int else x)
    return run


_binary("add", operator.add, NUMBER, NUMBER, "Numeric", "Add two numbers or arrays.")
_binary("subtract", operator.sub, NUMBER, NUMBER, "Numeric", "Subtract b from a.")
_binary("multiply", operator.mul, NUMBER, NUMBER, "Numeric", "Multiply two numbers or arrays.")
_binary("divide", operator.truediv, NUMBER, NUMBER, "Numeric",
        "Divide a by b.  Two integers give a float.")
_binary("floor_divide", operator.floordiv, REAL, REAL, "Numeric",
        "Divide a by b and round down.")
_binary("remainder", operator.mod, REAL, REAL, "Numeric", "Remainder of a divided by b.")
_binary("power", _power, NUMBER, NUMBER, "Numeric",
        "Raise a to the power b.  Real powers without a real result give NaN.")
_binary("minimum", _extreme(np.minimum, min), REAL, REAL, "Numeric", "Smaller of a and b, element by element.")
_binary("maximum", _extreme(np.maximum, max), REAL, REAL, "Numeric", "Larger of a and b, element by element.")
_unary("negate", operator.neg, NUMBER, NUMBER, "Numeric", "Change the sign.")
_unary("absolute", abs, NUMBER, NUMBER, "Numeric", "Absolute value, or magnitude of a complex number.")
_unary("sqrt", _float_if_int(np.sqrt), NUMBER, NUMBER, "Numeric", "Square root.  Negative reals give NaN.")
_unary("exp", _float_if_int(np.exp), NUMBER, NUMBER, "Numeric", "Exponential: e raised to the power x.")
_unary("log", _float_if_int(np.log), NUMBER, NUMBER, "Numeric", "Natural logarithm.")
_unary("sin", _float_if_int(np.sin), NUMBER, NUMBER, "Numeric", "Sine of x, in radians.")
_unary("cos", _float_if_int(np.cos), NUMBER, NUMBER, "Numeric", "Cosine of x, in radians.")


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def _may_be_array(t: Optional[T.Type]) -> bool:
    if t is None or isinstance(t, (T.AnyT, T.VarT, ArrayT, T.ObjectT)):
        return True
    if isinstance(t, T.UnionT):
        return any(_may_be_array(o) for o in t.options)
    return False


def _infer_equality(fn: Callable[[Any, Any], Any]) -> Callable[..., Mapping[str, T.Type]]:
    def infer(inputs: Mapping[str, Optional[T.Type]], config: Mapping[str, Any]
              ) -> Mapping[str, T.Type]:
        a, b = inputs.get("a"), inputs.get("b")
        if not _may_be_array(a) and not _may_be_array(b):
            return {"out": BOOL}
        result = _result_type(fn, a, b)
        return {} if result is None else {"out": result}
    return infer


_COMPARE_OUT = T.union(BOOL, ArrayT("bool", None))

_binary("equal", operator.eq, ANY, ANY, "Comparison",
        "True where a equals b.  Arrays compare element by element.",
        infer=_infer_equality(_keep_arrays(operator.eq)))
_binary("not_equal", operator.ne, ANY, ANY, "Comparison",
        "True where a differs from b.  Arrays compare element by element.",
        infer=_infer_equality(_keep_arrays(operator.ne)))
_binary("less", operator.lt, ORDERABLE, _COMPARE_OUT, "Comparison", "True where a < b.")
_binary("less_equal", operator.le, ORDERABLE, _COMPARE_OUT, "Comparison", "True where a <= b.")
_binary("greater", operator.gt, ORDERABLE, _COMPARE_OUT, "Comparison", "True where a > b.")
_binary("greater_equal", operator.ge, ORDERABLE, _COMPARE_OUT, "Comparison", "True where a >= b.")


# --------------------------------------------------------------------------
# Boolean
# --------------------------------------------------------------------------

def logical_and(a: bool, b: bool) -> bool:
    """True if both inputs are true."""
    return bool(a and b)


def logical_or(a: bool, b: bool) -> bool:
    """True if at least one input is true."""
    return bool(a or b)


def logical_xor(a: bool, b: bool) -> bool:
    """True if exactly one input is true."""
    return bool(a) != bool(b)


def logical_not(x: bool) -> bool:
    """Invert a boolean."""
    return not x


def select(condition: bool, if_true: Any, if_false: Any) -> Any:
    """Pass ``if_true`` when the condition holds, otherwise ``if_false``."""
    return if_true if condition else if_false


for _fn in (logical_and, logical_or, logical_xor, logical_not):
    _add(_fn, _fn.__name__, "Boolean")
_add(select, "select", "Boolean",
     types={"if_true": "~T", "if_false": "~T", "out": "~T"})


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

_TO_NUMBER = T.union(BOOL, INT, FLOAT, STR)


def to_int(x: Any) -> int:
    """Convert to an integer.  Floats are truncated towards zero."""
    return int(x)


def to_float(x: Any) -> float:
    """Convert to a float."""
    return float(x)


def to_bool(x: Any) -> bool:
    """Convert a number to a boolean: zero is false."""
    return bool(x)


def to_text(x: Any) -> str:
    """Convert any value to its text form."""
    return str(x)


def round_to(x: float, digits: int = 0) -> float:
    """Round to a number of decimal digits."""
    return float(round(x, digits))


def floor(x: float) -> int:
    """Largest integer not above x."""
    return math.floor(x)


def ceil(x: float) -> int:
    """Smallest integer not below x."""
    return math.ceil(x)


_add(to_int, "to_int", "Conversion", types={"x": _TO_NUMBER})
_add(to_float, "to_float", "Conversion", types={"x": _TO_NUMBER})
_add(to_bool, "to_bool", "Conversion", types={"x": T.union(BOOL, INT, FLOAT)})
_add(to_text, "to_text", "Conversion")
_add(round_to, "round", "Conversion")
_add(floor, "floor", "Conversion")
_add(ceil, "ceil", "Conversion")


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

def concat(a: str, b: str) -> str:
    """Join two texts."""
    return a + b


def text_length(text: str) -> int:
    """Number of characters."""
    return len(text)


def format_value(value: Any, template: str = "{}") -> str:
    """Insert a value into a template, e.g. ``"{:.2f} mm"``."""
    return template.format(value)


def join_text(items: list, separator: str = "") -> str:
    """Join a list of texts with a separator."""
    return separator.join(items)


def split_text(text: str, separator: Optional[str] = None) -> list:
    """Split a text at a separator (default: any whitespace)."""
    return text.split(separator)


_add(concat, "concat", "Text")
_add(text_length, "text_length", "Text")
_add(format_value, "format_value", "Text")
_add(join_text, "join_text", "Text", types={"items": "List[Str]"})
_add(split_text, "split_text", "Text", types={"out": "List[Str]"})


# --------------------------------------------------------------------------
# List
# --------------------------------------------------------------------------

def list_length(items: list) -> int:
    """Number of elements."""
    return len(items)


def list_get(items: list, index: int) -> Any:
    """Element at an index.  Negative indexes count from the end."""
    return items[index]


def list_append(items: list, item: Any) -> list:
    """A new list with one more element at the end."""
    return [*items, item]


def list_concat(a: list, b: list) -> list:
    """A new list holding the elements of a, then those of b."""
    return [*a, *b]


def list_slice(items: list, start: Optional[int] = None, stop: Optional[int] = None,
               step: Optional[int] = None) -> list:
    """Part of a list, like ``items[start:stop:step]``."""
    return items[start:stop:step]


def list_reverse(items: list) -> list:
    """A new list in reverse order."""
    return items[::-1]


def range_list(stop: int, start: int = 0, step: int = 1) -> list:
    """Integers from start (included) to stop (excluded)."""
    return list(range(start, stop, step))


_LIST = {"items": "List[~T]", "out": "List[~T]"}
_add(list_length, "list_length", "List", types={"items": "List[~T]"})
_add(list_get, "list_get", "List", types={"items": "List[~T]", "out": "~T"})
_add(list_append, "list_append", "List", types={**_LIST, "item": "~T"})
_add(list_concat, "list_concat", "List",
     types={"a": "List[~T]", "b": "List[~T]", "out": "List[~T]"})
_add(list_slice, "list_slice", "List", types=_LIST)
_add(list_reverse, "list_reverse", "List", types=_LIST)
_add(range_list, "range_list", "List", types={"out": "List[Int]"})


# --------------------------------------------------------------------------
# Array
# --------------------------------------------------------------------------

def array_from_list(items: list, dtype: Optional[str] = None) -> np.ndarray:
    """Build an array from a (nested) list of numbers.

    Set ``dtype`` to fix the element type; otherwise NumPy chooses it from
    the values.
    """
    return np.array(items, dtype=dtype)


def _infer_array_from_list(inputs: Mapping[str, Optional[T.Type]], config: Mapping[str, Any]
                           ) -> Mapping[str, T.Type]:
    # A list type says nothing about length, and NumPy makes ``[]`` a 1-d
    # float64 array whatever the list was meant to hold.  So only an explicit
    # dtype is known statically; the rank never is.
    wanted = config.get("dtype")
    if wanted is None:
        return {}
    try:
        return {"out": ArrayT(T.dtype_name(wanted), None)}
    except TypeError:
        return {}


def array_to_list(array: np.ndarray) -> list:
    """Convert an array to a (nested) list of Python numbers.

    A 0-d array gives a list with one element, so the result is always a list.
    """
    return np.atleast_1d(array).tolist()


def _infer_array_to_list(inputs: Mapping[str, Optional[T.Type]], config: Mapping[str, Any]
                         ) -> Mapping[str, T.Type]:
    t = inputs.get("array")
    if isinstance(t, ArrayT) and isinstance(t.ndim, int) and isinstance(t.dtype, str):
        out: T.Type = _scalar_of(t.dtype)
        for _ in range(max(t.ndim, 1)):
            out = ListT(out)
        return {"out": out}
    return {}


def array_shape(array: np.ndarray) -> tuple:
    """Length of the array along each dimension."""
    return tuple(int(n) for n in array.shape)


def array_get(array: np.ndarray, index: int) -> Any:
    """Element (1-d array) or sub-array (higher dimensions) at an index."""
    return array[index]


def _infer_array_get(inputs: Mapping[str, Optional[T.Type]], config: Mapping[str, Any]
                     ) -> Mapping[str, T.Type]:
    t = inputs.get("array")
    if not isinstance(t, ArrayT) or not isinstance(t.ndim, int):
        return {}
    if t.ndim == 0:
        return {}  # indexing a 0-d array fails at run time
    if t.ndim == 1:
        return {"out": _scalar_of(t.dtype)} if isinstance(t.dtype, str) else {}
    return {"out": ArrayT(t.dtype if isinstance(t.dtype, str) else None, t.ndim - 1)}


def array_sum(array: np.ndarray) -> Any:
    """Sum of all elements."""
    return np.sum(array)


def array_mean(array: np.ndarray) -> Any:
    """Mean of all elements."""
    return np.mean(array)


def array_min(array: np.ndarray) -> Any:
    """Smallest element."""
    return np.min(array)


def array_max(array: np.ndarray) -> Any:
    """Largest element."""
    return np.max(array)


def array_zeros(shape: list, dtype: str = "float64") -> np.ndarray:
    """An array of zeros with the given shape."""
    return np.zeros(tuple(shape), dtype=dtype)


def _infer_array_zeros(inputs: Mapping[str, Optional[T.Type]], config: Mapping[str, Any]
                       ) -> Mapping[str, T.Type]:
    try:
        return {"out": ArrayT(T.dtype_name(config.get("dtype", "float64")), None)}
    except TypeError:
        return {}


def array_range(count: int) -> np.ndarray:
    """Integers 0, 1, ... count - 1 as a 1-d array."""
    return np.arange(count, dtype=np.int64)


def linspace(start: float, stop: float, count: int = 50) -> np.ndarray:
    """Evenly spaced numbers from start to stop, both included."""
    return np.linspace(start, stop, count, dtype=np.float64)


_REDUCED = T.union(BOOL, INT, FLOAT, COMPLEX)
_add(array_from_list, "array_from_list", "Array", config=("dtype",),
     types={"items": "List", "dtype": "Optional[Str]"}, infer=_infer_array_from_list)
_add(array_to_list, "array_to_list", "Array", types={"out": "List"},
     infer=_infer_array_to_list)
_add(array_shape, "array_shape", "Array", types={"out": "Tuple[Int, ...]"})
_add(array_get, "array_get", "Array", types={"out": ANY}, infer=_infer_array_get)
_add(array_sum, "array_sum", "Array", types={"out": _REDUCED},
     infer=_infer_with(np.sum, "array"))
_add(array_mean, "array_mean", "Array", types={"out": T.union(FLOAT, COMPLEX)},
     infer=_infer_with(np.mean, "array"))
_add(array_min, "array_min", "Array", types={"out": _REDUCED},
     infer=_infer_with(np.min, "array"))
_add(array_max, "array_max", "Array", types={"out": _REDUCED},
     infer=_infer_with(np.max, "array"))
_add(array_zeros, "array_zeros", "Array", config=("dtype",),
     types={"shape": "List[Int]"}, infer=_infer_array_zeros)
_add(array_range, "array_range", "Array", types={"out": "Array[int64, 1]"})
_add(linspace, "linspace", "Array", types={"out": "Array[float64, 1]"})


# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------

def constant(value: Any) -> Any:
    """A fixed value, set in the node's configuration."""
    return value


def _infer_constant(inputs: Mapping[str, Optional[T.Type]], config: Mapping[str, Any]
                    ) -> Mapping[str, T.Type]:
    return {"out": T.type_of(config["value"])} if "value" in config else {}


def identity(value: Any) -> Any:
    """Pass a value through unchanged."""
    return value


def fail(message: str = "failed") -> None:
    """Stop the run with an error message."""
    raise RuntimeError(message)


def random_uniform(low: float = 0.0, high: float = 1.0) -> float:
    """A random float between low and high."""
    return _random.uniform(low, high)


def current_time() -> float:
    """Seconds since 1970-01-01 (the Unix epoch)."""
    return _time.time()


_add(constant, "constant", "Utility", config=("value",), infer=_infer_constant)
_add(identity, "identity", "Utility", types={"value": "~T", "out": "~T"})
_add(fail, "fail", "Utility", pure=False)
_add(random_uniform, "random_uniform", "Utility", pure=False)
_add(current_time, "current_time", "Utility", pure=False)

del _fn
