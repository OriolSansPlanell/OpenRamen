"""Real NumPy calls used to measure and test the importer end to end.

Each case is ``(function, positional arguments, keyword arguments)``.  The
node built by the importer is run on the same arguments and must return what
calling NumPy directly returns.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np

import openramen as ramen
from openramen import Graph, analyze, run

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = np.array([3, 1, 2, 3, 1], dtype="int64")
U8 = np.array([[10, 200], [30, 40]], dtype="uint8")
V = np.array([1.0, 2.0, 3.0])
M = np.array([[2.0, 1.0], [1.0, 3.0]])
MASK = np.array([True, False, True])

CASES: list[tuple[str, tuple, dict]] = [
    # creation
    ("zeros", ((2, 3),), {}), ("zeros", (4,), {"dtype": "uint8"}), ("ones", ((2, 2),), {}),
    ("full", ((2, 2), 7.5), {}), ("arange", (5,), {}), ("arange", (1, 10, 3), {}),
    ("linspace", (0.0, 1.0), {"num": 5}), ("eye", (3,), {}), ("identity", (2,), {}),
    ("zeros_like", (A,), {}), ("ones_like", (U8,), {}), ("full_like", (A, 2.0), {}),
    ("array", ([[1, 2], [3, 4]],), {}), ("asarray", ([1.5, 2.5],), {}),
    ("meshgrid", (V, V), {}), ("diag", (V,), {}), ("tri", (3,), {}),
    # shape
    ("reshape", (A, (3, 2)), {}), ("ravel", (A,), {}), ("transpose", (A,), {}),
    ("squeeze", (A[None],), {}), ("expand_dims", (V, 0), {}), ("flip", (A,), {"axis": 1}),
    ("roll", (V, 1), {}), ("concatenate", ([V, V],), {}), ("stack", ([V, V],), {}),
    ("vstack", ([V, V],), {}), ("hstack", ([V, V],), {}), ("tile", (V, 2), {}),
    ("repeat", (V, 2), {}), ("swapaxes", (A, 0, 1), {}), ("atleast_2d", (V,), {}),
    ("pad", (V, 1), {}), ("split", (np.arange(6.0), 3), {}), ("broadcast_to", (V, (2, 3)), {}),
    # arithmetic and ufuncs
    ("add", (A, 1.0), {}), ("add", (2, 3), {}), ("subtract", (A, A), {}),
    ("multiply", (U8, 2), {}), ("divide", (A, 2.0), {}), ("power", (A, 2), {}),
    ("sqrt", (A,), {}), ("sqrt", (4.0,), {}), ("exp", (V,), {}), ("log", (V,), {}),
    ("abs", (-V,), {}), ("sin", (V,), {}), ("maximum", (A, 3.0), {}), ("minimum", (V, V[::-1]), {}),
    ("divmod", (B, 2), {}), ("floor", (V / 2,), {}), ("round", (V / 3, 2), {}),
    ("clip", (A, 2.0, 5.0), {}), ("sign", (-V,), {}), ("negative", (V,), {}),
    ("isnan", (V,), {}), ("isfinite", (A,), {}), ("greater", (A, 3.0), {}),
    ("equal", (B, 3), {}), ("logical_and", (MASK, ~MASK), {}), ("logical_not", (MASK,), {}),
    ("where", (MASK, V, -V), {}), ("hypot", (3.0, 4.0), {}), ("arctan2", (V, V), {}),
    # reductions and statistics
    ("sum", (A,), {}), ("sum", (A,), {"axis": 0}), ("mean", (A,), {}), ("mean", (A,), {"axis": 1}),
    ("std", (A,), {}), ("var", (V,), {}), ("min", (A,), {}), ("max", (A,), {"axis": 0}),
    ("prod", (V,), {}), ("median", (A,), {}), ("percentile", (A, 50), {}),
    ("argmax", (B,), {}), ("argmin", (A,), {"axis": 1}), ("cumsum", (V,), {}),
    ("any", (MASK,), {}), ("all", (MASK,), {}), ("count_nonzero", (MASK,), {}),
    ("ptp", (A,), {}), ("average", (V,), {"weights": V}), ("nanmean", (V,), {}),
    ("histogram", (B,), {"bins": 3}), ("bincount", (B,), {}), ("corrcoef", (V, V * 2), {}),
    ("diff", (B,), {}), ("gradient", (V,), {}), ("convolve", (V, V), {}),
    ("interp", (1.5, V, V * 10), {}), ("cross", (V, V[::-1]), {}),
    # sorting, searching, sets
    ("sort", (B,), {}), ("argsort", (B,), {}), ("unique", (B,), {}),
    ("unique", (B,), {"return_counts": True}), ("searchsorted", (V, 2.5), {}),
    ("nonzero", (MASK,), {}), ("argwhere", (MASK,), {}), ("isin", (B, [1, 2]), {}),
    ("flatnonzero", (MASK,), {}), ("take", (V, [0, 2]), {}), ("intersect1d", (B, [1, 9]), {}),
    # linear algebra
    ("dot", (V, V), {}), ("dot", (M, M), {}), ("matmul", (M, M), {}), ("outer", (V, V), {}),
    ("trace", (M,), {}), ("inner", (V, V), {}), ("tensordot", (M, M), {}), ("kron", (V, V), {}),
    # inspection
    ("shape", (A,), {}), ("ndim", (A,), {}), ("size", (A,), {}), ("allclose", (V, V), {}),
    ("array_equal", (V, V), {}), ("isscalar", (3.0,), {}), ("result_type", (U8, 1.0), {}),
    # awkward ones: flags that change the outputs, variadics, in-place, nested results
    ("linspace", (0.0, 1.0), {"num": 3, "retstep": True}), ("unique", (B,), {"return_index": True}),
    ("histogram2d", (V, V), {"bins": 2}), ("histogramdd", (A,), {"bins": 2}),
    ("polyfit", (V, V * 2, 1), {}), ("polyval", ([1.0, 2.0], V), {}), ("broadcast_arrays", (V, A), {}),
    ("ix_", ([0, 1], [0, 2]), {}), ("lexsort", ((B, B),), {}), ("unravel_index", (4, (2, 3)), {}),
    ("indices", ((2, 2),), {}), ("array_split", (V, 2), {}), ("frexp", (V,), {}), ("modf", (V,), {}),
    ("einsum", ("ij,ij->i", A, A), {}), ("apply_along_axis", (np.sum, 0, A), {}),
    ("partition", (B, 2), {}), ("digitize", (V, [1.5, 2.5]), {}), ("trapezoid", (V,), {}),
    ("cov", (A,), {}), ("quantile", (A, [0.25, 0.75]), {}), ("nanmax", (V,), {}),
    ("triu", (M,), {}), ("tril_indices", (3,), {}), ("diag_indices", (2,), {}),
    ("select", ([MASK, ~MASK], [V, -V]), {}), ("piecewise", (V, [V < 2, V >= 2], [0.0, 1.0]), {}),
    ("copy", (A,), {}), ("astype", (A, "int32"), {}), ("unwrap", (V,), {}),
    ("linalg.norm", (V,), {}), ("linalg.inv", (M,), {}), ("linalg.det", (M,), {}),
    ("linalg.eig", (M,), {}), ("linalg.eigh", (M,), {}), ("linalg.svd", (M,), {}),
    ("linalg.solve", (M, [1.0, 2.0]), {}), ("linalg.qr", (M,), {}), ("linalg.lstsq", (M, [1.0, 2.0]), {}),
    ("linalg.matrix_rank", (M,), {}), ("linalg.slogdet", (M,), {}), ("linalg.pinv", (M,), {}),
    ("fft.fft", (V,), {}), ("fft.ifft", (V,), {}), ("fft.fft2", (M,), {}), ("fft.fftfreq", (4,), {}),
    ("fft.fftshift", (V,), {}), ("fft.rfft", (V,), {}),
]

MODULES = ("numpy", "numpy.linalg", "numpy.fft")


def function(name: str) -> Any:
    obj: Any = np
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def registry(pack: bool) -> ramen.Registry:
    reg = ramen.default_registry()
    for module in MODULES:
        reg.import_module(module, pack=pack)
    return reg


def expected(name: str, args: tuple, kwargs: dict) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(name)(*args, **kwargs)


def same(a: Any, b: Any) -> bool:
    """Equal values of the same kind, NaN-aware, element types included."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.dtype) or isinstance(b, np.dtype):
        return a == b
    if isinstance(a, np.generic) or isinstance(b, np.generic):
        a, b = ramen.types.plain_scalar(a), ramen.types.plain_scalar(b)
    return type(a) is type(b) and (a == b or (a != a and b != b))


def build(registry: ramen.Registry, name: str, args: tuple, kwargs: dict) -> Graph:
    spec = registry.get(f"numpy.{name}")
    graph = Graph(name)
    graph.add_node(spec, "n")
    fixed = [p.name for p in spec.inputs if not p.variadic and p.kind != "keyword_only"]
    pins = {}
    for index, value in enumerate(args):
        if index < len(fixed):
            pins[fixed[index]] = value
        else:
            assert spec.var_positional is not None, f"{name}: too many arguments"
            pins[f"{spec.var_positional.name}_{index - len(fixed)}"] = value
    config_names = {p.name for p in spec.config}
    for key, value in kwargs.items():
        if key in config_names:
            graph.node("n").config[key] = value
        else:
            pins[key] = value
    for pin, value in pins.items():
        source = graph.add_node("base.constant", f"in_{pin}", config={"value": value})
        graph.connect(source, "out", "n", pin)
    return graph


def outcome(registry: ramen.Registry, name: str, args: tuple, kwargs: dict
            ) -> tuple[str, Optional[str]]:
    """``("ok", None)`` or ``(kind of failure, detail)`` for one case."""
    try:
        graph = build(registry, name, args, kwargs)
    except Exception as exc:
        return "not-imported", f"{type(exc).__name__}: {exc}"
    analysis = analyze(graph, registry)
    if not analysis.ok:
        return "rejected", "; ".join(p.message for p in analysis.errors)
    want = expected(name, args, kwargs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run(graph, registry)
    except ramen.NodeError as exc:
        kind = "wrong-type" if exc.pin and "output is declared" in str(exc) else "error"
        return kind, str(exc)
    spec = registry.get(f"numpy.{name}")
    values = [result["n", p.name] for p in spec.outputs]
    got = values[0] if len(values) == 1 else tuple(values)
    if len(values) != 1:
        want = tuple(want)
    if not same(got, ramen.types.plain_scalar(want) if not isinstance(want, tuple) else want):
        return "different", f"got {got!r}, NumPy gives {want!r}"
    return "ok", None
