"""The base palette: values, inferred types, and their agreement."""

import itertools
import math
import warnings

import numpy as np
import pytest

import openramen as ramen
from openramen import types as T
from openramen.library import default_registry

from helpers import call, out_type, single

REGISTRY = default_registry()
U8 = np.array([1, 2, 3], dtype="uint8")
F32 = np.array([[1.5, 2.5]], dtype="float32")
I64 = np.array([4, 5, 6], dtype="int64")
C64 = np.array([1 + 1j, 2j], dtype="complex64")
ZERO_D = np.array(4.0)


def test_every_base_node_is_documented_and_well_formed():
    assert len(REGISTRY) >= 60
    for spec in REGISTRY:
        assert spec.id.startswith("base."), spec.id
        assert spec.summary and spec.summary[0].isupper(), spec.id
        assert spec.category, spec.id
        assert spec.tier == "curated" and spec.executor == "inline", spec.id
        assert callable(spec.resolve())
        for pin in spec.inputs + spec.outputs + spec.config:
            assert isinstance(pin.type, T.Type)
    impure = {s.id for s in REGISTRY if not s.pure}
    assert impure == {"base.fail", "base.random_uniform", "base.current_time"}


@pytest.mark.parametrize("node, inputs, expected", [
    ("add", dict(a=2, b=3), 5), ("add", dict(a=2, b=0.5), 2.5), ("add", dict(a=1, b=2j), 1 + 2j),
    ("subtract", dict(a=2, b=3), -1), ("multiply", dict(a=2.5, b=2), 5.0),
    ("divide", dict(a=7, b=2), 3.5), ("divide", dict(a=6, b=3), 2.0),
    ("floor_divide", dict(a=7, b=2), 3), ("floor_divide", dict(a=-7, b=2), -4),
    ("floor_divide", dict(a=7.5, b=2), 3.0),
    ("remainder", dict(a=7, b=3), 1), ("remainder", dict(a=-7, b=3), 2),
    ("power", dict(a=2, b=10), 1024), ("power", dict(a=2.0, b=-1), 0.5),
    ("power", dict(a=10, b=30), 10**30), ("power", dict(a=4, b=0.5), 2.0),
    ("power", dict(a=1j, b=2), -1 + 0j),
    ("minimum", dict(a=2, b=3), 2), ("maximum", dict(a=2, b=3.5), 3.5),
    ("negate", dict(x=3), -3), ("negate", dict(x=-2.5), 2.5),
    ("absolute", dict(x=-3), 3), ("absolute", dict(x=3 + 4j), 5.0),
    ("sqrt", dict(x=9), 3.0), ("sqrt", dict(x=2.25), 1.5), ("sqrt", dict(x=-1 + 0j), 1j),
    ("exp", dict(x=0), 1.0), ("log", dict(x=1), 0.0), ("sin", dict(x=0), 0.0), ("cos", dict(x=0), 1.0),
    ("equal", dict(a=1, b=1.0), True), ("equal", dict(a="a", b="b"), False),
    ("equal", dict(a=[1, 2], b=[1, 2]), True), ("equal", dict(a=None, b=None), True),
    ("not_equal", dict(a=1, b=2), True),
    ("less", dict(a=1, b=2), True), ("less", dict(a="a", b="b"), True),
    ("less_equal", dict(a=2, b=2.0), True), ("greater", dict(a=1, b=2), False),
    ("greater_equal", dict(a=2.5, b=2), True),
    ("logical_and", dict(a=True, b=False), False), ("logical_or", dict(a=True, b=False), True),
    ("logical_xor", dict(a=True, b=True), False), ("logical_not", dict(x=False), True),
    ("select", dict(condition=True, if_true=1, if_false=2), 1),
    ("select", dict(condition=False, if_true="a", if_false="b"), "b"),
    ("to_int", dict(x=3.9), 3), ("to_int", dict(x=-3.9), -3), ("to_int", dict(x="42"), 42),
    ("to_int", dict(x=True), 1), ("to_float", dict(x=3), 3.0), ("to_float", dict(x="1e3"), 1000.0),
    ("to_bool", dict(x=0), False), ("to_bool", dict(x=0.1), True),
    ("to_text", dict(x=3.5), "3.5"), ("to_text", dict(x=[1, 2]), "[1, 2]"), ("to_text", dict(x=None), "None"),
    ("round", dict(x=2.567, digits=1), 2.6), ("round", dict(x=2.5), 2.0),
    ("floor", dict(x=-2.5), -3), ("ceil", dict(x=2.1), 3),
    ("concat", dict(a="ra", b="men"), "ramen"), ("text_length", dict(text="ramen"), 5),
    ("format_value", dict(value=3.14159, template="{:.2f} mm"), "3.14 mm"),
    ("format_value", dict(value=[1]), "[1]"),
    ("join_text", dict(items=["a", "b"], separator="-"), "a-b"), ("join_text", dict(items=[]), ""),
    ("split_text", dict(text="a b  c"), ["a", "b", "c"]),
    ("split_text", dict(text="a,b", separator=","), ["a", "b"]),
    ("list_length", dict(items=[1, 2, 3]), 3), ("list_length", dict(items=[]), 0),
    ("list_get", dict(items=[1, 2, 3], index=-1), 3),
    ("list_append", dict(items=[1], item=2), [1, 2]),
    ("list_concat", dict(a=[1], b=[2, 3]), [1, 2, 3]),
    ("list_slice", dict(items=[0, 1, 2, 3, 4], start=1, stop=4, step=2), [1, 3]),
    ("list_slice", dict(items=[0, 1, 2]), [0, 1, 2]),
    ("list_reverse", dict(items=[1, 2, 3]), [3, 2, 1]),
    ("range_list", dict(stop=3), [0, 1, 2]), ("range_list", dict(stop=7, start=1, step=3), [1, 4]),
    ("array_to_list", dict(array=F32), [[1.5, 2.5]]), ("array_shape", dict(array=F32), (1, 2)),
    ("array_get", dict(array=U8, index=1), 2), ("array_sum", dict(array=U8), 6),
    ("array_mean", dict(array=U8), 2.0), ("array_min", dict(array=F32), 1.5),
    ("array_max", dict(array=I64), 6), ("identity", dict(value="x"), "x"),
])
def test_values(node, inputs, expected):
    out = call(f"base.{node}", **inputs)
    assert out == expected and type(out) is type(expected)


def test_scalar_outputs_are_python_scalars_never_numpy_ones():
    for node, inputs in [("sqrt", dict(x=4)), ("minimum", dict(a=1, b=2)),
                         ("array_sum", dict(array=F32)), ("array_get", dict(array=U8, index=0)),
                         ("array_mean", dict(array=I64)), ("array_sum", dict(array=ZERO_D))]:
        out = call(f"base.{node}", **inputs)
        assert type(out) in (int, float, complex, bool), (node, type(out))


def test_array_nodes():
    out = call("base.add", a=U8, b=1)
    assert out.dtype == np.uint8 and out.tolist() == [2, 3, 4]
    out = call("base.multiply", a=U8, b=0.5)
    assert out.dtype == np.float64 and out.tolist() == [0.5, 1.0, 1.5]
    out = call("base.less", a=U8, b=2)
    assert out.dtype == bool and out.tolist() == [True, False, False]
    out = call("base.equal", a=U8, b=U8)
    assert out.tolist() == [True, True, True]
    out = call("base.array_from_list", items=[[1, 2], [3, 4]])
    assert out.shape == (2, 2) and out.dtype.kind == "i"
    out = call("base.array_from_list", items=[1, 2], config={"dtype": "float32"})
    assert out.dtype == np.float32
    out = call("base.array_zeros", shape=[2, 3], config={"dtype": "uint16"})
    assert out.shape == (2, 3) and out.dtype == np.uint16
    assert call("base.array_zeros", shape=[2]).dtype == np.float64
    assert call("base.array_range", count=4).tolist() == [0, 1, 2, 3]
    out = call("base.linspace", start=0.0, stop=1.0, count=5)
    assert out.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    row = call("base.array_get", array=F32, index=0)
    assert row.shape == (2,) and row.dtype == np.float32


@pytest.mark.parametrize("node, inputs, config, expected", [
    ("add", dict(a=1, b=2), None, "Int"), ("add", dict(a=1, b=2.0), None, "Float"),
    ("add", dict(a=1.5, b=2j), None, "Complex"),
    ("divide", dict(a=1, b=2), None, "Float"), ("floor_divide", dict(a=1, b=2), None, "Int"),
    ("power", dict(a=2, b=3), None, "Int"), ("power", dict(a=2, b=0.5), None, "Float"),
    ("sqrt", dict(x=4), None, "Float"), ("sqrt", dict(x=1j), None, "Complex"),
    ("absolute", dict(x=1j), None, "Float"),
    ("add", dict(a=U8, b=1), None, "Array[uint8, 1]"),
    ("add", dict(a=U8, b=1.0), None, "Array[float64, 1]"),
    ("add", dict(a=U8, b=F32), None, "Array[float32, 2]"),
    ("add", dict(a=U8, b=I64), None, "Array[int64, 1]"),
    ("divide", dict(a=U8, b=U8), None, "Array[float64, 1]"),
    ("multiply", dict(a=F32, b=2j), None, "Array[complex64, 2]"),
    ("sqrt", dict(x=U8), None, "Array[float16, 1]"),
    ("add", dict(a=ZERO_D, b=1), None, "Array[float64, 0]"),
    ("less", dict(a=ZERO_D, b=1), None, "Array[bool, 0]"),
    ("sqrt", dict(x=ZERO_D), None, "Array[float64, 0]"),
    ("less", dict(a=1, b=2), None, "Bool"), ("less", dict(a=U8, b=2), None, "Array[bool, 1]"),
    ("equal", dict(a="a", b=1), None, "Bool"), ("equal", dict(a=[1], b=[1]), None, "Bool"),
    ("equal", dict(a=U8, b=1), None, "Array[bool, 1]"),
    ("select", dict(condition=True, if_true=1, if_false=2), None, "Int"),
    ("select", dict(condition=True, if_true=1, if_false=2.5), None, "Float"),
    ("select", dict(condition=True, if_true=1, if_false=None), None, "Optional[Int]"),
    ("list_get", dict(items=["a"], index=0), None, "Str"),
    ("list_get", dict(items=[], index=0), None, "Any"),
    ("list_append", dict(items=[1], item=2), None, "List[Int]"),
    ("list_append", dict(items=[1], item=2.5), None, "List[Float]"),
    ("list_concat", dict(a=[1], b=[2.5]), None, "List[Float]"),
    ("list_slice", dict(items=["a"]), None, "List[Str]"),
    ("list_reverse", dict(items=[[1]]), None, "List[List[Int]]"),
    ("array_from_list", dict(items=[1.5, 2.5]), None, "Array[_, _]"),
    ("array_from_list", dict(items=[[1, 2]]), {"dtype": "uint8"}, "Array[uint8, _]"),
    ("array_from_list", dict(items=[]), {"dtype": "int16"}, "Array[int16, _]"),
    ("array_to_list", dict(array=F32), None, "List[List[Float]]"),
    ("array_to_list", dict(array=U8), None, "List[Int]"),
    ("array_get", dict(array=U8, index=0), None, "Int"),
    ("array_get", dict(array=F32, index=0), None, "Array[float32, 1]"),
    ("array_sum", dict(array=U8), None, "Int"), ("array_sum", dict(array=F32), None, "Float"),
    ("array_sum", dict(array=C64), None, "Complex"),
    ("array_mean", dict(array=U8), None, "Float"), ("array_max", dict(array=U8), None, "Int"),
    ("array_zeros", dict(shape=[2]), {"dtype": "int32"}, "Array[int32, _]"),
    ("array_zeros", dict(shape=[2]), None, "Array[float64, _]"),
    ("identity", dict(value=U8), None, "Array[uint8, 1]"),
    ("identity", dict(value=(1, "a")), None, "Tuple[Int, Str]"),
])
def test_inferred_types(node, inputs, config, expected):
    assert out_type(f"base.{node}", config=config, **inputs) == expected


SAMPLES = [3, -2, 2.5, -0.5, 1 + 2j, U8, F32, I64, C64, ZERO_D,
           np.array([True, False]), np.zeros((2, 1, 3), dtype="int16")]
BINARY = ["add", "subtract", "multiply", "divide", "floor_divide", "remainder", "power",
          "minimum", "maximum", "equal", "not_equal", "less", "less_equal", "greater",
          "greater_equal"]
UNARY = ["negate", "absolute", "sqrt", "exp", "log", "sin", "cos"]
REDUCE = ["array_sum", "array_mean", "array_min", "array_max", "array_to_list", "array_shape"]


def _agrees(node, **inputs):
    """Run a node; whatever it returns must conform to the statically inferred type."""
    graph = single(f"base.{node}", **inputs)
    analysis = ramen.analyze(graph)
    if not analysis.ok:
        assert {p.code for p in analysis.errors} <= {"type-mismatch"}, analysis.errors
        return "rejected"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value = ramen.run(graph, check_outputs=True)["n"]
        except ramen.NodeError as exc:
            # an honest run-time failure of the operation, never a type lie
            assert not isinstance(exc.original, T.TypeMismatch), (node, inputs, exc)
            return "failed"
    assert T.conforms(value, analysis.out_type("n", "out"))
    return "ok"


@pytest.mark.parametrize("node", BINARY)
def test_static_types_agree_with_what_binary_nodes_return(node):
    outcomes = [_agrees(node, a=a, b=b) for a, b in itertools.product(SAMPLES, repeat=2)]
    assert outcomes.count("ok") >= 20, (node, outcomes)


@pytest.mark.parametrize("node", UNARY + REDUCE)
def test_static_types_agree_with_what_unary_nodes_return(node):
    pin = "array" if node in REDUCE else "x"
    outcomes = [_agrees(node, **{pin: value}) for value in SAMPLES]
    assert outcomes.count("ok") >= 5, (node, outcomes)


def test_power_has_one_result_type_per_input_type():
    assert math.isnan(call("base.power", a=-8.0, b=0.5))       # not a complex number
    assert math.isnan(call("base.power", a=-2, b=0.5))
    with pytest.raises(ramen.NodeError, match="negative integer power"):
        call("base.power", a=2, b=-1)
    assert call("base.power", a=2.0, b=-1) == 0.5
    with pytest.raises(ramen.NodeError):
        call("base.power", a=U8, b=-1)                          # NumPy refuses too


def test_python_errors_surface_as_node_errors():
    for node, inputs, kind in [
        ("divide", dict(a=1, b=0), ZeroDivisionError),
        ("to_int", dict(x="abc"), ValueError),
        ("list_get", dict(items=[1], index=5), IndexError),
        ("less", dict(a=1, b="a"), TypeError),
        ("array_min", dict(array=np.array([])), ValueError),
        ("array_get", dict(array=ZERO_D, index=0), IndexError),
        ("add", dict(a=U8, b=300), OverflowError),
    ]:
        with pytest.raises(ramen.NodeError) as err:
            call(f"base.{node}", **inputs)
        assert isinstance(err.value.original, kind), node
        assert err.value.node == "n" and err.value.spec == f"base.{node}"


def test_big_integers_are_exact_or_fail_loudly():
    assert call("base.multiply", a=10**20, b=10**20) == 10**40
    assert call("base.sqrt", x=10**40) == 1e20
    assert call("base.maximum", a=10**30, b=1) == 10**30


def test_list_nodes_do_not_modify_their_inputs():
    items = [3, 1, 2]
    for node, extra in [("list_append", dict(item=9)), ("list_reverse", {}),
                        ("list_concat", None), ("list_slice", dict(start=1))]:
        if node == "list_concat":
            call("base.list_concat", a=items, b=items)
        else:
            call(f"base.{node}", items=items, **extra)
        assert items == [3, 1, 2]


def test_bool_is_not_a_number():
    for node, inputs in [("add", dict(a=True, b=1)), ("negate", dict(x=True)),
                         ("less", dict(a=True, b=False)), ("sqrt", dict(x=True))]:
        analysis = ramen.analyze(single(f"base.{node}", **inputs))
        assert {p.code for p in analysis.errors} == {"type-mismatch"}, node


def test_impure_nodes():
    assert 0.0 <= call("base.random_uniform") <= 1.0
    assert 5.0 <= call("base.random_uniform", low=5, high=6) <= 6.0
    assert call("base.current_time") > 1.6e9
    with pytest.raises(ramen.NodeError, match="boom") as err:
        call("base.fail", message="boom")
    assert isinstance(err.value.original, RuntimeError)
    assert "n" not in ramen.run(single("base.identity", value=1), targets=["in_value"])


def test_constant_needs_a_value():
    g = ramen.Graph()
    g.add_node("base.constant", "c")
    assert [p.code for p in ramen.analyze(g).errors] == ["missing-config"]
    for value in [None, 0, "", [], {"a": 1}, U8, (1, 2)]:
        out = call("base.identity", value=value)
        assert T.type_of(out) == T.type_of(value)


def test_zero_dimensional_arrays_stay_arrays():
    """NumPy would return a scalar here; on a wire the type must not depend on the shape."""
    for node, inputs in [("add", dict(a=ZERO_D, b=1)), ("multiply", dict(a=ZERO_D, b=ZERO_D)),
                         ("less", dict(a=ZERO_D, b=5.0)), ("equal", dict(a=ZERO_D, b=4.0)),
                         ("sqrt", dict(x=ZERO_D)), ("negate", dict(x=ZERO_D)),
                         ("maximum", dict(a=ZERO_D, b=1)), ("power", dict(a=ZERO_D, b=2))]:
        out = call(f"base.{node}", **inputs)
        assert isinstance(out, np.ndarray) and out.ndim == 0, node
    assert call("base.add", a=ZERO_D, b=1).item() == 5.0
    assert call("base.array_to_list", array=ZERO_D) == [4.0]


def test_array_from_an_empty_list_matches_its_static_type():
    for items in ([], [[]], [1, 2], [[1.5]]):
        for dtype in (None, "uint8", "float32"):
            config = {} if dtype is None else {"dtype": dtype}
            out = call("base.array_from_list", items=items, config=config)   # outputs checked
            assert dtype is None or out.dtype == np.dtype(dtype)


def test_array_power_dtype_does_not_depend_on_the_exponent_value():
    """``**`` has shortcuts for exponents like 1, 2 and 0.5; the node must not."""
    base_array = np.array([[1.5, 2.5]], dtype="float32")
    for exponent in (0.5, 1.0, 2.0, 4.0, -1.0):
        zero_d = call("base.power", a=base_array, b=np.array(exponent))
        assert zero_d.dtype == np.float64, exponent          # float32 with a float64 array
        weak = call("base.power", a=base_array, b=exponent)
        assert weak.dtype == np.float32, exponent            # a Python float adapts to the array
