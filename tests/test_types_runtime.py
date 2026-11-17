"""Run-time side of the type system: type_of, conforms, accept, annotations."""

import collections.abc as cabc
import dataclasses
import enum
import os
import random
import typing
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import (
    Annotated, Any, Callable, Dict, Iterable, Iterator, List, Literal,
    NamedTuple, Optional, Sequence, Tuple, TypedDict, TypeVar, Union,
)

import numpy as np
import numpy.typing as npt
import pytest

from openramen import types as T
from openramen.types import (
    ANY, BOOL, BYTES, COMPLEX, FLOAT, INT, NONE, PATH, STR,
    ArrayT, CallableT, DictT, IteratorT, ListT, ObjectT, RecordT, TupleT,
    TypeMismatch, VarT, Verdict,
    accept, check, conforms, dtype_name, from_annotation, optional,
    parse_type, type_of, union,
)

from typegen import Color, Opaque, Pair, Point, random_type, random_value


def P(text):
    return parse_type(text)


class Shade(enum.IntEnum):
    DARK = 1
    LIGHT = 2


class Movie(TypedDict):
    title: str
    year: int


@dataclasses.dataclass
class Tree:
    value: int
    children: "List[Tree]"


@pytest.mark.parametrize("value, expected", [
    (None, "None"), (True, "Bool"), (False, "Bool"), (0, "Int"), (-7, "Int"),
    (10**30, "Int"), (1.5, "Float"), (float("nan"), "Float"), (1j, "Complex"),
    ("", "Str"), ("x", "Str"), (b"x", "Bytes"),
    (Path("a"), "Path"), (PurePosixPath("/a"), "Path"), (PureWindowsPath("c:/a"), "Path"),
    (np.bool_(True), "Bool"), (np.int8(1), "Int"), (np.uint64(1), "Int"),
    (np.float32(1), "Float"), (np.float64(1), "Float"), (np.complex64(1), "Complex"),
    (np.str_("a"), "Str"),
    (np.zeros((2, 3)), "Array[float64, 2]"), (np.zeros(3, dtype="uint8"), "Array[uint8, 1]"),
    (np.array(1.0), "Array[float64, 0]"), (np.zeros((1, 1, 1), bool), "Array[bool, 3]"),
    (np.array(["2020-01-01"], dtype="datetime64[ns]"), "Array[datetime64, 1]"),
    ([], "List[Any]"), ([1, 2], "List[Int]"), ([1, 2.5], "List[Union[Float, Int]]"),
    ([1, None], "List[Optional[Int]]"), ([[1], [2]], "List[List[Int]]"),
    ([1, "a", None, 2.5, b""], "List[Any]"),
    ((), "Tuple[]"), ((1, "a"), "Tuple[Int, Str]"), ((1,), "Tuple[Int]"),
    (tuple(range(20)), "Tuple[Int, ...]"),
    ({}, "Dict[Any, Any]"), ({"a": 1}, "Dict[Str, Int]"), ({1: [1.5]}, "Dict[Int, List[Float]]"),
    (len, "Callable[..., Any]"), (lambda x: x, "Callable[..., Any]"),
    (iter([1]), "Iterator[Any]"), ((x for x in ()), "Iterator[Any]"),
])
def test_type_of(value, expected):
    assert type_of(value) == P(expected)


def test_type_of_user_classes():
    assert type_of(Color.RED) == T.EnumT("typegen.Color", ("RED", "GREEN"))
    assert type_of(Shade.DARK) == T.EnumT(f"{__name__}.Shade", ("DARK", "LIGHT"))
    assert type_of(Point(1.0, 2.0)) == RecordT("typegen.Point", (("x", FLOAT), ("y", FLOAT)))
    assert type_of(Pair(1, "a")) == RecordT("typegen.Pair", (("first", INT), ("second", STR)))
    assert type_of(Opaque()) == ObjectT("typegen:Opaque")
    assert type_of({1, 2}) == ObjectT("builtins:set")
    assert type_of(int) == ObjectT("builtins:type")


def test_classes_defined_inside_functions_get_parseable_names():
    @dataclasses.dataclass
    class Local:
        n: int

    class Thing:
        pass

    t = type_of(Local(1))
    assert parse_type(str(t)) == t and conforms(Local(2), t)
    obj_t = type_of(Thing())
    assert parse_type(str(obj_t)) == obj_t
    assert conforms(Thing(), obj_t) and not conforms(Opaque(), obj_t)


@pytest.mark.parametrize("value, type_text, ok", [
    (1, "Int", True), (True, "Int", False), (1.0, "Int", False), (np.int32(1), "Int", True),
    (Shade.DARK, "Int", False),
    (1.0, "Float", True), (1, "Float", False), (np.float32(1), "Float", True),
    (True, "Bool", True), (1, "Bool", False), (np.bool_(False), "Bool", True),
    (1j, "Complex", True), (1.0, "Complex", False),
    ("a", "Str", True), (b"a", "Str", False), (b"a", "Bytes", True), (bytearray(b"a"), "Bytes", False),
    (Path("a"), "Path", True), ("a", "Path", False),
    (None, "None", True), (0, "None", False), (None, "Int", False),
    (object(), "Any", True), (None, "Any", True), (3, "~T", True),
    (np.zeros((2, 2)), "Array", True), (np.zeros((2, 2)), "Array[float64, 2]", True),
    (np.zeros((2, 2)), "Array[float32, 2]", False), (np.zeros((2, 2)), "Array[float64, 1]", False),
    (np.zeros((2, 2)), "Array[_, 2]", True), ([[0.0]], "Array", False), (1.0, "Array", False),
    (np.float64(1), "Array", False),
    ([1, 2], "List[Int]", True), ([1, 2.0], "List[Int]", False), ([], "List[Int]", True),
    ((1, 2), "List[Int]", False), ([1, "a"], "List", True),
    ((1, "a"), "Tuple[Int, Str]", True), ((1,), "Tuple[Int, Str]", False),
    ((1, 2), "Tuple[Int, ...]", True), ((), "Tuple[Int, ...]", True), ((1, "a"), "Tuple[Int, ...]", False),
    ([1], "Tuple[Int]", False), ((), "Tuple[]", True), ((1,), "Tuple[]", False),
    ({"a": 1}, "Dict[Str, Int]", True), ({"a": "b"}, "Dict[Str, Int]", False),
    ({1: 1}, "Dict[Str, Int]", False), ({}, "Dict[Str, Int]", True), ([], "Dict", False),
    (1, "Optional[Int]", True), (None, "Optional[Int]", True), ("a", "Optional[Int]", False),
    (1, "Union[Int, Str]", True), (1.5, "Union[Int, Str]", False),
    (len, "Callable[[Int], Int]", True), (3, "Callable", False),
    (iter([]), "Iterator[Int]", True), ([], "Iterator[Int]", False),
])
def test_conforms(value, type_text, ok):
    assert conforms(value, P(type_text)) is ok


def test_conforms_enums_records_objects():
    color_t = type_of(Color.RED)
    assert conforms(Color.GREEN, color_t)
    assert not conforms(Shade.DARK, color_t) and not conforms("RED", color_t)
    assert not conforms(Color.RED, T.EnumT("typegen.Color", ("GREEN",)))

    point_t = type_of(Point(0.0, 0.0))
    assert conforms(Point(1.0, 2.0), point_t)
    assert not conforms(Point(1, 2.0), point_t)          # field of the wrong type
    assert not conforms(Pair(1, "a"), point_t)
    assert not conforms((1.0, 2.0), point_t)
    assert conforms(Pair(1, "a"), type_of(Pair(0, "")))

    movie_t = from_annotation(Movie)
    assert conforms({"title": "Tampopo", "year": 1985}, movie_t)
    assert not conforms({"title": "Tampopo"}, movie_t)
    assert not conforms({"title": "Tampopo", "year": "1985"}, movie_t)

    assert conforms(Opaque(), ObjectT("typegen:Opaque"))
    assert not conforms(object(), ObjectT("typegen:Opaque"))
    assert conforms({}, ObjectT("builtins:dict"))
    assert not conforms({}, ObjectT("no.such:Class"))


@pytest.mark.parametrize("value, type_text, expected", [
    (3, "Float", 3.0), (3, "Complex", 3 + 0j), (2.5, "Complex", 2.5 + 0j),
    (np.int64(3), "Float", 3.0), ("a/b", "Path", Path("a/b")), (Path("a"), "Str", "a"),
    ([1, 2.5], "List[Float]", [1.0, 2.5]), ((1, 2), "List[Float]", [1.0, 2.0]),
    ([[1], [2.5]], "List[List[Float]]", [[1.0], [2.5]]),
    ((1, "a"), "Tuple[Float, Str]", (1.0, "a")), ((1, 2), "Tuple[Float, ...]", (1.0, 2.0)),
    ({"a": 1}, "Dict[Str, Float]", {"a": 1.0}),
    (3, "Optional[Float]", 3.0), (None, "Optional[Float]", None),
    (3, "Union[Float, Str]", 3.0), ([1, None], "List[Optional[Float]]", [1.0, None]),
])
def test_accept_coerces(value, type_text, expected):
    out = accept(value, P(type_text))
    assert out == expected and type(out) is type(expected)
    if isinstance(expected, (list, tuple)):
        assert [type(x) for x in out] == [type(x) for x in expected]


@pytest.mark.parametrize("value, type_text", [
    (1.5, "Int"), ("3", "Int"), (True, "Int"), (1, "Bool"), (1, "Str"), (None, "Float"),
    (1j, "Float"), ([1, "a"], "List[Float]"), ((1, 2), "Tuple[Float]"), ((1, "a"), "Tuple[Float, ...]"),
    ({1: 1}, "Dict[Str, Float]"), ({"a": "b"}, "Dict[Str, Float]"), ("a", "Optional[Float]"),
    ([1.5], "Array"), (np.zeros(2), "Array[float64, 2]"), (np.zeros(2), "Array[int64, 1]"),
    (1.5, "Array[float64, 1]"), (3, "Callable"), ([1], "Iterator[Str]"), (b"a", "Union[Int, Str]"),
])
def test_accept_rejects(value, type_text):
    with pytest.raises(TypeMismatch) as err:
        accept(value, P(type_text))
    assert "expected" in str(err.value)


def test_accept_returns_conforming_values_untouched():
    xs = [1, 2, 3]
    assert accept(xs, P("List[Int]")) is xs
    arr = np.zeros(3)
    assert accept(arr, ArrayT()) is arr
    assert accept(arr, ANY) is arr


def test_accept_arrays():
    out = accept(np.arange(3, dtype="uint8"), P("Array[float64, 1]"))
    assert out.dtype == np.float64
    with pytest.raises(TypeMismatch):
        accept(2, P("Array[float64, 0]"))
    out = accept([1, 2], P("Iterator[Float]"))
    assert list(out) == [1.0, 2.0]


def test_dtype_name_is_always_an_identifier():
    for dt in ["float64", "<U5", "datetime64[ns]", "timedelta64[s]", "S3", "bool",
               np.dtype([("a", "i4"), ("b", "f8")]), object]:
        name = dtype_name(dt)
        assert name.isidentifier(), name
        ArrayT(name, 1)  # constructible


@pytest.mark.parametrize("seed", range(500))
def test_every_value_conforms_to_its_own_type(seed):
    value = random_value(random.Random(seed))
    t = type_of(value)
    assert conforms(value, t), (value, t)
    assert accept(value, t) is value
    assert parse_type(str(t)) == t
    assert accept(value, ANY) is value


@pytest.mark.parametrize("seed", range(500))
def test_accept_agrees_with_check(seed):
    """A value may only be accepted where the static verdict allows it."""
    rng = random.Random(seed)
    value, target = random_value(rng), random_type(rng, variables=False)
    verdict = check(type_of(value), target).verdict
    try:
        out = accept(value, target)
    except TypeMismatch:
        assert verdict is not Verdict.EXACT, (value, target)
    else:
        assert verdict is not Verdict.INCOMPATIBLE, (value, target)
        if not isinstance(out, cabc.Iterator):
            assert conforms(out, target), (value, target, out)


# -- Python annotations -----------------------------------------------------

TV = TypeVar("TV")


@pytest.mark.parametrize("annotation, expected", [
    (bool, "Bool"), (int, "Int"), (float, "Float"), (complex, "Complex"), (str, "Str"),
    (bytes, "Bytes"), (None, "None"), (type(None), "None"), (Any, "Any"), (object, "Any"),
    (Path, "Path"), (PurePosixPath, "Path"), (os.PathLike, "Path"),
    (list, "List"), (List, "List"), (List[int], "List[Int]"), (list[float], "List[Float]"),
    (Sequence[str], "List[Str]"), (cabc.Sequence, "List"),
    (tuple, "Tuple[Any, ...]"), (Tuple, "Tuple[Any, ...]"), (Tuple[int, str], "Tuple[Int, Str]"),
    (tuple[int, ...], "Tuple[Int, ...]"),
    (dict, "Dict"), (Dict[str, int], "Dict[Str, Int]"), (cabc.Mapping[str, float], "Dict[Str, Float]"),
    (Optional[int], "Optional[Int]"), (Union[int, str], "Union[Int, Str]"),
    (int | None, "Optional[Int]"), (Union[int, Any], "Any"),
    (Literal["a", "b"], "Str"), (Literal[1, 2], "Int"), (Literal[1, "a"], "Union[Int, Str]"),
    (Iterator[int], "Iterator[Int]"), (Iterable[str], "Iterator[Str]"),
    (cabc.Generator[int, None, None], "Iterator[Int]"), (cabc.Iterable, "Iterator"),
    (Callable, "Callable"), (Callable[[int], str], "Callable[[Int], Str]"),
    (Callable[..., int], "Callable[..., Int]"), (cabc.Callable[[], None], "Callable[[], None]"),
    (TV, "~TV"), (List[TV], "List[~TV]"),
    (np.ndarray, "Array"), (npt.NDArray[np.float32], "Array[float32, _]"),
    (npt.NDArray[np.uint8], "Array[uint8, _]"), (npt.NDArray[Any], "Array"),
    (np.float64, "Float"), (np.int16, "Int"), (np.bool_, "Bool"), (np.complex128, "Complex"),
    (np.integer, "Int"), (np.floating, "Float"),
    (Annotated[int, "units: mm"], "Int"), (Annotated[np.ndarray, ArrayT("float64", 2)], "Array[float64, 2]"),
    (ArrayT("uint8", 3), "Array[uint8, 3]"), ("Array[float64, 2]", "Array[float64, 2]"),
    ("not a type", "Any"), (42, "Any"), (set, "Object[builtins:set]"),
    (set[int], "Object[builtins:set]"), (Opaque, "Object[typegen:Opaque]"),
])
def test_from_annotation(annotation, expected):
    assert from_annotation(annotation) == P(expected)


def test_from_annotation_records_and_enums():
    assert from_annotation(Color) == T.EnumT("typegen.Color", ("RED", "GREEN"))
    assert from_annotation(Point) == RecordT("typegen.Point", (("x", FLOAT), ("y", FLOAT)))
    assert from_annotation(Pair) == RecordT("typegen.Pair", (("first", INT), ("second", STR)))
    assert from_annotation(Movie) == RecordT(f"{__name__}.Movie", (("title", STR), ("year", INT)))
    assert from_annotation(Optional[Point]) == optional(from_annotation(Point))


def test_recursive_dataclass_does_not_loop():
    t = from_annotation(Tree)
    assert isinstance(t, RecordT)
    assert dict(t.fields)["children"] == ListT(ObjectT(f"{__name__}:Tree"))
    assert conforms(Tree(1, []), t)


def test_from_annotation_never_raises():
    class Weird:
        def __eq__(self, other):
            raise RuntimeError("no comparisons")
        __hash__ = None  # type: ignore[assignment]

    assert from_annotation(Weird()) == ANY
    assert from_annotation(np.zeros(3)) == ANY
    assert from_annotation(typing.ForwardRef("Nope")) == ANY
    assert from_annotation(...) == ANY
    assert from_annotation(Union) == ANY
    assert isinstance(from_annotation(CallableT((INT,), INT)), CallableT)
    assert from_annotation(IteratorT(INT)) == IteratorT(INT)
    assert from_annotation(DictT()) == DictT(ANY, ANY)
    assert from_annotation(TupleT(())) == TupleT(())
    assert from_annotation(VarT("X")) == VarT("X")
    assert from_annotation(union(INT, STR)) == union(INT, STR)
    assert from_annotation(BOOL) is BOOL and from_annotation(BYTES) is BYTES
    assert from_annotation(COMPLEX) is COMPLEX and from_annotation(PATH) is PATH
    assert from_annotation(NONE) is NONE


class TestRecordsAtRunTime:
    """NamedTuples are tuples and TypedDicts are dicts; the checker knows."""

    def test_namedtuple_flows_into_tuple_and_list_pins(self):
        pair_t = from_annotation(Pair)
        assert check(pair_t, P("Tuple[Int, Str]")).verdict is Verdict.EXACT
        assert check(pair_t, P("Tuple[Float, Str]")).verdict is Verdict.COERCION
        assert check(pair_t, P("Tuple[Str, Str]")).verdict is Verdict.INCOMPATIBLE
        assert check(pair_t, P("List[Any]")).verdict is Verdict.COERCION
        assert check(pair_t, P("Dict")).verdict is Verdict.INCOMPATIBLE
        assert accept(Pair(1, "a"), P("Tuple[Int, Str]")) == (1, "a")
        assert accept(Pair(1, "a"), P("List[Any]")) == [1, "a"]

    def test_dataclass_does_not_flow_into_tuple(self):
        point_t = from_annotation(Point)
        assert check(point_t, P("Tuple[Float, Float]")).verdict is Verdict.INCOMPATIBLE
        with pytest.raises(TypeMismatch):
            accept(Point(1.0, 2.0), P("Tuple[Float, Float]"))

    def test_typed_dict(self):
        movie_t = from_annotation(Movie)
        assert check(P("Dict[Str, Any]"), movie_t).verdict is Verdict.DYNAMIC
        assert check(P("Dict[Int, Any]"), movie_t).verdict is Verdict.INCOMPATIBLE
        assert check(movie_t, P("Dict[Str, Any]")).verdict is Verdict.EXACT
        value = {"title": "Tampopo", "year": 1985}
        assert accept(value, movie_t) is value
        with pytest.raises(TypeMismatch):
            accept({"title": "Tampopo"}, movie_t)

    def test_dict_is_not_a_dataclass_record(self):
        point_t = from_annotation(Point)
        assert not conforms({"x": 1.0, "y": 2.0}, point_t)
        assert check(P("Dict[Str, Float]"), point_t).verdict is Verdict.INCOMPATIBLE

    def test_record_flows_into_its_class_and_bases(self):
        point_t = from_annotation(Point)
        assert check(point_t, ObjectT("typegen:Point")).verdict is Verdict.EXACT
        assert check(point_t, ObjectT("builtins:object")).verdict is Verdict.EXACT
        assert check(point_t, ObjectT("builtins:dict")).verdict is Verdict.INCOMPATIBLE
        assert check(from_annotation(Movie), ObjectT("builtins:dict")).verdict is Verdict.EXACT

    def test_unknown_record_class_is_decided_at_run_time(self):
        ghost = P("Record[no.such.Rec: a=Int]")
        assert check(ghost, P("Tuple[Int]")).verdict is Verdict.DYNAMIC
        assert check(ghost, P("Dict[Str, Int]")).verdict is Verdict.DYNAMIC
        assert check(ghost, INT).verdict is Verdict.INCOMPATIBLE
        assert check(P("Dict[Str, Int]"), ghost).verdict is Verdict.DYNAMIC


class TestObjectsVersusStructuralTypes:
    @pytest.mark.parametrize("src, dst, verdict", [
        ("Int", "Object[builtins:object]", Verdict.EXACT),
        ("List[Int]", "Object[builtins:list]", Verdict.EXACT),
        ("Dict[Str, Int]", "Object[builtins:dict]", Verdict.EXACT),
        ("Array[float64, 2]", "Object[numpy:ndarray]", Verdict.EXACT),
        ("Str", "Object[builtins:str]", Verdict.EXACT),
        ("Int", "Object[builtins:int]", Verdict.DYNAMIC),      # NumPy ints are not ``int``
        ("Bool", "Object[builtins:int]", Verdict.DYNAMIC),
        ("Dict[Str, Int]", "Object[collections:OrderedDict]", Verdict.DYNAMIC),
        ("Callable", "Object[builtins:object]", Verdict.DYNAMIC),
        ("Str", "Object[builtins:list]", Verdict.INCOMPATIBLE),
        ("Int", "Object[no.such:Class]", Verdict.INCOMPATIBLE),
        ("Object[builtins:object]", "Int", Verdict.DYNAMIC),
        ("Object[builtins:dict]", "Dict[Str, Int]", Verdict.DYNAMIC),
        ("Object[collections:OrderedDict]", "Dict[Str, Int]", Verdict.DYNAMIC),
        ("Object[builtins:dict]", "List[Int]", Verdict.INCOMPATIBLE),
        ("Object[builtins:set]", "Callable", Verdict.DYNAMIC),
        ("Object[no.such:Class]", "Int", Verdict.INCOMPATIBLE),
    ])
    def test_check(self, src, dst, verdict):
        assert check(P(src), P(dst)).verdict is verdict

    def test_runtime_agrees(self):
        assert accept(1, ObjectT("builtins:object")) == 1
        assert accept([1], ObjectT("builtins:list")) == [1]
        assert accept(np.int64(1), ObjectT("builtins:object")) == 1
        with pytest.raises(TypeMismatch):
            accept(np.int64(1), ObjectT("builtins:int"))
        with pytest.raises(TypeMismatch):
            accept("a", ObjectT("builtins:list"))


def test_delivered_type():
    from openramen.types import delivered
    assert delivered(P("Array[float64, 2]"), P("Array")) == P("Array[float64, 2]")   # exact: keep
    assert delivered(INT, FLOAT) == FLOAT                                           # coerced
    assert delivered(INT, P("Union[Float, Str]")) == FLOAT                          # chosen option
    assert delivered(INT, P("Union[Int, Float]")) == INT
    assert delivered(ANY, FLOAT) == FLOAT                                           # checked at run time
    assert delivered(ANY, ANY) == ANY
    assert delivered(P("List[Int]"), P("List[Float]")) == P("List[Float]")
    assert delivered(P("Optional[Int]"), INT) == INT


def test_a_failing_conversion_is_a_type_mismatch_not_a_crash():
    with pytest.raises(TypeMismatch, match="cannot convert Int to Float"):
        accept(10**400, FLOAT)
    with pytest.raises(TypeMismatch):
        accept([1, 10**400], P("List[Float]"))
