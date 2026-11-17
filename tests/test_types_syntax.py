"""Text form of types: parsing, formatting, unions."""

import random

import pytest

from openramen.types import (
    ANY, BOOL, BYTES, COMPLEX, FLOAT, INT, NONE, PATH, STR,
    ArrayT, CallableT, DictT, EnumT, IteratorT, ListT, ObjectT, RecordT,
    TupleT, TypeSyntaxError, UnionT, VarT,
    format_type, free_vars, optional, parse_type, substitute, union,
)

from typegen import random_type


@pytest.mark.parametrize("text, expected", [
    ("Bool", BOOL), ("Int", INT), ("Float", FLOAT), ("Complex", COMPLEX),
    ("Str", STR), ("Bytes", BYTES), ("Path", PATH), ("None", NONE), ("Any", ANY),
    ("~T", VarT("T")),
    ("Array", ArrayT()),
    ("Array[_, _]", ArrayT(None, None)),
    ("Array[float64, 2]", ArrayT("float64", 2)),
    ("Array[uint8, _]", ArrayT("uint8", None)),
    ("Array[_, 0]", ArrayT(None, 0)),
    ("Array[~D, ~N]", ArrayT(VarT("D"), VarT("N"))),
    ("List", ListT(ANY)),
    ("List[Int]", ListT(INT)),
    ("List[List[Float]]", ListT(ListT(FLOAT))),
    ("Tuple", TupleT((ANY,), variadic=True)),
    ("Tuple[]", TupleT(())),
    ("Tuple[Int]", TupleT((INT,))),
    ("Tuple[Int, Str]", TupleT((INT, STR))),
    ("Tuple[Int, ...]", TupleT((INT,), variadic=True)),
    ("Dict", DictT(ANY, ANY)),
    ("Dict[Str, Int]", DictT(STR, INT)),
    ("Iterator", IteratorT(ANY)),
    ("Iterator[Str]", IteratorT(STR)),
    ("Optional[Int]", UnionT((INT, NONE))),
    ("Union[Int, Str]", UnionT((INT, STR))),
    ("Enum[pkg.Color]", EnumT("pkg.Color", ())),
    ("Enum[pkg.Color: red, green]", EnumT("pkg.Color", ("red", "green"))),
    ("Record[pkg.Point]", RecordT("pkg.Point", ())),
    ("Record[pkg.Point: x=Float, y=Float]",
     RecordT("pkg.Point", (("x", FLOAT), ("y", FLOAT)))),
    ("Object[numpy.random:Generator]", ObjectT("numpy.random:Generator")),
    ("Object[pkg.mod:Outer.Inner]", ObjectT("pkg.mod:Outer.Inner")),
    ("Callable", CallableT(None, ANY)),
    ("Callable[..., Int]", CallableT(None, INT)),
    ("Callable[[], Int]", CallableT((), INT)),
    ("Callable[[Int, Str], Bool]", CallableT((INT, STR), BOOL)),
])
def test_parse(text, expected):
    assert parse_type(text) == expected


@pytest.mark.parametrize("text", [
    "  Int  ", "Array[ float64 ,2 ]", "List[\tInt\n]", "Dict[Str,Int]",
    "Tuple[Int,...]", "Record[p.P:x=Int,y=Str]",
])
def test_whitespace_is_ignored(text):
    compact = "".join(text.split())
    assert parse_type(text) == parse_type(compact)


@pytest.mark.parametrize("text", [
    "", "   ", "Foo", "int", "List[", "List[Int", "List[Int]]", "List[]",
    "Array[float64]", "Array[float64, x]", "Array[2, 2]", "Array[float64, -1]",
    "Tuple[..., Int]", "Tuple[Int, Str, ...]", "Dict[Int]", "Dict[Int, Str, Str]",
    "Union[]", "Optional[]", "Optional[Int, Str]", "Enum[]", "Enum[c: ]",
    "Enum[c: a.b]", "Record[p: x]", "Record[p: x=]", "Record[p: x=Int, x=Str]",
    "Object[]", "Object[module]", "Object[a:b:c]", "Callable[Int, Int]",
    "Callable[[Int]]", "~", "~1", "~a.b", "Int Str", "Int,", "List[Int] x",
    "Lisé", "Int!", "Array[float64; 2]", "Enum[a..b: x]", "Object[a.:b]",
])
def test_syntax_errors(text):
    with pytest.raises(TypeSyntaxError):
        parse_type(text)


def test_parse_accepts_type_instances_and_rejects_other_objects():
    assert parse_type(INT) is INT
    with pytest.raises(TypeError):
        parse_type(42)  # type: ignore[arg-type]


def test_str_and_repr():
    t = ListT(optional(INT))
    assert str(t) == "List[Optional[Int]]"
    assert "List[Optional[Int]]" in repr(t)


def test_types_are_hashable_and_comparable():
    a, b = parse_type("Dict[Str, List[Array[float64, 2]]]"), \
        parse_type("Dict[Str, List[Array[float64, 2]]]")
    assert a == b and hash(a) == hash(b)
    assert len({a, b, INT, parse_type("Int")}) == 2


def test_types_are_immutable():
    with pytest.raises(Exception):
        INT.name = "Float"  # type: ignore[misc]
    with pytest.raises(Exception):
        ListT(INT).elem = STR  # type: ignore[misc]


def test_invalid_constructions():
    with pytest.raises(ValueError):
        ArrayT("float 64", 2)
    with pytest.raises(ValueError):
        ArrayT("float64", -1)
    with pytest.raises(ValueError):
        ArrayT("float64", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TupleT((INT, STR), variadic=True)
    with pytest.raises(ValueError):
        union()


class TestUnion:
    def test_single_option_collapses(self):
        assert union(INT) is INT
        assert union(INT, INT) == INT

    def test_order_does_not_matter(self):
        assert union(INT, STR, NONE) == union(NONE, STR, INT)

    def test_nested_unions_flatten(self):
        assert union(INT, union(STR, FLOAT)) == union(INT, STR, FLOAT)

    def test_any_absorbs(self):
        assert union(INT, ANY) is ANY

    def test_optional(self):
        assert optional(INT) == union(INT, NONE)
        assert optional(optional(INT)) == optional(INT)
        assert optional(NONE) == NONE
        assert format_type(optional(INT)) == "Optional[Int]"

    def test_three_way_union_with_none_is_not_called_optional(self):
        assert format_type(union(INT, STR, NONE)).startswith("Union[")


def test_free_vars_and_substitute():
    t = parse_type("Dict[~K, List[Callable[[~A, Array[~D, ~N]], ~R]]]")
    assert free_vars(t) == {"K", "A", "D", "N", "R"}
    bound = substitute(t, {"K": STR, "A": INT, "D": "float32", "N": 2, "R": BOOL})
    assert str(bound) == "Dict[Str, List[Callable[[Int, Array[float32, 2]], Bool]]]"
    assert free_vars(bound) == set()
    # unbound variables become unknown, never stay as variables
    assert str(substitute(t, {})) == "Dict[Any, List[Callable[[Any, Array[_, _]], Any]]]"


def test_substitute_ignores_bindings_of_the_wrong_kind():
    t = parse_type("Array[~D, ~N]")
    assert substitute(t, {"D": 3, "N": "float64"}) == ArrayT(None, None)
    assert substitute(t, {"N": True}) == ArrayT(None, None)
    assert substitute(VarT("T"), {"T": "float64"}) == ANY


# -- property-style tests over many random types ---------------------------

SEEDS = range(400)


@pytest.mark.parametrize("seed", SEEDS)
def test_roundtrip_random(seed):
    t = random_type(random.Random(seed))
    text = format_type(t)
    assert parse_type(text) == t
    assert format_type(parse_type(text)) == text


@pytest.mark.parametrize("seed", SEEDS)
def test_substitute_removes_all_variables(seed):
    t = random_type(random.Random(seed))
    once = substitute(t, {})
    assert free_vars(once) == set()
    assert substitute(once, {}) == once
