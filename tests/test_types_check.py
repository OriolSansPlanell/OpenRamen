"""Wire compatibility, join, unification."""

import collections
import random
from pathlib import Path

import numpy as np
import pytest

from openramen.types import (
    ANY, BOOL, COMPLEX, FLOAT, INT, NONE, PATH, STR,
    ArrayT, ListT, ObjectT, TupleT, UnifyError, VarT, Verdict,
    check, free_vars, join, optional, parse_type, substitute, unify, union,
)

from typegen import random_type

E, C, D, X = Verdict.EXACT, Verdict.COERCION, Verdict.DYNAMIC, Verdict.INCOMPATIBLE


def P(text):
    return parse_type(text)


@pytest.mark.parametrize("src, dst, verdict", [
    # identical and Any
    ("Int", "Int", E), ("Int", "Any", E), ("Any", "Int", D), ("Any", "Any", E),
    ("~T", "Int", D), ("Int", "~T", E),
    # scalar coercions: only lossless, only one direction
    ("Int", "Float", C), ("Int", "Complex", C), ("Float", "Complex", C),
    ("Float", "Int", X), ("Complex", "Float", X), ("Complex", "Int", X),
    ("Path", "Str", C), ("Str", "Path", C),
    ("Bool", "Int", X), ("Int", "Bool", X), ("Bool", "Float", X),
    ("Str", "Int", X), ("Int", "Str", X), ("Bytes", "Str", X), ("None", "Int", X),
    # arrays
    ("Array[float64, 2]", "Array[float64, 2]", E),
    ("Array[float64, 2]", "Array", E),
    ("Array[float64, 2]", "Array[_, 2]", E),
    ("Array[float64, 2]", "Array[float64, _]", E),
    ("Array", "Array[float64, 2]", D),
    ("Array[float64, _]", "Array[float64, 2]", D),
    ("Array[_, 2]", "Array[float64, 2]", D),
    ("Array[float64, 2]", "Array[float64, 3]", X),
    ("Array[_, 2]", "Array[_, 3]", X),
    ("Array[float32, 2]", "Array[float64, 2]", C),
    ("Array[uint8, 2]", "Array[float64, 2]", C),
    ("Array[uint8, 2]", "Array[int16, 2]", C),
    ("Array[float64, 2]", "Array[float32, 2]", X),
    ("Array[int64, 2]", "Array[uint8, 2]", X),
    ("Array[float64, 1]", "Array[int64, 1]", X),
    ("Array[float32, _]", "Array[float64, 2]", D),
    ("Array[~D, ~N]", "Array[float64, 2]", D),
    ("Array[float64, 2]", "Array[~D, ~N]", E),
    ("Array[str96, 1]", "Array[float64, 1]", X),
    # a scalar is never silently turned into an array, nor the reverse
    ("Int", "Array", X), ("Float", "Array[float64, 0]", X), ("Bool", "Array[bool, 0]", X),
    ("Complex", "Array[complex128, 0]", X), ("Str", "Array", X),
    ("Array[float64, 0]", "Float", X), ("List[Float]", "Array", X),
    ("Complex", "Union[Int, Float, Array]", X), ("Bool", "Union[Int, Float, Array]", X),
    # lists and tuples
    ("List[Int]", "List[Int]", E), ("List[Int]", "List", E), ("List", "List[Int]", D),
    ("List[Int]", "List[Float]", C), ("List[Float]", "List[Int]", X),
    ("List[Str]", "List[Int]", X), ("List[List[Int]]", "List[List[Float]]", C),
    ("Tuple[Int, Int]", "List[Int]", C), ("Tuple[Int, Float]", "List[Float]", C),
    ("Tuple[Int, Str]", "List[Int]", X), ("Tuple[Int, ...]", "List[Int]", C),
    ("Tuple[Int, ...]", "List[Float]", C), ("Tuple[Any, Int]", "List[Int]", D),
    ("Tuple[]", "List[Int]", C), ("List[Int]", "Tuple[Int, ...]", X),
    ("Tuple[Int, Str]", "Tuple[Int, Str]", E), ("Tuple[Int, Str]", "Tuple[Float, Str]", C),
    ("Tuple[Int, Str]", "Tuple[Int]", X), ("Tuple[Int]", "Tuple[Int, Str]", X),
    ("Tuple[Int, Str]", "Tuple[Str, Int]", X),
    ("Tuple[Int, Int]", "Tuple[Int, ...]", E), ("Tuple[Int, Int]", "Tuple[Float, ...]", C),
    ("Tuple[Int, Str]", "Tuple[Int, ...]", X), ("Tuple[]", "Tuple[Int, ...]", E),
    ("Tuple[Int, ...]", "Tuple[Int, Int]", D), ("Tuple[Str, ...]", "Tuple[Int, Int]", X),
    ("Tuple[Int, ...]", "Tuple[Float, ...]", C), ("Tuple[Int, ...]", "Tuple[]", D),
    ("Tuple[Int, ...]", "Tuple", E), ("Int", "Tuple[Int]", X),
    # dicts
    ("Dict[Str, Int]", "Dict[Str, Int]", E), ("Dict[Str, Int]", "Dict[Str, Float]", C),
    ("Dict[Str, Int]", "Dict", E), ("Dict", "Dict[Str, Int]", D),
    ("Dict[Int, Int]", "Dict[Float, Int]", X), ("Dict[Str, Str]", "Dict[Str, Int]", X),
    ("Dict[Any, Int]", "Dict[Str, Float]", D), ("List[Int]", "Dict", X),
    # unions and optionals
    ("Int", "Optional[Int]", E), ("None", "Optional[Int]", E),
    ("Optional[Int]", "Int", D), ("Optional[Int]", "Optional[Int]", E),
    ("Int", "Union[Int, Str]", E), ("Float", "Union[Int, Str]", X),
    ("Int", "Union[Float, Str]", C), ("Union[Int, Str]", "Union[Int, Str, None]", E),
    ("Union[Int, Str]", "Int", D), ("Union[Int, Str]", "Bool", X),
    ("Union[Int, Float]", "Float", D), ("Optional[Int]", "Optional[Float]", D),
    # iterators
    ("Iterator[Int]", "Iterator[Int]", E), ("Iterator[Int]", "Iterator[Float]", C),
    ("List[Int]", "Iterator[Int]", C), ("List[Int]", "Iterator[Float]", C),
    ("Tuple[Int, ...]", "Iterator[Int]", C), ("List[Str]", "Iterator[Int]", X),
    ("List", "Iterator[Int]", D), ("Iterator[Int]", "List[Int]", X),
    ("Tuple[Int, Int]", "Iterator[Int]", X),
    # enums, records
    ("Enum[p.C: a, b]", "Enum[p.C: a, b]", E), ("Enum[p.C: a, b]", "Enum[p.C: a]", X),
    ("Enum[p.C: a]", "Enum[q.C: a]", X), ("Enum[p.C: a]", "Str", X),
    ("Record[p.R: x=Int]", "Record[p.R: x=Int]", E),
    ("Record[p.R: x=Int]", "Record[p.R: x=Float]", X),
    ("Record[p.R: x=Any]", "Record[p.R: x=Float]", D),
    ("Record[p.R: x=Int]", "Record[p.S: x=Int]", X),
    ("Record[p.R: x=Int]", "Record[p.R: x=Int, y=Int]", X),
    ("Record[p.R: x=Int, y=Str]", "Record[p.R: y=Str, x=Int]", E),
    # callables
    ("Callable[[Int], Int]", "Callable[[Int], Int]", E),
    ("Callable[[Int], Int]", "Callable", E), ("Callable", "Callable[[Int], Int]", D),
    ("Callable[[Int], Int]", "Callable[..., Int]", E),
    ("Callable[[Int], Int]", "Callable[[Int, Int], Int]", X),
    ("Callable[[Int], Int]", "Callable[[Int], Float]", X),
    ("Callable[[Any], Int]", "Callable[[Int], Int]", E),
    ("Callable[[Int], Any]", "Callable[[Int], Int]", D),
    ("Callable[[Int], Int]", "Callable[[Any], Int]", D),
    ("Int", "Callable", X),
])
def test_check(src, dst, verdict):
    assert check(P(src), P(dst)).verdict is verdict


@pytest.mark.parametrize("src, dst, value, expected", [
    ("Int", "Float", 3, 3.0),
    ("Int", "Complex", 3, 3 + 0j),
    ("Float", "Complex", 1.5, 1.5 + 0j),
    ("Path", "Str", Path("a") / "b", str(Path("a") / "b")),
    ("Str", "Path", "a/b", Path("a/b")),
    ("List[Int]", "List[Float]", [1, 2], [1.0, 2.0]),
    ("List[List[Int]]", "List[List[Float]]", [[1], [2, 3]], [[1.0], [2.0, 3.0]]),
    ("Tuple[Int, Float]", "List[Float]", (1, 2.5), [1.0, 2.5]),
    ("Tuple[Int, ...]", "List[Int]", (1, 2), [1, 2]),
    ("Tuple[Int, ...]", "List[Float]", (1, 2), [1.0, 2.0]),
    ("Tuple[]", "List[Int]", (), []),
    ("Tuple[Int, Str]", "Tuple[Float, Str]", (1, "a"), (1.0, "a")),
    ("Tuple[Int, Int]", "Tuple[Float, ...]", (1, 2), (1.0, 2.0)),
    ("Tuple[Int, ...]", "Tuple[Float, ...]", (1, 2, 3), (1.0, 2.0, 3.0)),
    ("Dict[Str, Int]", "Dict[Str, Float]", {"a": 1}, {"a": 1.0}),
    ("Int", "Union[Float, Str]", 2, 2.0),
])
def test_coercion_converts_values(src, dst, value, expected):
    compat = check(P(src), P(dst))
    assert compat.verdict is C and compat.convert is not None
    out = compat.convert(value)
    assert out == expected and type(out) is type(expected)
    if isinstance(out, (list, tuple)) and out:
        assert [type(x) for x in out] == [type(x) for x in expected]


def test_array_coercions_convert_values():
    conv = check(P("Array[uint8, 1]"), P("Array[float64, 1]")).convert
    out = conv(np.array([1, 2], dtype="uint8"))
    assert out.dtype == np.float64 and out.tolist() == [1.0, 2.0]



def test_iterator_coercions_are_lazy():
    conv = check(P("List[Int]"), P("Iterator[Float]")).convert
    it = conv([1, 2])
    assert next(it) == 1.0 and isinstance(next(it), float)
    conv = check(P("Iterator[Int]"), P("Iterator[Float]")).convert
    assert list(conv(iter([1, 2]))) == [1.0, 2.0]
    conv = check(P("List[Int]"), P("Iterator[Int]")).convert
    assert list(conv([1, 2])) == [1, 2]


def test_only_coercion_verdicts_carry_a_converter():
    for seed in range(300):
        rng = random.Random(seed)
        a, b = random_type(rng), random_type(rng)
        compat = check(a, b)
        assert (compat.convert is not None) == (compat.verdict is C), (a, b)


def test_incompatible_carries_a_reason():
    compat = check(STR, INT)
    assert not compat.ok and "Str" in compat.reason and "Int" in compat.reason
    assert check(INT, FLOAT).ok


class TestObjects:
    def test_subclass_flows_into_base(self):
        base = ObjectT("collections:OrderedDict")
        derived = ObjectT("builtins:dict")
        assert check(base, derived).verdict is E      # OrderedDict is a dict
        assert check(derived, base).verdict is D      # may or may not be one
        assert check(base, ObjectT("builtins:list")).verdict is X

    def test_unresolvable_classes_only_match_themselves(self):
        ghost = ObjectT("no.such.module:Thing")
        assert check(ghost, ghost).verdict is E
        assert check(ghost, ObjectT("builtins:dict")).verdict is X
        assert check(ObjectT("builtins:dict"), ghost).verdict is X

    def test_path_to_a_non_class_is_not_resolved(self):
        assert check(ObjectT("os:getcwd"), ObjectT("builtins:object")).verdict is X

    def test_join_picks_the_base_class(self):
        a, b = ObjectT("collections:OrderedDict"), ObjectT("builtins:dict")
        assert join(a, b) == b and join(b, a) == b
        assert collections.OrderedDict  # keep the import honest
        assert join(a, ObjectT("builtins:list")) is None


@pytest.mark.parametrize("a, b, expected", [
    ("Int", "Int", "Int"), ("Int", "Float", "Float"), ("Float", "Complex", "Complex"),
    ("Int", "Complex", "Complex"), ("Int", "Str", None), ("Bool", "Int", None),
    ("Path", "Str", None), ("Int", "Any", "Any"), ("Any", "List[Int]", "Any"),
    ("Int", "None", "Optional[Int]"), ("None", "List[Int]", "Optional[List[Int]]"),
    ("Optional[Int]", "None", "Optional[Int]"), ("Optional[Int]", "Int", "Optional[Int]"),
    ("Union[Int, Str]", "Str", "Union[Int, Str]"), ("Union[Int, Str]", "Float", None),
    ("Union[Int, Str]", "Union[Int, Str, None]", "Union[Int, Str, None]"),
    ("Union[Int, Str]", "Union[Int, Bool]", None),
    ("List[Int]", "List[Float]", "List[Float]"), ("List[Int]", "List[Str]", None),
    ("List[Int]", "Tuple[Int, ...]", None),
    ("Iterator[Int]", "Iterator[Float]", "Iterator[Float]"),
    ("Dict[Str, Int]", "Dict[Str, Float]", "Dict[Str, Float]"),
    ("Dict[Str, Int]", "Dict[Int, Int]", None),
    ("Tuple[Int, Str]", "Tuple[Float, Str]", "Tuple[Float, Str]"),
    ("Tuple[Int, Str]", "Tuple[Int]", None), ("Tuple[Int, ...]", "Tuple[Int]", None),
    ("Tuple[Int, ...]", "Tuple[Float, ...]", "Tuple[Float, ...]"),
    ("Array[float32, 2]", "Array[float64, 2]", "Array[float64, 2]"),
    ("Array[uint8, 2]", "Array[int8, 2]", "Array[int16, 2]"),
    ("Array[float64, 2]", "Array[float64, 3]", None),
    ("Array[float64, 2]", "Array[float64, _]", "Array[float64, _]"),
    ("Array[float64, 2]", "Array[_, 2]", "Array[_, 2]"),
    ("Array[float64, 1]", "Float", None),
    ("Enum[p.C: a]", "Enum[p.C: a]", "Enum[p.C: a]"), ("Enum[p.C: a]", "Enum[p.D: a]", None),
])
def test_join(a, b, expected):
    result = join(P(a), P(b))
    assert result == (None if expected is None else P(expected))
    assert join(P(b), P(a)) == result  # symmetric


@pytest.mark.parametrize("seed", range(300))
def test_properties_on_random_types(seed):
    rng = random.Random(seed)
    a, b = random_type(rng), random_type(rng)
    assert check(a, a).verdict is E          # reflexive
    assert check(a, ANY).verdict is E        # everything flows into Any
    assert join(a, a) == a                   # idempotent
    assert join(a, b) == join(b, a)          # symmetric
    assert union(a, b) == union(b, a)
    assert union(a, a) == a
    j = join(a, b)
    if j is not None and free_vars(j) == set():
        # both sides can be wired into their join without a run-time check
        assert check(substitute(a, {}), j).verdict in (E, C, D)
    check(a, b), check(b, a)                 # never raises


class TestUnify:
    def solve(self, pattern, *actuals):
        bindings = {}
        for actual in actuals:
            unify(P(pattern), None if actual is None else P(actual), bindings)
        return bindings

    def test_binds_a_variable(self):
        assert self.solve("~T", "Int") == {"T": INT}
        assert self.solve("List[~T]", "List[Str]") == {"T": STR}
        assert self.solve("Dict[~K, ~V]", "Dict[Str, Int]") == {"K": STR, "V": INT}
        assert self.solve("Iterator[~T]", "Iterator[Int]") == {"T": INT}
        assert self.solve("Iterator[~T]", "List[Int]") == {"T": INT}

    def test_unknown_actuals_bind_nothing(self):
        assert self.solve("~T", "Any") == {}
        assert self.solve("~T", None) == {}
        assert self.solve("List[~T]", "List") == {}
        assert self.solve("~T", "~U") == {}

    def test_structural_mismatch_binds_nothing(self):
        assert self.solve("List[~T]", "Int") == {}
        assert self.solve("Array[~D, ~N]", "List[Int]") == {}
        assert self.solve("Tuple[~T, ~U]", "Tuple[Int]") == {}
        assert self.solve("Dict[~K, ~V]", "List[Int]") == {}
        assert self.solve("Callable[[~T], ~R]", "Int") == {}

    def test_same_variable_twice_joins(self):
        assert self.solve("~T", "Int", "Float") == {"T": FLOAT}
        assert self.solve("~T", "Float", "Int") == {"T": FLOAT}
        assert self.solve("~T", "Int", "None") == {"T": optional(INT)}
        assert self.solve("~T", "Int", "Any") == {"T": INT}

    def test_conflict_raises(self):
        with pytest.raises(UnifyError) as err:
            self.solve("~T", "Int", "Str")
        assert err.value.var == "T" and "~T" in str(err.value)
        with pytest.raises(UnifyError):
            self.solve("Tuple[~T, ~T]", "Tuple[Bool, Str]")

    def test_arrays(self):
        assert self.solve("Array[~D, ~N]", "Array[float32, 2]") == {"D": "float32", "N": 2}
        assert self.solve("Array[~D, ~N]", "Array[_, 2]") == {"N": 2}
        assert self.solve("Array[~D, ~N]", "Array") == {}
        # dtypes promote like NumPy does, dimensions must agree
        assert self.solve("Array[~D, _]", "Array[uint8, 1]", "Array[float32, 1]") \
            == {"D": "float32"}
        with pytest.raises(UnifyError):
            self.solve("Array[_, ~N]", "Array[_, 1]", "Array[_, 2]")
        with pytest.raises(UnifyError):
            self.solve("Array[~D, _]", "Array[float64, 1]", "Array[datetime64, 1]")

    def test_tuples(self):
        assert self.solve("Tuple[~A, ~B]", "Tuple[Int, Str]") == {"A": INT, "B": STR}
        assert self.solve("Tuple[~T, ...]", "Tuple[Int, Float]") == {"T": FLOAT}
        assert self.solve("List[~T]", "Tuple[Int, Float]") == {"T": FLOAT}
        assert self.solve("Tuple[~A, ~B]", "Tuple[Int, ...]") == {}

    def test_optional_pattern(self):
        assert self.solve("Optional[~T]", "Int") == {"T": INT}
        assert self.solve("Optional[~T]", "None") == {}
        assert self.solve("Optional[~T]", "Optional[Str]") == {"T": STR}
        assert self.solve("Optional[~T]", "Union[Int, Str, None]") == {"T": union(INT, STR)}

    def test_callables(self):
        assert self.solve("Callable[[~A], ~R]", "Callable[[Int], Str]") == {"A": INT, "R": STR}
        assert self.solve("Callable[[~A], ~R]", "Callable[..., Str]") == {"R": STR}
        assert self.solve("Callable[[~A], ~R]", "Callable[[Int, Int], Str]") == {"R": STR}

    def test_generic_node_end_to_end(self):
        # select(cond: Bool, a: ~T, b: ~T) -> ~T wired with Int and Float
        bindings = {}
        unify(VarT("T"), INT, bindings)
        unify(VarT("T"), FLOAT, bindings)
        resolved = substitute(VarT("T"), bindings)
        assert resolved == FLOAT
        assert check(INT, resolved).verdict is C and check(FLOAT, resolved).verdict is E

    @pytest.mark.parametrize("seed", range(300))
    def test_unify_recovers_what_substitute_put_in(self, seed):
        rng = random.Random(seed)
        pattern = random_type(rng, vars_everywhere=False)
        truth = {"T": random_type(rng, variables=False),
                 "U": random_type(rng, variables=False),
                 "D": rng.choice(["float64", "uint8", "int32"]),
                 "N": rng.randint(0, 4)}
        concrete = substitute(pattern, truth)
        solved = {}
        unify(pattern, concrete, solved)
        assert substitute(pattern, solved) == concrete
        for name, value in solved.items():
            assert value == truth[name]


def test_join_of_scalar_and_path_like_is_never_guessed():
    # Str <-> Path coerce both ways, so neither is "wider"
    assert join(STR, PATH) is None
    assert join(BOOL, COMPLEX) is None
    assert join(NONE, NONE) == NONE
    assert join(ListT(INT), TupleT((INT,), True)) is None
