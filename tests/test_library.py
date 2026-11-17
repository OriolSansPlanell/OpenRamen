"""NodeSpec, the registry and building specs from functions."""

import copy
import pickle
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import pytest

from openramen import types as T
from openramen.library import (
    MISSING, NodeResultError, NodeSpec, PinSpec, Registry, SpecError,
    UnknownSpecError, default_registry, spec_from_function,
)


def test_missing_is_a_singleton_that_survives_copy_and_pickle():
    assert repr(MISSING) == "MISSING" and not MISSING
    assert copy.copy(MISSING) is MISSING and copy.deepcopy(MISSING) is MISSING
    assert pickle.loads(pickle.dumps(MISSING)) is MISSING
    assert type(MISSING)() is MISSING


class TestPinSpec:
    def test_defaults(self):
        pin = PinSpec("x")
        assert pin.type == T.ANY and pin.required and pin.access == "read"
        assert not PinSpec("x", default=None).required
        assert not PinSpec("x", default=0).required

    def test_type_may_be_given_as_text(self):
        assert PinSpec("x", "Array[float64, 2]").type == T.ArrayT("float64", 2)
        with pytest.raises(T.TypeSyntaxError):
            PinSpec("x", "Nonsense")

    @pytest.mark.parametrize("name", ["", "1x", "a b", "a.b", "class", "def", None, 3])
    def test_invalid_names(self, name):
        with pytest.raises(SpecError):
            PinSpec(name)

    def test_invalid_fields(self):
        with pytest.raises(SpecError):
            PinSpec("x", access="write")
        with pytest.raises(SpecError):
            PinSpec("x", kind="variadic")

    def test_is_immutable(self):
        with pytest.raises(Exception):
            PinSpec("x").name = "y"  # type: ignore[misc]


def spec(**overrides):
    fields = dict(id="t.node", inputs=(PinSpec("a"), PinSpec("b", default=1)),
                  outputs=(PinSpec("out"),), fn=lambda a, b=1: a + b)
    fields.update(overrides)
    return NodeSpec(**fields)


class TestNodeSpec:
    def test_lookups(self):
        s = spec(config=(PinSpec("mode", default="x"),), fn=lambda a, b=1, mode="x": a)
        assert s.input("a").name == "a" and s.input("zzz") is None
        assert s.output("out").name == "out" and s.output("a") is None
        assert s.config_pin("mode").default == "x" and s.config_pin("a") is None

    @pytest.mark.parametrize("bad_id", ["", "a b", "a..b", ".a", "a.", "1a", "a-b", None, 3])
    def test_invalid_ids(self, bad_id):
        with pytest.raises(SpecError):
            spec(id=bad_id)

    @pytest.mark.parametrize("overrides, message", [
        (dict(inputs=(PinSpec("a"), PinSpec("a"))), "duplicate input"),
        (dict(inputs=(PinSpec("a"),), config=(PinSpec("a"),)), "duplicate input"),
        (dict(outputs=(PinSpec("o"), PinSpec("o"))), "duplicate output"),
        (dict(outputs=(PinSpec("o", aliases="nope"),)), "aliases unknown input"),
        (dict(inputs=(PinSpec("a"), PinSpec("b", kind="positional"))), "follows a keyword"),
        (dict(config=(PinSpec("c", kind="positional"),)), "must be a keyword"),
        (dict(executor="gpu"), "executor"),
        (dict(tier="gold"), "tier"),
        (dict(fn=None), "needs a function or a target"),
        (dict(fn=42), "not callable"),
        (dict(inputs=("a",)), "PinSpec"),
    ])
    def test_inconsistent_specs_are_refused(self, overrides, message):
        with pytest.raises(SpecError, match=message):
            spec(**overrides)

    def test_same_name_as_input_and_output_is_fine(self):
        s = spec(inputs=(PinSpec("x"),), outputs=(PinSpec("x", aliases="x"),), fn=lambda x: x)
        assert s.call({"x": 3}) == {"x": 3}

    def test_equality_ignores_the_function_object(self):
        assert spec() == spec()
        assert spec() != spec(summary="different")

    def test_is_immutable(self):
        with pytest.raises(Exception):
            spec().id = "other"  # type: ignore[misc]


class TestCalling:
    def test_inputs_left_out_use_the_function_default(self):
        s = spec()
        assert s.call({"a": 1}) == {"out": 2}
        assert s.call({"a": 1, "b": 10}) == {"out": 11}
        with pytest.raises(TypeError):
            s.call({})

    def test_positional_only_parameters(self):
        def f(a, b=2, /, c=3, *, d=4):
            return (a, b, c, d)
        s = spec_from_function(f, "t.pos")
        assert [p.kind for p in s.inputs] == ["positional", "positional", "keyword", "keyword"]
        assert s.call({"a": 1})["out"] == (1, 2, 3, 4)
        assert s.call({"a": 1, "b": 9, "d": 7})["out"] == (1, 9, 3, 7)
        with pytest.raises(TypeError, match="'a' before it is not"):
            s.call({"b": 9})

    def test_config_is_passed_by_keyword(self):
        def f(x, scale=1.0):
            return x * scale
        s = spec_from_function(f, "t.cfg", config=("scale",))
        assert [p.name for p in s.inputs] == ["x"] and [p.name for p in s.config] == ["scale"]
        assert s.call({"x": 2}) == {"out": 2.0}
        assert s.call({"x": 2}, {"scale": 3.0}) == {"out": 6.0}
        assert s.call({"x": 2}, {"other": 1}) == {"out": 2.0}   # unknown keys never reach fn

    def test_no_outputs(self):
        s = spec(outputs=(), fn=lambda a, b=1: "ignored")
        assert s.call({"a": 1}) == {}

    def test_single_output_takes_the_whole_result(self):
        s = spec(fn=lambda a, b=1: (a, b))
        assert s.call({"a": 1}) == {"out": (1, 1)}
        assert spec(fn=lambda a, b=1: None).call({"a": 1}) == {"out": None}

    def test_several_outputs_unpack(self):
        class QR(NamedTuple):
            q: int
            r: int

        two = (PinSpec("q"), PinSpec("r"))
        assert spec(outputs=two, fn=lambda a, b=1: divmod(a, b)).call({"a": 7, "b": 2}) \
            == {"q": 3, "r": 1}
        assert spec(outputs=two, fn=lambda a, b=1: [a, b]).call({"a": 7}) == {"q": 7, "r": 1}
        assert spec(outputs=two, fn=lambda a, b=1: QR(a, b)).call({"a": 7}) == {"q": 7, "r": 1}
        assert spec(outputs=two, fn=lambda a, b=1: {"r": a, "q": b}).call({"a": 7}) \
            == {"q": 1, "r": 7}

    @pytest.mark.parametrize("result", [
        (1,), (1, 2, 3), 5, None, "ab", b"ab", {"q": 1}, {"q": 1, "x": 2}, np.zeros(2),
    ])
    def test_wrong_shape_of_result(self, result):
        s = spec(outputs=(PinSpec("q"), PinSpec("r")), fn=lambda a, b=1: result)
        with pytest.raises(NodeResultError):
            s.call({"a": 1})

    def test_target_is_imported_lazily(self):
        s = NodeSpec("t.sqrt", inputs=(PinSpec("x", kind="positional"),),
                     outputs=(PinSpec("out"),), target="math:sqrt")
        assert s.fn is None
        assert s.call({"x": 9.0}) == {"out": 3.0}
        assert s.fn is not None                      # cached after first use
        nested = NodeSpec("t.join", inputs=(PinSpec("a", kind="positional"),),
                          outputs=(PinSpec("out"),), target="os:path.basename")
        assert nested.call({"a": "/x/y.txt"}) == {"out": "y.txt"}

    @pytest.mark.parametrize("target", [
        "math", "math:", ":sqrt", "no.such.module:f", "math:no_such", "math:pi",
    ])
    def test_bad_targets(self, target):
        s = NodeSpec("t.bad", target=target)
        with pytest.raises(SpecError):
            s.resolve()


class TestSpecFromFunction:
    def test_reads_annotations_defaults_and_docstring(self):
        def scale(image: np.ndarray, factor: float = 2.0, *, clip: bool = False) -> np.ndarray:
            """Multiply an image by a factor.

            More text.
            """
            return image * factor

        s = spec_from_function(scale)
        assert s.id.endswith(".scale") and s.summary == "Multiply an image by a factor."
        assert [(p.name, str(p.type), p.required) for p in s.inputs] == [
            ("image", "Array[_, _]", True), ("factor", "Float", False), ("clip", "Bool", False)]
        assert s.inputs[1].default == 2.0
        assert [(p.name, str(p.type)) for p in s.outputs] == [("out", "Array[_, _]")]
        assert s.tier == "curated" and s.pure and s.fn is scale

    def test_unannotated_parameters(self):
        def f(a, b=3, c=None, d="x", e=1.5):
            return a
        s = spec_from_function(f, "t.f")
        assert [str(p.type) for p in s.inputs] == ["Any", "Int", "Any", "Str", "Float"]
        assert str(s.outputs[0].type) == "Any"

    def test_type_overrides_win(self):
        def f(a: int, b) -> float:
            return a
        s = spec_from_function(f, "t.f", types={"a": "Float", "b": T.ListT(T.INT), "out": "~T"})
        assert [str(p.type) for p in s.inputs] == ["Float", "List[Int]"]
        assert str(s.outputs[0].type) == "~T"

    def test_openramen_types_work_as_annotations(self):
        def f(a: T.ArrayT("uint16", 2), b: "Array[float64, 1]") -> T.FLOAT:  # noqa: F821
            return 0.0
        s = spec_from_function(f, "t.f")
        assert [str(p.type) for p in s.inputs] == ["Array[uint16, 2]", "Array[float64, 1]"]
        assert s.outputs[0].type == T.FLOAT

    def test_return_none_means_no_outputs(self):
        def f(a) -> None:
            pass
        assert spec_from_function(f, "t.f").outputs == ()
        assert [p.name for p in spec_from_function(f, "t.g", outputs=("done",)).outputs] == ["done"]

    def test_named_outputs_take_tuple_element_types(self):
        def f(a: int, b: int) -> Tuple[int, float]:
            return a, b / 2
        s = spec_from_function(f, "t.f", outputs=("whole", "half"))
        assert [(p.name, str(p.type)) for p in s.outputs] == [("whole", "Int"), ("half", "Float")]
        assert s.call({"a": 4, "b": 3}) == {"whole": 4, "half": 1.5}
        mismatch = spec_from_function(f, "t.g", outputs=("x", "y", "z"))
        assert [str(p.type) for p in mismatch.outputs] == ["Any", "Any", "Any"]
        single = spec_from_function(f, "t.h", outputs="pair")
        assert [(p.name, str(p.type)) for p in single.outputs] == [("pair", "Tuple[Int, Float]")]

    def test_named_outputs_take_record_field_types(self):
        class Stats(NamedTuple):
            mean: float
            count: int

        def f(xs: list) -> Stats:
            return Stats(sum(xs) / len(xs), len(xs))

        s = spec_from_function(f, "t.f", outputs=("mean", "count"))
        assert [str(p.type) for p in s.outputs] == ["Float", "Int"]
        assert s.call({"xs": [1, 3]}) == {"mean": 2.0, "count": 2}

    def test_mutates_docs_and_extra_fields(self):
        def f(buffer: list, value: int = 0) -> None:
            buffer.append(value)
        s = spec_from_function(f, "t.f", mutates=("buffer",), docs={"buffer": "target"},
                               category="Lists", pure=False, executor="inline", tier="annotated")
        assert s.input("buffer").access == "mutate" and s.input("value").access == "read"
        assert s.input("buffer").doc == "target"
        assert (s.category, s.pure, s.executor, s.tier) == ("Lists", False, "inline", "annotated")

    def test_errors(self):
        def variadic(*args):
            return args

        def keywords(**kwargs):
            return kwargs

        def f(a, /, b):
            return a

        for fn in (variadic, keywords):
            with pytest.raises(SpecError, match="variadic"):
                spec_from_function(fn, "t.v")
        with pytest.raises(SpecError, match="no parameter named"):
            spec_from_function(f, "t.f", config=("zzz",))
        with pytest.raises(SpecError, match="no parameter named"):
            spec_from_function(f, "t.f", mutates=("zzz",))
        with pytest.raises(SpecError, match="positional-only"):
            spec_from_function(f, "t.f", config=("a",))
        with pytest.raises(SpecError):
            spec_from_function(42, "t.f")  # type: ignore[arg-type]
        with pytest.raises(SpecError):
            spec_from_function(lambda x: x)         # no usable default id
        with pytest.raises(T.TypeSyntaxError):
            spec_from_function(f, "t.f", types={"a": "Nonsense"})

    def test_builtins_with_signatures_work(self):
        s = spec_from_function(divmod, "t.divmod", outputs=("q", "r"))
        assert [p.kind for p in s.inputs] == ["positional", "positional"]
        assert s.call({"x": 7, "y": 2}) == {"q": 3, "r": 1}

    def test_broken_annotations_fall_back_to_any(self):
        def f(a: "NoSuchName", b: "int") -> "AlsoMissing":  # noqa: F821
            return a
        s = spec_from_function(f, "t.f")
        assert [(p.name, str(p.type)) for p in s.inputs] == [("a", "Any"), ("b", "Int")]
        assert str(s.outputs[0].type) == "Any"

    def test_one_bad_annotation_does_not_lose_the_others(self):
        def f(a: "NoSuchName", b: "Optional[float]", c: "List[Int]", d: bool = False  # noqa: F821
              ) -> "Array[uint8, 2]":  # noqa: F821
            return a
        s = spec_from_function(f, "t.f")
        assert [str(p.type) for p in s.inputs] == ["Any", "Optional[Float]", "List[Int]", "Bool"]
        assert str(s.outputs[0].type) == "Array[uint8, 2]"

    def test_default_optional(self):
        def f(a: Optional[int] = None, b: Any = None) -> Optional[int]:
            return a
        s = spec_from_function(f, "t.f")
        assert [str(p.type) for p in s.inputs] == ["Optional[Int]", "Any"]
        assert not s.inputs[0].required and s.inputs[0].default is None


class TestRegistry:
    def test_register_and_get(self):
        r = Registry()
        s = r.register(spec())
        assert r.get("t.node") is s and r.find("t.node") is s and "t.node" in r
        assert r.find("zzz") is None and "zzz" not in r and len(r) == 1 and list(r) == [s]
        with pytest.raises(UnknownSpecError) as err:
            r.get("zzz")
        assert "zzz" in str(err.value) and isinstance(err.value, KeyError)

    def test_duplicates(self):
        r = Registry()
        r.register(spec())
        with pytest.raises(SpecError, match="already registered"):
            r.register(spec())
        newer = spec(summary="newer")
        assert r.register(newer, replace=True) is newer and r.get("t.node") is newer
        with pytest.raises(SpecError):
            r.register("not a spec")  # type: ignore[arg-type]

    def test_decorator(self):
        r = Registry()

        @r.node("my.double", category="Demo")
        def double(x: float) -> float:
            """Twice x."""
            return 2 * x

        assert double(2.0) == 4.0                     # the function is unchanged
        assert double.spec is r.get("my.double")
        assert r.get("my.double").category == "Demo"
        assert r.get("my.double").summary == "Twice x."
        with pytest.raises(SpecError):
            r.node("my.double")(double)
        r.node("my.double", replace=True)(double)
        r.node("my.len")(len)                         # builtins cannot hold .spec: no crash
        assert r.get("my.len").call({"obj": [1, 2]}) == {"out": 2}

    def test_categories_copy_update(self):
        r = Registry()
        r.register(spec(id="b.two", category="Zeta"))
        r.register(spec(id="a.one", category="Alpha"))
        r.register(spec(id="a.zero", category="Zeta"))
        assert list(r.categories()) == ["Alpha", "Zeta"]
        assert [s.id for s in r.categories()["Zeta"]] == ["a.zero", "b.two"]
        assert r.ids() == ["a.one", "a.zero", "b.two"]
        clone = r.copy()
        clone.register(spec(id="c.three"))
        assert "c.three" not in r
        other = Registry()
        other.update(clone)
        assert len(other) == 4
        with pytest.raises(SpecError):
            other.update(r)
        other.update(r, replace=True)

    def test_default_registry_is_fresh_each_time(self):
        a, b = default_registry(), default_registry()
        assert a is not b and a.ids() == b.ids() and len(a) >= 60
        a.register(spec())
        assert "t.node" not in b and "t.node" not in default_registry()
        assert all(s.id.startswith("base.") for s in b)
