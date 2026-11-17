"""Functions with *args, **kwargs and keyword-only parameters as nodes."""

import pickle

import pytest

import openramen as ramen
from openramen import Graph, analyze, run
from openramen import types as T
from openramen.library import (
    DEFAULT, MISSING, NodeSpec, PinSpec, SpecError, default_registry, spec_from_function,
)

from helpers import const


def total(first: float, *rest: float, scale: float = 1.0) -> float:
    """Sum of all arguments, times a scale."""
    return (first + sum(rest)) * scale


def describe(title: str, sep: str = ", ", *items: int, upper: bool = False, **fields: int) -> str:
    parts = [str(i) for i in items] + [f"{k}={v}" for k, v in sorted(fields.items())]
    text = f"{title}: {sep.join(parts)}"
    return text.upper() if upper else text


def registry():
    r = default_registry()
    r.register(spec_from_function(total, "t.total"))
    r.register(spec_from_function(describe, "t.describe"))
    r.register(spec_from_function(lambda *values: list(values), "t.pack",
                                  types={"values": "~T", "out": "List[~T]"}))
    r.register(spec_from_function(lambda **named: dict(named), "t.named",
                                  types={"named": "~T", "out": "Dict[Str, ~T]"}))
    return r


def test_default_sentinel():
    assert repr(DEFAULT) == "DEFAULT" and DEFAULT is not MISSING
    assert pickle.loads(pickle.dumps(DEFAULT)) is DEFAULT and type(DEFAULT)() is DEFAULT
    pin = PinSpec("x", default=DEFAULT)
    assert not pin.required and not pin.variadic


class TestSpec:
    def test_kinds_from_a_signature(self):
        s = spec_from_function(describe, "t.d")
        assert [(p.name, p.kind) for p in s.inputs] == [
            ("title", "keyword"), ("sep", "keyword"), ("items", "var_positional"),
            ("upper", "keyword_only"), ("fields", "var_keyword")]
        assert s.var_positional.name == "items" and s.var_keyword.name == "fields"
        assert [p.required for p in s.inputs] == [True, False, False, False, False]
        assert s.var_positional.variadic and str(s.var_positional.type) == "Int"
        assert spec_from_function(total, "t.t").var_keyword is None

    def test_pin_for(self):
        s = spec_from_function(describe, "t.d")
        assert s.pin_for("title").name == "title"
        assert s.pin_for("items_0") is s.var_positional
        assert s.pin_for("items_12") is s.var_positional
        assert s.pin_for("anything") is s.var_keyword
        assert s.pin_for("items_x") is s.var_keyword          # not a number: a keyword
        assert s.pin_for("items") is None and s.pin_for("fields") is None
        assert s.pin_for("not valid") is None and s.pin_for("class") is None
        assert s.variadic_index("items_3") == 3 and s.variadic_index("items_03") is None
        assert s.variadic_index("items_") is None and s.variadic_index("other_1") is None
        t = spec_from_function(total, "t.t")
        assert t.pin_for("rest_0") is t.var_positional and t.pin_for("extra") is None

    def test_config_must_stay_passable_by_keyword(self):
        def f(a, mode="x", *rest, flag=False):
            return a
        assert [p.name for p in spec_from_function(f, "t.f", config=("flag",)).config] == ["flag"]
        with pytest.raises(SpecError, match="passed by keyword"):
            spec_from_function(f, "t.f", config=("mode",))
        with pytest.raises(SpecError, match="passed by keyword"):
            spec_from_function(f, "t.f", config=("rest",))


class TestCall:
    def test_without_variadic_values_keywords_are_used(self):
        s = spec_from_function(describe, "t.d")
        assert s.call({"title": "t"}) == {"out": "t: "}
        assert s.call({"title": "t", "upper": True}) == {"out": "T: "}

    def test_var_positional(self):
        s = spec_from_function(total, "t.t")
        assert s.call({"first": 1.0}) == {"out": 1.0}
        assert s.call({"first": 1.0, "rest_0": 2.0, "rest_1": 3.0}) == {"out": 6.0}
        assert s.call({"first": 1.0, "rest_0": 2.0, "scale": 10.0}) == {"out": 30.0}
        # the order of the dictionary does not matter, the numbers do
        pack = spec_from_function(lambda *v: list(v), "t.p")
        assert pack.call({"v_2": "c", "v_0": "a", "v_1": "b"}) == {"out": ["a", "b", "c"]}
        assert pack.call({}) == {"out": []}

    def test_parameters_before_args_are_filled_with_their_defaults(self):
        s = spec_from_function(describe, "t.d")
        assert s.call({"title": "t", "items_0": 1, "items_1": 2}) == {"out": "t: 1, 2"}
        assert s.call({"title": "t", "sep": "-", "items_0": 1, "items_1": 2}) == {"out": "t: 1-2"}

    def test_var_keyword(self):
        s = spec_from_function(describe, "t.d")
        assert s.call({"title": "t", "b": 2, "a": 1}) == {"out": "t: a=1, b=2"}
        assert s.call({"title": "t", "items_0": 7, "z": 0, "upper": True}) == {"out": "T: 7, Z=0"}

    @pytest.mark.parametrize("inputs, message", [
        ({"first": 1.0, "rest_1": 2.0}, "without gaps"),
        ({"first": 1.0, "rest_0": 2.0, "rest_2": 2.0}, "without gaps"),
        ({"first": 1.0, "nope": 2.0}, "no input named 'nope'"),
        ({"rest_0": 2.0}, "'first' needs a value"),
    ])
    def test_errors(self, inputs, message):
        with pytest.raises(TypeError, match=message):
            spec_from_function(total, "t.t").call(inputs)

    def test_unknown_default_blocks_variadics_only(self):
        s = NodeSpec("t.u", inputs=(PinSpec("a"), PinSpec("b", default=DEFAULT),
                                    PinSpec("rest", kind="var_positional")),
                     outputs=(PinSpec("out"),), fn=lambda a, b="B", *rest: (a, b, rest))
        assert s.call({"a": 1}) == {"out": (1, "B", ())}
        assert s.call({"a": 1, "b": 2, "rest_0": 3}) == {"out": (1, 2, (3,))}
        with pytest.raises(TypeError, match="'b' needs a value"):
            s.call({"a": 1, "rest_0": 3})


class TestInGraphs:
    def test_growable_pins(self):
        r = registry()
        g = Graph()
        for i, value in enumerate([1.5, 2, 3]):
            const(g, value, f"c{i}")
        g.add_node("t.total", "n", values={"scale": 2.0})
        g.connect("c0", "out", "n", "first")
        g.connect("c1", "out", "n", "rest_0")
        g.connect("c2", "out", "n", "rest_1")
        analysis = analyze(g, r)
        assert analysis.ok, analysis.problems
        assert analysis.nodes["n"].in_types == {
            "first": T.FLOAT, "scale": T.FLOAT, "rest_0": T.FLOAT, "rest_1": T.FLOAT}
        assert analysis.wire_verdict("n", "rest_0") is T.Verdict.COERCION   # Int into Float
        assert run(g, r)["n"] == 13.0
        assert run(Graph.loads(g.dumps()), r)["n"] == 13.0

    def test_literals_on_grown_pins(self):
        r = registry()
        g = Graph()
        g.add_node("t.total", "n", values={"first": 1, "rest_0": 2, "rest_1": 3.5})
        analysis = analyze(g, r)
        assert analysis.ok and analysis.nodes["n"].literals == {
            "first": 1.0, "rest_0": 2.0, "rest_1": 3.5}
        assert run(g, r)["n"] == 6.5

    def test_gap_and_type_errors_are_reported(self):
        r = registry()
        g = Graph()
        g.add_node("t.total", "n", values={"first": 1.0, "rest_1": 2.0})
        assert [p.code for p in analyze(g, r).problems] == ["variadic-gap"]
        g.node("n").values = {"first": 1.0, "rest_0": "text"}
        assert [p.code for p in analyze(g, r).problems] == ["bad-literal"]
        g.node("n").values = {"first": 1.0, "extra": 1.0}       # no **kwargs on this one
        assert [p.code for p in analyze(g, r).problems] == ["unknown-pin"]
        g.node("n").values = {"first": 1.0, "rest": [1.0]}      # the bare name is not a pin
        assert [p.code for p in analyze(g, r).problems] == ["unknown-pin"]

    def test_keyword_extras(self):
        r = registry()
        g = Graph()
        const(g, 5, "c")
        g.add_node("t.describe", "n", values={"title": "t", "height": 3, "items_0": 9})
        g.connect("c", "out", "n", "width")
        analysis = analyze(g, r)
        assert analysis.ok and analysis.nodes["n"].in_types["width"] == T.INT
        assert run(g, r)["n"] == "t: 9, height=3, width=5"
        g.node("n").values["depth"] = "deep"                    # **fields are ints
        assert [p.code for p in analyze(g, r).problems] == ["bad-literal"]

    def test_generic_variadics(self):
        r = registry()
        g = Graph()
        g.add_node("t.pack", "n", values={"values_0": 1, "values_1": 2.5, "values_2": 3})
        analysis = analyze(g, r)
        assert str(analysis.out_type("n", "out")) == "List[Float]"
        assert run(g, r, check_outputs=True)["n"] == [1.0, 2.5, 3.0]
        g = Graph()
        g.add_node("t.named", "n", values={"a": "x", "b": "y"})
        assert str(analyze(g, r).out_type("n", "out")) == "Dict[Str, Str]"
        assert run(g, r, check_outputs=True)["n"] == {"a": "x", "b": "y"}
        g.node("n").values["c"] = 3
        assert "type-conflict" in [p.code for p in analyze(g, r).problems]

    def test_no_variadic_inputs_at_all(self):
        r = registry()
        g = Graph()
        g.add_node("t.pack", "n")
        assert analyze(g, r).ok and run(g, r)["n"] == []
