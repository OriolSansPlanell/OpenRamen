"""Static analysis: what the checker reports and which types it resolves."""

import numpy as np
import pytest

import openramen as ramen
from openramen import Graph, analyze
from openramen import types as T
from openramen.library import NodeSpec, PinSpec, default_registry

from helpers import const, single

V = T.Verdict


def codes(analysis):
    return [p.code for p in analysis.problems]


def test_clean_graph():
    g = Graph()
    a, b = const(g, 2), const(g, 0.5)
    add = g.add_node("base.add")
    g.connect(a, "out", add, "a")
    g.connect(b, "out", add, "b")
    analysis = analyze(g)
    assert analysis.ok and analysis.problems == [] and analysis.warnings == []
    assert analysis.order == g.topological_order()
    assert analysis.out_type(a.id, "out") == T.INT
    assert analysis.out_type(add.id, "out") == T.FLOAT
    assert analysis.wire_verdict(add.id, "a") is V.EXACT
    assert analysis.wire_verdict(add.id, "zzz") is None
    assert analysis.wire_verdict("ghost", "a") is None
    assert analysis.out_type("ghost", "out") == T.ANY
    assert analysis.out_type(add.id, "nope") == T.ANY
    analysis.raise_for_errors()


def test_empty_graph_is_fine():
    analysis = analyze(Graph())
    assert analysis.ok and analysis.order == [] and analysis.nodes == {}


def test_unknown_spec_is_reported_and_downstream_still_checked():
    g = Graph()
    g.add_node("no.such.node", "ghost")
    g.add_node("base.negate", "neg")
    g.add_node("base.to_text", "text")
    g.connect("ghost", "out", "neg", "x")
    g.connect("neg", "out", "text", "x")
    analysis = analyze(g)
    assert codes(analysis) == ["unknown-spec"]
    assert analysis.problems[0].node == "ghost" and "no.such.node" in str(analysis.problems[0])
    assert analysis.wire_verdict("neg", "x") is V.DYNAMIC      # unknown source: decide later
    assert analysis.out_type("text", "out") == T.STR


@pytest.mark.parametrize("build, code, pin", [
    (lambda g: g.connect("c", "out", "n", "zzz"), "unknown-pin", "zzz"),
    (lambda g: g.connect("c", "zzz", "n", "a", replace=True), "unknown-pin", "zzz"),
    (lambda g: g.node("n").values.update(zzz=1), "unknown-pin", "zzz"),
    (lambda g: g.node("n").config.update(zzz=1), "unknown-config", "zzz"),
    (lambda g: g.node("c").config.pop("value"), "missing-config", "value"),
    (lambda g: g.disconnect("n", "b") or g.node("n").values.clear(), "missing-input", "b"),
    (lambda g: g.node("n").values.update(b="text"), "bad-literal", "b"),
    (lambda g: g.node("n").values.update(b=True), "bad-literal", "b"),
])
def test_each_kind_of_error(build, code, pin):
    g = Graph()
    const(g, 1, "c")
    g.add_node("base.add", "n", values={"b": 2})
    g.connect("c", "out", "n", "a")
    assert analyze(g).ok
    build(g)
    analysis = analyze(g)
    assert code in codes(analysis) and not analysis.ok
    problem = next(p for p in analysis.problems if p.code == code)
    assert problem.pin == pin and problem.severity == "error"
    with pytest.raises(ramen.GraphCheckError) as err:
        analysis.raise_for_errors()
    assert err.value.problems == analysis.errors and code not in str(err.value)
    assert pin in str(err.value) or problem.node in str(err.value)


def test_type_mismatch_names_the_wire_and_both_types():
    g = single("base.text_length", text=42)
    analysis = analyze(g)
    assert codes(analysis) == ["type-mismatch"]
    message = analysis.problems[0].message
    assert "in_text.out -> n.text" in message and "Int" in message and "Str" in message
    assert analysis.wire_verdict("n", "text") is V.INCOMPATIBLE


def test_bad_config_value():
    g = single("base.array_zeros", shape=[2], config={"dtype": 3})
    analysis = analyze(g)
    assert codes(analysis) == ["bad-config"] and analysis.problems[0].pin == "dtype"


def test_all_problems_are_collected_not_just_the_first():
    g = Graph()
    g.add_node("base.add", "n1")                       # two missing inputs
    g.add_node("base.concat", "n2", values={"a": 1, "b": 2})   # two bad literals
    g.add_node("nope", "n3")
    assert sorted(codes(analyze(g))) == [
        "bad-literal", "bad-literal", "missing-input", "missing-input", "unknown-spec"]


def test_literal_wins_only_when_there_is_no_wire():
    g = single("base.negate", x=5)
    g.node("n").values["x"] = "ignored while wired"
    assert analyze(g).ok
    g.disconnect("n", "x")
    assert codes(analyze(g)) == ["bad-literal"]


def test_literals_are_coerced_at_check_time():
    g = Graph()
    g.add_node("base.round", "n", values={"x": 2, "digits": 1})     # Int literal, Float pin
    info = analyze(g).nodes["n"]
    assert info.literals == {"x": 2.0, "digits": 1} and type(info.literals["x"]) is float
    g = Graph()
    g.add_node("base.join_text", "n", values={"items": ("a", "b")})  # tuple, List pin
    assert analyze(g).nodes["n"].literals == {"items": ["a", "b"]}
    g = Graph()
    g.add_node("base.add", "n", values={"a": np.float32(1.5), "b": np.int8(2)})
    info = analyze(g).nodes["n"]
    assert [type(v) for v in info.literals.values()] == [float, int]
    assert info.out_types["out"] == T.FLOAT


class TestWireVerdicts:
    def wire(self, source_value, spec_id, pin, **others):
        g = Graph()
        const(g, source_value, "src")
        g.add_node(spec_id, "n", values=others)
        g.connect("src", "out", "n", pin)
        return analyze(g)

    def test_exact(self):
        assert self.wire(1.5, "base.round", "x").wire_verdict("n", "x") is V.EXACT

    def test_coercion(self):
        analysis = self.wire(2, "base.round", "x")
        assert analysis.ok and analysis.wire_verdict("n", "x") is V.COERCION
        assert self.wire(("a", "b"), "base.join_text", "items").wire_verdict("n", "items") \
            is V.COERCION

    def test_dynamic(self):
        analysis = self.wire([1, 2.5], "base.list_get", "items", index=0)
        assert analysis.out_type("n", "out") == T.union(T.INT, T.FLOAT)
        g = Graph()
        const(g, [1, 2.5], "src")
        g.add_node("base.list_get", "pick", values={"index": 0})
        g.add_node("base.round", "n")
        g.connect("src", "out", "pick", "items")
        g.connect("pick", "out", "n", "x")
        analysis = analyze(g)
        assert analysis.ok and analysis.wire_verdict("n", "x") is V.DYNAMIC

    def test_incompatible(self):
        analysis = self.wire("text", "base.round", "x")
        assert not analysis.ok and analysis.wire_verdict("n", "x") is V.INCOMPATIBLE


class TestGenerics:
    def test_select_joins_its_branches(self):
        g = single("base.select", condition=True, if_true=1, if_false=2.5)
        analysis = analyze(g)
        assert analysis.out_type("n", "out") == T.FLOAT
        assert analysis.wire_verdict("n", "if_true") is V.COERCION
        assert analysis.wire_verdict("n", "if_false") is V.EXACT
        assert analysis.nodes["n"].in_types["if_true"] == T.FLOAT

    def test_conflicting_branches_are_an_error(self):
        analysis = analyze(single("base.select", condition=True, if_true=1, if_false="a"))
        assert "type-conflict" in codes(analysis)
        assert "~T" in next(p.message for p in analysis.problems if p.code == "type-conflict")

    def test_variable_flows_through_containers(self):
        g = Graph()
        const(g, [[1, 2], [3]], "src")
        g.add_node("base.list_get", "row", values={"index": 0})
        g.add_node("base.list_get", "cell", values={"index": 1})
        g.add_node("base.list_append", "more", values={"item": 9})
        g.connect("src", "out", "row", "items")
        g.connect("row", "out", "cell", "items")
        g.connect("row", "out", "more", "items")
        analysis = analyze(g)
        assert analysis.ok
        assert str(analysis.out_type("row", "out")) == "List[Int]"
        assert analysis.out_type("cell", "out") == T.INT
        assert str(analysis.out_type("more", "out")) == "List[Int]"

    def test_unknown_inputs_leave_the_variable_open(self):
        g = Graph()
        const(g, [], "src")
        g.add_node("base.list_get", "n", values={"index": 0})
        g.connect("src", "out", "n", "items")
        assert analyze(g).out_type("n", "out") == T.ANY

    def test_identity_chain_keeps_precise_array_types(self):
        g = Graph()
        const(g, np.zeros((2, 3), dtype="uint16"), "src")
        previous = "src"
        for i in range(5):
            node = g.add_node("base.identity", f"id{i}")
            g.connect(previous, "out", node, "value")
            previous = node.id
        assert str(analyze(g).out_type("id4", "out")) == "Array[uint16, 2]"


class TestInferHook:
    def registry(self, infer):
        registry = default_registry()
        registry.register(NodeSpec(
            "t.poly", inputs=(PinSpec("x", "Union[Float, Str]"),),
            outputs=(PinSpec("out", "Any"), PinSpec("other", "Bool")),
            config=(PinSpec("mode", "Str", default="plain"),),
            fn=lambda x, mode="plain": (x, True), infer=infer))
        return registry

    def test_hook_sees_delivered_types_and_config(self):
        seen = {}

        def infer(inputs, config):
            seen.update(inputs=dict(inputs), config=dict(config))
            return {"out": inputs["x"], "unknown_pin": T.STR, "other": "not a type"}

        registry = self.registry(infer)
        g = single("t.poly", registry, x=3, config={"mode": "fancy"})
        analysis = analyze(g, registry)
        assert analysis.ok
        # the Int wire is coerced to Float: that is what the function receives
        assert seen["inputs"] == {"x": T.FLOAT} and seen["config"] == {"mode": "fancy"}
        assert analysis.out_type("n", "out") == T.FLOAT
        assert analysis.out_type("n", "other") == T.BOOL     # junk from the hook is ignored
        assert ramen.run(g, registry, check_outputs=True)["n"] == {"out": 3.0, "other": True}

    def test_hook_gets_none_for_missing_inputs_and_defaults_for_config(self):
        seen = {}
        registry = self.registry(lambda i, c: seen.update(i=dict(i), c=dict(c)) or {})
        g = Graph()
        g.add_node("t.poly", "n")
        analysis = analyze(g, registry)
        assert codes(analysis) == ["missing-input"]
        assert seen == {"i": {"x": None}, "c": {"mode": "plain"}}

    def test_a_crashing_hook_is_only_a_warning(self):
        def infer(inputs, config):
            raise RuntimeError("hook bug")

        registry = self.registry(infer)
        g = single("t.poly", registry, x="text")
        analysis = analyze(g, registry)
        assert analysis.ok and codes(analysis) == ["infer-failed"]
        assert analysis.warnings[0].severity == "warning" and "hook bug" in str(analysis.warnings[0])
        assert analysis.out_type("n", "out") == T.ANY          # falls back to the declaration
        assert ramen.run(g, registry)["n"]["out"] == "text"


def test_problem_text():
    problem = ramen.Problem("error", "x", "something is wrong", "node_1", "pin")
    assert str(problem) == "error: something is wrong [node_1.pin]"
    assert str(ramen.Problem("warning", "x", "hm", "node_1")) == "warning: hm [node_1]"
    assert str(ramen.Problem("error", "x", "global")) == "error: global"


def test_analysis_does_not_modify_the_graph():
    g = single("base.round", x=2, literals={"digits": 1})
    before = g.dumps()
    analyze(g)
    assert g.dumps() == before


def test_default_registry_is_used_and_custom_registries_are_respected():
    g = single("base.negate", x=1)
    assert analyze(g).ok
    assert codes(analyze(g, ramen.Registry())) == ["unknown-spec", "unknown-spec"]


def test_a_literal_that_cannot_be_converted_is_reported_not_raised():
    g = Graph()
    g.add_node("base.round", "n", values={"x": 10**400})
    analysis = analyze(g)
    assert codes(analysis) == ["bad-literal"] and "cannot convert" in analysis.problems[0].message


def test_a_wired_value_that_cannot_be_converted_fails_on_its_pin():
    g = single("base.round", x=10**400)
    assert analyze(g).ok
    with pytest.raises(ramen.NodeError) as err:
        ramen.run(g)
    assert err.value.pin == "x" and isinstance(err.value.original, OverflowError)
