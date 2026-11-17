"""Static analysis of structures: tunnels, shift registers, cases."""

import enum

import numpy as np
import pytest

from openramen import Graph, GraphCheckError, analyze, run
from openramen import types as T
from openramen.check import concatenated_type, element_type, interface, stacked_type
from openramen.library import default_registry
from openramen.model import INPUTS, OUTPUTS

from helpers import const


class Colour(enum.Enum):
    RED = 1
    GREEN = 2


def codes(analysis):
    return sorted(p.code for p in analysis.problems)


def loop_over(value, mode="index", kind="for"):
    """A graph with a loop fed by one constant; returns (graph, loop, frame)."""
    g = Graph()
    const(g, value, "src")
    loop = g.add_structure(kind, "loop")
    loop.add_input("x", mode=mode)
    g.connect("src", "out", loop, "x")
    return g, loop, loop.frame


def out_types(g, node="loop"):
    return {k: str(v) for k, v in analyze(g).nodes[node].out_types.items()}


@pytest.mark.parametrize("text, expected", [
    ("List[Int]", "Int"), ("List", "Any"), ("Tuple[Int, ...]", "Int"),
    ("Tuple[Int, Str]", "Union[Int, Str]"), ("Tuple[]", "Any"), ("Iterator[Str]", "Str"),
    ("Array[float64, 1]", "Float"), ("Array[uint8, 1]", "Int"), ("Array[bool, 1]", "Bool"),
    ("Array[complex64, 1]", "Complex"), ("Array[float32, 3]", "Array[float32, 2]"),
    ("Array[_, 2]", "Array[_, 1]"), ("Array[_, 1]", "Any"),
    ("Array[float64, _]", "Union[Array[float64, _], Float]"), ("Array", "Any"),
    ("Any", "Any"), ("Union[List[Int], List[Str]]", "Union[Int, Str]"),
    ("Optional[List[Int]]", "Int"),
    ("Int", None), ("Str", None), ("Dict[Str, Int]", None), ("Array[float64, 0]", None),
    ("Union[Int, Str]", None),
])
def test_element_type(text, expected):
    result = element_type(T.parse_type(text))
    assert (None if result is None else str(result)) == expected


@pytest.mark.parametrize("text, expected", [
    ("Int", "Array[int64, 1]"), ("Float", "Array[float64, 1]"), ("Bool", "Array[bool, 1]"),
    ("Complex", "Array[complex128, 1]"), ("Array[uint8, 2]", "Array[uint8, 3]"),
    ("Array[_, 1]", "Array[_, 2]"), ("Array[float32, _]", "Array[float32, _]"),
    ("Str", "Array[_, _]"), ("Any", "Array[_, _]"), ("List[Int]", "Array[_, _]"),
])
def test_stacked_type(text, expected):
    assert str(stacked_type(T.parse_type(text))) == expected


@pytest.mark.parametrize("text, expected", [
    ("List[Int]", "List[Int]"), ("Array[uint8, 2]", "Array[uint8, 2]"), ("Any", "Any"),
    ("Union[List[Int], List[Str]]", "List[Union[Int, Str]]"),
    ("Union[List[Any], List[Int]]", "List[Any]"),
    ("Union[Array[uint8, 1], Array[float64, 1]]", "Array[_, 1]"),
    ("Union[Array[uint8, 1], Array[uint8, 2]]", "Array[uint8, _]"),
    ("Array[_, _]", "Array[_, _]"),
    ("Int", None), ("Array[float64, 0]", None), ("Union[List[Int], Array[uint8, 1]]", None),
    ("Optional[List[Int]]", None), ("Tuple[Int, ...]", None), ("Str", None),
])
def test_concatenated_type(text, expected):
    result = concatenated_type(T.parse_type(text))
    assert (None if result is None else str(result)) == expected


def test_interface():
    g = Graph()
    loop = g.add_for("loop")
    loop.add_input("xs", mode="index", type="List[Float]")
    loop.add_register("acc")
    loop.add_output("ys", mode="collect")
    spec = interface(loop)
    assert [(p.name, str(p.type), p.required) for p in spec.inputs] == [
        ("count", "Int", False), ("xs", "List[Float]", True), ("acc", "Any", True)]
    assert [p.name for p in spec.outputs] == ["ys", "acc"]
    case = g.add_case(cases=[None])
    assert [p.name for p in interface(case).inputs] == ["selector"]
    assert interface(g.add_while()).inputs == ()


class TestForLoop:
    def test_index_and_outputs(self):
        g, loop, frame = loop_over([1.5, 2.5])
        for mode in ("last", "collect", "stack"):
            loop.add_output(mode, mode=mode)
            frame.connect(INPUTS, "x", OUTPUTS, mode)
        loop.add_output("index")
        frame.connect(INPUTS, "i", OUTPUTS, "index")
        analysis = analyze(g)
        assert analysis.ok, analysis.problems
        assert out_types(g) == {"last": "Float", "collect": "List[Float]",
                                "stack": "Array[float64, 1]", "index": "Int"}
        assert analysis.nodes["loop"].frames[0].nodes[INPUTS].out_types["x"] == T.FLOAT

    def test_rows_of_an_image(self):
        g, loop, frame = loop_over(np.zeros((4, 5), dtype="uint16"))
        loop.add_output("rows", mode="stack")
        loop.add_output("flat", mode="concat")
        loop.add_output("sums", mode="collect")
        frame.add_node("base.array_sum", "sum")
        frame.connect(INPUTS, "x", "sum", "array")
        frame.connect(INPUTS, "x", OUTPUTS, "rows")
        frame.connect(INPUTS, "x", OUTPUTS, "flat")
        frame.connect("sum", "out", OUTPUTS, "sums")
        assert out_types(g) == {"rows": "Array[uint16, 2]", "flat": "Array[uint16, 1]",
                                "sums": "List[Int]"}

    def test_plain_input_keeps_its_type(self):
        g, loop, frame = loop_over([1, 2], mode="plain")
        loop.values["count"] = 3
        loop.add_output("same")
        frame.connect(INPUTS, "x", OUTPUTS, "same")
        assert analyze(g).ok and out_types(g) == {"same": "List[Int]"}

    def test_problems(self):
        g, loop, frame = loop_over(42)
        assert codes(analyze(g)) == ["bad-tunnel"]
        g, loop, frame = loop_over([1], mode="plain")
        assert codes(analyze(g)) == ["no-count"]
        loop.values["count"] = "three"
        assert codes(analyze(g)) == ["bad-literal"]
        loop.values["count"] = 3
        loop.add_output("y")
        assert codes(analyze(g)) == ["unwired-tunnel"]
        loop.add_output("z", mode="concat")
        frame.connect(INPUTS, "i", OUTPUTS, "y")
        frame.connect(INPUTS, "i", OUTPUTS, "z")
        analysis = analyze(g)
        assert codes(analysis) == ["bad-tunnel"] and "concatenates lists or arrays" in \
            analysis.problems[0].message

    def test_missing_tunnel_input(self):
        g = Graph()
        loop = g.add_for("loop")
        loop.add_input("x", mode="index")
        problems = analyze(g).problems
        assert [(p.code, p.node, p.pin) for p in problems] == [("missing-input", "loop", "x")]

    def test_declared_tunnel_types(self):
        g, loop, frame = loop_over([1, 2])
        loop.inputs[0].type = "List[Float]"          # Int list coerced on the way in
        loop.add_output("y", type="Float")
        frame.connect(INPUTS, "x", OUTPUTS, "y")
        analysis = analyze(g)
        assert analysis.ok and analysis.wire_verdict("loop", "x") is T.Verdict.COERCION
        assert out_types(g) == {"y": "Float"}
        loop.outputs[0].type = "Str"
        assert codes(analyze(g)) == ["type-mismatch"]
        loop.inputs[0].type = "List[Str]"
        assert "type-mismatch" in codes(analyze(g))

    def test_fallback_joins_into_the_output_type(self):
        g, loop, frame = loop_over([1, 2])
        loop.add_output("y", fallback=0.5)
        frame.connect(INPUTS, "x", OUTPUTS, "y")
        assert out_types(g) == {"y": "Float"}
        loop.outputs[0].fallback = None
        assert out_types(g) == {"y": "Optional[Int]"}
        loop.outputs[0].fallback = "none"
        assert codes(analyze(g)) == ["tunnel-conflict"]


class TestShiftRegisters:
    def accumulate(self, start, values, spec="base.add"):
        g, loop, frame = loop_over(values)
        loop.add_register("acc")
        loop.values["acc"] = start
        frame.add_node(spec, "op")
        frame.connect(INPUTS, "acc", "op", "a")
        frame.connect(INPUTS, "x", "op", "b")
        frame.connect("op", "out", OUTPUTS, "acc")
        return g

    def test_type_stays_when_the_loop_agrees(self):
        assert out_types(self.accumulate(0, [1, 2])) == {"acc": "Int"}
        assert out_types(self.accumulate(0.5, [1, 2])) == {"acc": "Float"}

    def test_type_widens_to_a_fixed_point(self):
        g = self.accumulate(0, [1.5, 2.5])           # starts Int, the body makes it Float
        analysis = analyze(g)
        assert analysis.ok and out_types(g) == {"acc": "Float"}
        inner = analysis.nodes["loop"].frames[0]
        assert inner.nodes[INPUTS].out_types["acc"] == T.FLOAT
        assert inner.nodes["op"].out_types["out"] == T.FLOAT
        assert out_types(self.accumulate(0, [1j])) == {"acc": "Complex"}

    def test_unrelated_types_are_a_conflict(self):
        g = self.accumulate("text", ["a", "b"], spec="base.text_length")
        g.node("loop").frame.disconnect("op", "a")
        g.node("loop").frame.disconnect("op", "b")
        g.node("loop").frame.connect(INPUTS, "x", "op", "text")
        analysis = analyze(g)
        assert codes(analysis) == ["register-conflict"]
        assert "holds Str but the loop feeds back Int" in analysis.problems[0].message

    def test_array_rank_must_not_change(self):
        g = Graph()
        const(g, np.zeros(3), "start")
        loop = g.add_for("loop")
        loop.values["count"] = 2
        loop.add_register("a")
        g.connect("start", "out", loop, "a")
        frame = loop.frame
        frame.add_node("base.array_get", "row", values={"index": 0})
        frame.connect(INPUTS, "a", "row", "array")
        frame.connect("row", "out", OUTPUTS, "a")
        assert codes(analyze(g)) == ["register-conflict"]

    def test_unwired_register(self):
        g, loop, frame = loop_over([1])
        loop.add_register("acc")
        assert codes(analyze(g)) == ["missing-input", "unwired-register"]

    def test_untyped_feedback_is_checked_at_run_time(self):
        registry = default_registry()
        registry.node("t.anything")(lambda x: x)
        g, loop, frame = loop_over([1, 2])
        loop.add_register("acc")
        loop.values["acc"] = 0
        frame.add_node("t.anything", "op")
        frame.connect(INPUTS, "x", "op", "x")
        frame.connect("op", "out", OUTPUTS, "acc")
        analysis = analyze(g, registry)
        assert analysis.ok and str(analysis.nodes["loop"].out_types["acc"]) == "Int"

    def test_two_registers_that_depend_on_each_other(self):
        # fibonacci-like swap: a, b = b, a + b with a Float sneaking in
        g = Graph()
        loop = g.add_for("loop")
        loop.values.update(count=5, a=0, b=1.0)
        loop.add_register("a")
        loop.add_register("b")
        frame = loop.frame
        frame.add_node("base.add", "sum")
        frame.connect(INPUTS, "a", "sum", "a")
        frame.connect(INPUTS, "b", "sum", "b")
        frame.connect(INPUTS, "b", OUTPUTS, "a")
        frame.connect("sum", "out", OUTPUTS, "b")
        assert analyze(g).ok and out_types(g) == {"a": "Float", "b": "Float"}


class TestWhileLoop:
    def make(self):
        g = Graph()
        loop = g.add_while("loop")
        loop.add_register("n")
        loop.values["n"] = 0
        frame = loop.frame
        frame.add_node("base.add", "next", values={"b": 1})
        frame.add_node("base.greater_equal", "done", values={"b": 3})
        frame.connect(INPUTS, "n", "next", "a")
        frame.connect("next", "out", "done", "a")
        frame.connect("next", "out", OUTPUTS, "n")
        frame.connect("done", "out", OUTPUTS, "stop")
        return g, loop, frame

    def test_ok(self):
        g, loop, frame = self.make()
        assert analyze(g).ok and out_types(g) == {"n": "Int"}

    def test_stop_must_be_wired_and_boolean(self):
        g, loop, frame = self.make()
        frame.disconnect(OUTPUTS, "stop")
        assert codes(analyze(g)) == ["unwired-stop"]
        frame.connect("next", "out", OUTPUTS, "stop")
        analysis = analyze(g)
        assert codes(analysis) == ["type-mismatch"] and "'stop' must be Bool" in \
            analysis.problems[0].message
        frame.disconnect(OUTPUTS, "stop")
        frame.node(OUTPUTS).values["stop"] = True      # a constant is fine
        assert analyze(g).ok

    def test_no_indexing_and_no_count(self):
        g, loop, frame = self.make()
        loop.values["count"] = 3
        assert codes(analyze(g)) == ["unknown-pin"]


class TestCase:
    def make(self, selector, cases):
        g = Graph()
        const(g, selector, "sel")
        case = g.add_case("case", cases=cases)
        case.add_output("y")
        g.connect("sel", "out", case, "selector")
        return g, case

    def test_bool(self):
        g, case = self.make(True, [[True], [False]])
        case.frames[0].node(OUTPUTS).values["y"] = 1
        case.frames[1].node(OUTPUTS).values["y"] = 2.5
        analysis = analyze(g)
        assert analysis.ok and out_types(g, "case") == {"y": "Float"}     # Int joins Float
        assert analysis.nodes["case"].frames[0].nodes[INPUTS].out_types["selector"] == T.BOOL

    @pytest.mark.parametrize("selector, cases, expected", [
        (True, [[True]], ["no-default-case"]),
        (True, [[True], None], []),
        (True, [[True], [True]], ["duplicate-case", "no-default-case"]),
        (True, [[1], None], ["bad-case"]),
        (3, [[1, 2], [(3, 9)]], ["no-default-case"]),
        (3, [[1, 2], [(3, 9)], None], []),
        (3, [["a"], None], ["bad-case"]),
        (3, [[True], None], ["bad-case"]),
        (3, [None, None], ["duplicate-case"]),
        ("a", [["a", "b"], None], []),
        ("a", [["a"]], ["no-default-case"]),
        ("a", [[1], None], ["bad-case"]),
        (Colour.RED, [["RED"], ["GREEN"]], []),
        (Colour.RED, [["RED"]], ["no-default-case"]),
        (Colour.RED, [["RED"], None], []),
        (Colour.RED, [["BLUE"], None], ["bad-case"]),
        (2.5, [None], ["bad-selector"]),
        ([1], [None], ["bad-selector"]),
    ])
    def test_selector_and_coverage(self, selector, cases, expected):
        g, case = self.make(selector, cases)
        for frame in case.frames:
            frame.node(OUTPUTS).values["y"] = 0
        assert codes(analyze(g)) == expected

    def test_dynamic_selector_accepts_any_cases_but_needs_a_default(self):
        registry = default_registry()
        registry.node("t.anything")(lambda: 3)
        g = Graph()
        g.add_node("t.anything", "sel")
        case = g.add_case("case", cases=[[1], ["a"], [True], None])
        g.connect("sel", "out", case, "selector")
        assert analyze(g, registry).ok
        case.cases[-1] = [2]
        assert codes(analyze(g, registry)) == ["no-default-case"]

    def test_outputs_must_be_wired_in_every_frame_or_have_a_fallback(self):
        g, case = self.make(True, [[True], [False]])
        case.frames[0].node(OUTPUTS).values["y"] = 1
        analysis = analyze(g)
        assert codes(analysis) == ["unwired-tunnel"] and "frame 1" in analysis.problems[0].message
        case.outputs[0].fallback = 0.0
        assert analyze(g).ok and out_types(g, "case") == {"y": "Float"}

    def test_unrelated_frame_types_conflict(self):
        g, case = self.make(True, [[True], [False]])
        case.frames[0].node(OUTPUTS).values["y"] = 1
        case.frames[1].node(OUTPUTS).values["y"] = "one"
        analysis = analyze(g)
        assert codes(analysis) == ["tunnel-conflict"]
        assert "Int, Str" in analysis.problems[0].message

    def test_inputs_reach_every_frame(self):
        g, case = self.make(True, [[True], [False]])
        const(g, [1, 2], "data")
        case.add_input("xs")
        g.connect("data", "out", case, "xs")
        for frame in case.frames:
            frame.add_node("base.list_length", "n")
            frame.connect(INPUTS, "xs", "n", "items")
            frame.connect("n", "out", OUTPUTS, "y")
        assert analyze(g).ok and out_types(g, "case") == {"y": "Int"}


class TestNesting:
    def test_problems_carry_the_path_of_the_node(self):
        g = Graph()
        loop = g.add_for("outer")
        loop.values["count"] = 2
        case = loop.frame.add_case("pick", cases=[[True], None])
        case.values["selector"] = True
        case.frames[1].add_node("base.add", "broken")
        inner = case.frames[0].add_subgraph("deep")
        inner.frame.add_node("no.such.node", "ghost")
        problems = {(p.code, p.node) for p in analyze(g).problems}
        assert problems == {
            ("missing-input", "outer/pick[1]/broken"),
            ("unknown-spec", "outer/pick[0]/deep/ghost"),
        }

    def test_frame_pins_are_checked(self):
        g, loop, frame = loop_over([1])
        frame.add_node("base.negate", "n")
        frame.connect(INPUTS, "nope", "n", "x")
        frame.connect("n", "out", OUTPUTS, "unknown_output")
        problems = {(p.code, p.node, p.pin) for p in analyze(g).problems}
        assert problems == {("unknown-pin", "loop/inputs", "nope"),
                            ("unknown-pin", "loop/outputs", "unknown_output")}

    def test_frame_without_boundary_nodes(self):
        g, loop, frame = loop_over([1])
        frame.remove_node(OUTPUTS)
        analysis = analyze(g)
        assert "bad-frame" in codes(analysis)
        g2 = Graph()
        g2.add_node("frame.inputs", "inputs")        # boundary nodes mean nothing at top level
        assert codes(analyze(g2)) == ["unknown-spec"]

    def test_subgraph_types_flow_through(self):
        g = Graph()
        const(g, np.zeros((2, 2), dtype="float32"), "image")
        sub = g.add_subgraph("double")
        sub.add_input("a")
        sub.add_output("b")
        sub.frame.add_node("base.add", "sum")
        sub.frame.connect(INPUTS, "a", "sum", "a")
        sub.frame.connect(INPUTS, "a", "sum", "b")
        sub.frame.connect("sum", "out", OUTPUTS, "b")
        g.connect("image", "out", sub, "a")
        assert analyze(g).ok and out_types(g, "double") == {"b": "Array[float32, 2]"}

    def test_the_count_is_not_visible_inside_the_loop(self):
        g, loop, frame = loop_over([1, 2])
        loop.add_output("y")
        frame.connect(INPUTS, "count", OUTPUTS, "y")
        problems = [(p.code, p.node, p.pin) for p in analyze(g).problems]
        assert problems == [("unknown-pin", "loop/inputs", "count")]

    @pytest.mark.parametrize("spoil, message", [
        (lambda s: s.frames.clear(), "has no frame"),
        (lambda s: s.frames.append(s.frames[0].copy()), "exactly one frame, not 2"),
        (lambda s: setattr(s, "kind", "until"), "unknown structure kind 'until'"),
        (lambda s: setattr(s, "kind", "subgraph"), "only a For loop can index its inputs"),
        (lambda s: setattr(s, "kind", "case"), "1 frames but 0 case entries"),
        (lambda s: s.cases.append([1]), "only a case structure has cases"),
        (lambda s: setattr(s.inputs[0], "mode", "zip"), "unknown mode 'zip'"),
        (lambda s: setattr(s.inputs[0], "type", "Nope["), "input 'x'"),
        (lambda s: s.registers.append("x"), "more than one input pin is called 'x'"),
        (lambda s: s.registers.append("i"), "'i' is a pin the structure uses itself"),
        (lambda s: s.outputs.append(s.outputs[0]), "more than one output pin is called 'y'"),
    ])
    def test_malformed_structures_are_reported(self, spoil, message):
        g, loop, frame = loop_over([1, 2])
        loop.add_output("y", mode="collect")
        frame.connect(INPUTS, "x", OUTPUTS, "y")
        g.add_node("base.list_length", "after")
        g.connect(loop, "y", "after", "items")
        assert analyze(g).ok and loop.flaws() == []
        spoil(loop)
        analysis = analyze(g)                       # reported, never raised
        flaws = [p for p in analysis.problems if p.code == "bad-frame"]
        assert flaws and all(p.node == "loop" for p in flaws)
        assert any(message in p.message for p in flaws), [p.message for p in flaws]
        assert not analysis.ok
        with pytest.raises(GraphCheckError):
            run(g)

    def test_kinds_keep_their_limits(self):
        g = Graph()
        sub = g.add_subgraph("s")
        sub.registers.append("r")
        sub.add_output("y")
        sub.outputs[0].mode = "stack"
        messages = [p.message for p in analyze(g).problems if p.code == "bad-frame"]
        assert messages == ["only loops have shift registers",
                            "output 'y': only loops can stack their outputs"]

    def test_frames_cannot_import_modules(self):
        g, loop, frame = loop_over([1])
        frame.imports = ["numpy"]
        assert "bad-frame" in codes(analyze(g))
