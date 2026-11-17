"""Structures in the graph model and the file format."""

import json

import pytest

import openramen as ramen
from openramen.model import (
    FRAME_IN, FRAME_OUT, INPUTS, OUTPUTS, UNSET, Graph, GraphError, Structure, Tunnel,
)


def sum_of_squares():
    g = Graph("sum of squares")
    g.add_node("base.constant", "numbers", config={"value": [1, 2, 3]})
    loop = g.add_for("loop")
    loop.add_input("x", mode="index")
    loop.add_register("total")
    loop.add_output("squares", mode="collect")
    frame = loop.frame
    frame.add_node("base.multiply", "square")
    frame.add_node("base.add", "add")
    frame.connect(INPUTS, "x", "square", "a")
    frame.connect(INPUTS, "x", "square", "b")
    frame.connect(INPUTS, "total", "add", "a")
    frame.connect("square", "out", "add", "b")
    frame.connect("add", "out", OUTPUTS, "total")
    frame.connect("square", "out", OUTPUTS, "squares")
    g.connect("numbers", "out", loop, "x")
    loop.values["total"] = 0
    return g


class TestBuilding:
    def test_a_structure_is_a_node_with_frames(self):
        g = Graph()
        loop = g.add_for()
        assert isinstance(loop, Structure) and loop.id == "for_1" and loop.spec == "structure.for"
        assert g.node("for_1") is loop and loop.kind == "for" and len(loop.frames) == 1
        frame = loop.frame
        assert [n.id for n in frame] == [INPUTS, OUTPUTS]
        assert frame.node(INPUTS).spec == FRAME_IN and frame.node(OUTPUTS).spec == FRAME_OUT
        assert g.add_while().id == "while_1" and g.add_subgraph("mine").kind == "subgraph"
        assert g.has_structures() and not Graph().has_structures()

    def test_tunnels_and_registers(self):
        loop = Graph().add_for("loop")
        tunnel = loop.add_input("xs", mode="index", type="List[Float]")
        assert tunnel == Tunnel("xs", "index", "List[Float]") and loop.input("xs") is tunnel
        out = loop.add_output("ys", mode="stack", fallback=0)
        assert out.fallback == 0 and loop.output("ys") is out and loop.output("zzz") is None
        assert loop.add_output("plain").fallback is UNSET
        assert loop.add_register("acc") == "acc" and loop.registers == ["acc"]
        assert repr(UNSET) == "UNSET"

    def test_case_frames(self):
        g = Graph()
        case = g.add_case("choose", cases=[[True], [False]])
        assert len(case.frames) == 2 and case.cases == [[True], [False]]
        other = g.add_case(cases=[[1, 2, (10, 20)], ["x"], None])
        assert other.cases == [[1, 2, (10, 20)], ["x"], None]
        extra = other.add_frame([99])
        assert other.frames[-1] is extra and len(other.frames) == 4

    @pytest.mark.parametrize("make, message", [
        (lambda g: g.add_structure("until"), "structure kind"),
        (lambda g: g.add_structure("case"), "'cases' is needed"),
        (lambda g: g.add_structure("for", cases=[[1]]), "'cases' is needed"),
        (lambda g: g.add_case(cases=[]), "at least one frame"),
        (lambda g: g.add_case(cases=[[]]), "non-empty list"),
        (lambda g: g.add_case(cases=[[1.5]]), "invalid case value"),
        (lambda g: g.add_case(cases=[[(5, 1)]]), "invalid case value"),
        (lambda g: g.add_case(cases=[[(1, 2, 3)]]), "invalid case value"),
        (lambda g: g.add_case(cases=["ab"]), "non-empty list"),
        (lambda g: g.add_for("a b"), "invalid node id"),
        (lambda g: g.add_for().add_input("i"), "uses itself"),
        (lambda g: g.add_for().add_input("count"), "uses itself"),
        (lambda g: g.add_while().add_output("stop"), "uses itself"),
        (lambda g: g.add_case(cases=[None]).add_input("selector"), "uses itself"),
        (lambda g: g.add_for().add_input("not valid"), "invalid pin name"),
        (lambda g: g.add_for().add_input("x", mode="zip"), "input mode"),
        (lambda g: g.add_while().add_input("x", mode="index"), "only a For loop"),
        (lambda g: g.add_subgraph().add_output("x", mode="collect"), "only loops"),
        (lambda g: g.add_case(cases=[None]).add_output("x", mode="stack"), "only loops"),
        (lambda g: g.add_for().add_output("x", mode="sum"), "output mode"),
        (lambda g: g.add_subgraph().add_register("r"), "only loops"),
        (lambda g: g.add_for().add_frame([1]), "only a case structure"),
        (lambda g: g.add_for().add_input("x", type="Nonsense"), "invalid tunnel type"),
    ])
    def test_invalid_structures(self, make, message):
        g = Graph()
        with pytest.raises(GraphError, match=message):
            make(g)

    def test_names_are_unique_per_side(self):
        loop = Graph().add_for()
        loop.add_input("a")
        loop.add_register("r")
        loop.add_output("a")            # an input and an output may share a name
        for add in (loop.add_input, loop.add_output, loop.add_register):
            with pytest.raises(GraphError, match="already has a pin"):
                add("r")
        with pytest.raises(GraphError, match="already has a pin"):
            loop.add_input("a")
        with pytest.raises(GraphError, match="already has a pin"):
            loop.add_register("a")

    def test_a_failed_structure_leaves_no_node_behind(self):
        g = Graph()
        with pytest.raises(GraphError):
            g.add_case("c", cases=[[1.5]])
        assert "c" not in g and len(g) == 0

    def test_wiring_and_removing(self):
        g = sum_of_squares()
        assert g.incoming("loop")["x"].src == "numbers"
        g.add_node("base.to_text", "show")
        g.connect("loop", "total", "show", "x")
        g.remove_node("loop")
        assert "loop" not in g and g.wires == []


class TestJson:
    def test_roundtrip(self):
        g = sum_of_squares()
        text = g.dumps()
        data = json.loads(text)
        assert data["version"] == 2
        loaded = Graph.loads(text)
        assert loaded == g and loaded.dumps() == text
        loop = loaded.node("loop")
        assert isinstance(loop, Structure) and loop.registers == ["total"]
        assert [(t.name, t.mode) for t in loop.inputs] == [("x", "index")]
        assert loop.values == {"total": 0} and len(loop.frame.wires) == 6

    def test_graphs_without_structures_stay_version_one(self):
        g = Graph()
        g.add_node("base.constant", config={"value": 1})
        assert json.loads(g.dumps())["version"] == 1

    def test_frames_are_written_one_node_per_line(self):
        text = sum_of_squares().dumps()
        lines = [line.strip() for line in text.splitlines()]
        assert '{"id": "square", "spec": "base.multiply"},' in lines or \
            '{"id": "square", "spec": "base.multiply"}' in lines
        assert '["inputs", "x", "square", "a"],' in lines
        assert '"registers": ["total"],' in lines

    def test_everything_survives(self):
        g = Graph("all kinds")
        loop = g.add_while("w", max_iterations=50)
        loop.add_input("limit", type="Int")
        loop.add_output("trace", mode="collect")
        loop.add_output("answer", fallback=(1, 2.5), type="Tuple[Int, Float]")
        loop.add_register("state")
        loop.frame.node(OUTPUTS).values["stop"] = True
        loop.frame.node("outputs").pos = (300.0, 20.0)
        case = loop.frame.add_case("choose", cases=[[0, (5, 9)], ["a"], [True], None])
        case.add_input("value")
        case.add_output("result", fallback=float("nan"))
        inner = case.frames[1].add_subgraph("deep")
        inner.add_output("x")
        inner.frame.node(OUTPUTS).values["x"] = b"\x00"
        text = g.dumps()
        loaded = Graph.loads(text)
        assert loaded.dumps() == text
        w = loaded.node("w")
        assert w.max_iterations == 50 and w.input("limit").type == "Int"
        assert w.output("answer").fallback == (1, 2.5)
        assert w.frame.node(OUTPUTS).values == {"stop": True}
        choose = w.frame.node("choose")
        assert choose.cases == [[0, (5, 9)], ["a"], [True], None]
        assert choose.frames[1].node("deep").frame.node(OUTPUTS).values == {"x": b"\x00"}
        assert json.loads(text)["version"] == 2

    def test_copy_is_deep(self):
        g = sum_of_squares()
        clone = g.copy()
        assert clone == g
        clone.node("loop").frame.remove_node("add")
        clone.node("loop").add_output("extra")
        clone.node("loop").registers.append("more")
        original = g.node("loop")
        assert "add" in original.frame and original.output("extra") is None
        assert original.registers == ["total"] and clone != g

    def test_equality_looks_inside_frames(self):
        a, b = sum_of_squares(), sum_of_squares()
        assert a == b
        b.node("loop").frame.node("add").values["b"] = 1
        assert a != b

    @pytest.mark.parametrize("structure, message", [
        ("nope", "malformed structure"),
        ({"kind": "until"}, "malformed structure"),
        ({"kind": "while", "frames": [{}]}, "does not match kind"),
        ({"kind": "for"}, "'frames' must be a non-empty list"),
        ({"kind": "for", "frames": []}, "'frames' must be a non-empty list"),
        ({"kind": "for", "frames": [{}, {}]}, "exactly one frame"),
        ({"kind": "for", "frames": [3]}, "'frames' must be a non-empty list"),
        ({"kind": "for", "frames": [{"nodes": [3]}]}, "malformed node"),
        ({"kind": "for", "frames": [{}], "inputs": [{"mode": "plain"}]}, "malformed tunnel"),
        ({"kind": "for", "frames": [{}], "inputs": [{"name": "x", "mode": "zip"}]}, "input mode"),
        ({"kind": "for", "frames": [{}], "inputs": [{"name": "x", "mode": "plain", "fallback": 1}]},
         "no fallback"),
        ({"kind": "for", "frames": [{}], "outputs": "x"}, "must be a list"),
        ({"kind": "for", "frames": [{}], "registers": ["i"]}, "uses itself"),
        ({"kind": "for", "frames": [{}], "registers": "r"}, "must be a list"),
        ({"kind": "for", "frames": [{}], "cases": [[1]]}, "only a case structure"),
        ({"kind": "for", "frames": [{}], "max_iterations": 0}, "positive integer"),
        ({"kind": "for", "frames": [{}], "outputs": [{"name": "x", "mode": "last",
                                                    "fallback": {"$nope": 1}}]}, "unknown literal"),
    ])
    def test_malformed_structures_are_refused(self, structure, message):
        data = {"format": "openramen.graph", "version": 2,
                "nodes": [{"id": "s", "spec": "structure.for", "structure": structure}]}
        with pytest.raises(GraphError, match=message):
            Graph.from_dict(data)

    def test_malformed_cases(self):
        def data(cases):
            return {"format": "openramen.graph", "version": 2, "nodes": [{
                "id": "c", "spec": "structure.case",
                "structure": {"kind": "case", "frames": [{}], "cases": cases}}]}
        assert Graph.from_dict(data([[1, {"range": [2, 4]}]])).node("c").cases == [[1, (2, 4)]]
        assert Graph.from_dict(data([None])).node("c").cases == [None]
        for bad in ([[1], [2]], "x", [[1.5]], [[{"range": [4, 2]}]], [[{"range": "ab"}]], [[]]):
            with pytest.raises(GraphError):
                Graph.from_dict(data(bad))

    def test_newer_files_are_refused(self):
        with pytest.raises(GraphError, match="newer"):
            Graph.from_dict({"format": "openramen.graph", "version": 3})

    def test_public_names(self):
        assert ramen.Structure is Structure and ramen.Tunnel is Tunnel
