"""Headless execution."""

import random

import numpy as np
import pytest

import openramen as ramen
from openramen import Graph, NodeError, compile_graph, run
from openramen import types as T
from openramen.library import NodeSpec, PinSpec, default_registry

from helpers import const, single


def expression():
    """(a + b) * c with a=2, b=0.5, c=4."""
    g = Graph("expr")
    const(g, 2, "a")
    const(g, 0.5, "b")
    const(g, 4, "c")
    g.add_node("base.add", "sum")
    g.add_node("base.multiply", "product")
    g.connect("a", "out", "sum", "a")
    g.connect("b", "out", "sum", "b")
    g.connect("sum", "out", "product", "a")
    g.connect("c", "out", "product", "b")
    return g


class TestResults:
    def test_values(self):
        result = run(expression())
        assert result["product"] == 10.0 and result["sum"] == 2.5 and result["a"] == 2
        assert result["product", "out"] == 10.0
        assert result.outputs("sum") == {"out": 2.5}
        assert "sum" in result and ("sum", "out") in result
        assert "ghost" not in result and ("sum", "zzz") not in result
        assert set(result.durations) == {"a", "b", "c", "sum", "product"}
        assert all(d >= 0 for d in result.durations.values())
        assert "5 nodes" in repr(result)
        with pytest.raises(KeyError):
            result["ghost"]
        with pytest.raises(KeyError):
            result["sum", "zzz"]

    def test_nodes_with_several_or_no_outputs(self):
        registry = default_registry()
        registry.node("t.divmod", outputs=("q", "r"))(lambda a, b: divmod(a, b))
        registry.node("t.sink", outputs=())(lambda value: None)
        g = single("t.divmod", registry, a=7, b=2)
        g.add_node("t.sink", "sink")
        g.connect("n", "r", "sink", "value")
        result = run(g, registry)
        assert result["n"] == {"q": 3, "r": 1} and result["n", "q"] == 3
        assert result["sink"] == {} and result.outputs("sink") == {}

    def test_empty_graph(self):
        result = run(Graph())
        assert result.values == {} and result.durations == {}

    def test_unwired_inputs_use_literals_then_defaults(self):
        g = Graph()
        g.add_node("base.round", "defaulted", values={"x": 2.567})
        g.add_node("base.round", "literal", values={"x": 2.567, "digits": 2})
        result = run(g)
        assert result["defaulted"] == 3.0 and result["literal"] == 2.57

    def test_running_twice_gives_the_same_result_and_leaves_the_graph_alone(self):
        g = expression()
        before = g.dumps()
        first, second = run(g), run(g)
        assert first.values == second.values and g.dumps() == before

    def test_saved_and_reloaded_graph_gives_the_same_result(self, tmp_path):
        g = expression()
        g.save(tmp_path / "g.json")
        reloaded = Graph.load(tmp_path / "g.json")
        assert run(reloaded).values == run(g).values

    def test_a_plan_can_be_reused(self):
        plan = compile_graph(expression())
        assert [s.node for s in plan.steps] == ["a", "b", "c", "sum", "product"]
        assert plan.name == "expr" and plan.analysis.ok
        assert run(plan)["product"] == run(plan)["product"] == 10.0

    def test_graph_with_errors_does_not_start(self):
        calls = []
        registry = default_registry()
        registry.node("t.spy")(lambda: calls.append(1))
        g = Graph()
        g.add_node("t.spy", "spy")
        g.add_node("base.add", "broken")
        with pytest.raises(ramen.GraphCheckError, match="2 errors"):
            run(g, registry)
        assert calls == []


class TestWires:
    def test_coercion_wire_converts(self):
        registry = default_registry()
        g = Graph()
        const(g, 3, "c")
        g.add_node("base.round", "r")
        g.connect("c", "out", "r", "x")
        assert run(g)["r"] == 3.0 and type(run(g)["r"]) is float

        registry.register(NodeSpec("t.floats", inputs=(PinSpec("xs", "List[Float]"),),
                                   outputs=(PinSpec("out"),), fn=lambda xs: xs))
        out = run(single("t.floats", registry, xs=[1, 2]), registry)["n"]
        assert out == [1.0, 2.0] and all(type(v) is float for v in out)

    def test_dynamic_wire_accepts_or_fails_on_the_pin(self):
        def graph(items):
            g = Graph()
            const(g, items, "src")
            g.add_node("base.list_get", "pick", values={"index": 0})
            g.add_node("base.round", "n")
            g.connect("src", "out", "pick", "items")
            g.connect("pick", "out", "n", "x")
            return g

        assert ramen.analyze(graph([1, 2.5])).wire_verdict("n", "x") is T.Verdict.DYNAMIC
        assert run(graph([2.4, 1]))["n"] == 2.0
        assert run(graph([7, 2.5]))["n"] == 7.0            # an Int arrives: coerced on the fly
        with pytest.raises(NodeError) as err:
            run(graph(["text", 2.5]))
        assert err.value.node == "n" and err.value.pin == "x"
        assert isinstance(err.value.original, T.TypeMismatch)
        assert "expected Float, got Str" in str(err.value)

    def test_fan_out(self):
        g = Graph()
        const(g, 3, "src")
        for i in range(5):
            g.add_node("base.multiply", f"m{i}", values={"b": i})
            g.connect("src", "out", f"m{i}", "a")
        result = run(g)
        assert [result[f"m{i}"] for i in range(5)] == [0, 3, 6, 9, 12]

    def test_literals_are_private_copies(self):
        registry = default_registry()

        @registry.node("t.grow")
        def grow(items: list, extra: list) -> int:
            items.append(1)
            extra.append(1)
            return len(items) + len(extra)

        g = Graph()
        g.add_node("t.grow", "n", values={"items": [], "extra": [0]})
        assert [run(g, registry)["n"] for _ in range(3)] == [3, 3, 3]
        assert g.node("n").values == {"items": [], "extra": [0]}

    def test_config_values_are_private_copies(self):
        registry = default_registry()
        registry.node("t.cfg", config=("table",))(lambda table: table.setdefault("n", 0))
        g = Graph()
        g.add_node("t.cfg", "n", config={"table": {}})
        run(g, registry), run(g, registry)
        assert g.node("n").config == {"table": {}}


class TestErrors:
    def failing(self):
        g = expression()
        g.add_node("base.divide", "bad", values={"b": 0})
        g.add_node("base.negate", "after")
        g.connect("product", "out", "bad", "a")
        g.connect("bad", "out", "after", "x")
        return g

    def test_node_error_names_the_node_and_keeps_the_cause(self):
        with pytest.raises(NodeError) as err:
            run(self.failing())
        e = err.value
        assert (e.node, e.spec, e.pin) == ("bad", "base.divide", None)
        assert isinstance(e.original, ZeroDivisionError) and e.__cause__ is e.original
        assert "node 'bad' (base.divide): ZeroDivisionError" in str(e)

    def test_nothing_downstream_runs_after_a_failure(self):
        events = []
        with pytest.raises(NodeError):
            run(self.failing(), listener=events.append)
        kinds = [(e.kind, e.node) for e in events]
        assert ("node-failed", "bad") in kinds and ("node-started", "after") not in kinds
        assert kinds[-1][0] == "graph-failed" and isinstance(events[-1].error, NodeError)

    def test_a_node_returning_the_wrong_number_of_values(self):
        registry = default_registry()
        registry.node("t.two", outputs=("a", "b"))(lambda: (1, 2, 3))
        g = Graph()
        g.add_node("t.two", "n")
        with pytest.raises(NodeError, match="3 values for 2 outputs"):
            run(g, registry)

    def test_keyboard_interrupt_is_not_swallowed(self):
        registry = default_registry()

        def interrupt():
            raise KeyboardInterrupt

        registry.node("t.stop")(interrupt)
        g = Graph()
        g.add_node("t.stop", "n")
        events = []
        with pytest.raises(KeyboardInterrupt):
            run(g, registry, listener=events.append)
        assert events[-1].kind == "graph-failed"

    def test_a_node_that_fails_inside_a_coercion(self):
        registry = default_registry()
        registry.register(NodeSpec("t.path", inputs=(PinSpec("p", "Path"),),
                                   outputs=(PinSpec("out", "Str"),), fn=lambda p: p.name))
        assert run(single("t.path", registry, p="a/b.txt"), registry)["n"] == "b.txt"


class TestOutputChecks:
    def registry(self, tier):
        registry = default_registry()
        registry.register(NodeSpec("t.liar", outputs=(PinSpec("out", "Int"),),
                                   fn=lambda: "not an int", tier=tier))
        return registry

    def graph(self):
        g = Graph()
        g.add_node("t.liar", "n")
        return g

    @pytest.mark.parametrize("tier", ["auto", "annotated"])
    def test_uncurated_nodes_are_checked_by_default(self, tier):
        with pytest.raises(NodeError) as err:
            run(self.graph(), self.registry(tier))
        assert err.value.pin == "out" and "declared Int but the node returned Str" in str(err.value)
        assert run(self.graph(), self.registry(tier), check_outputs=False)["n"] == "not an int"

    def test_curated_nodes_are_trusted_unless_asked(self):
        registry = self.registry("curated")
        assert run(self.graph(), registry)["n"] == "not an int"
        with pytest.raises(NodeError):
            run(self.graph(), registry, check_outputs=True)

    def test_numpy_scalars_become_python_scalars_on_the_wire(self):
        registry = default_registry()
        registry.register(NodeSpec("t.np", outputs=(PinSpec("out", "Float"),),
                                   fn=lambda: np.float32(1.5), tier="auto"))
        g = Graph()
        g.add_node("t.np", "n")
        value = run(g, registry)["n"]
        assert value == 1.5 and type(value) is float


class TestTargets:
    def test_only_what_is_needed_runs(self):
        events = []
        result = run(expression(), targets=["sum"], listener=events.append)
        ran = [e.node for e in events if e.kind == "node-finished"]
        assert ran == ["a", "b", "sum"]
        assert result["sum"] == 2.5 and "product" not in result and "c" not in result

    def test_several_targets_and_everything(self):
        result = run(expression(), targets=["sum", "c"])
        assert set(result.durations) == {"a", "b", "c", "sum"}
        assert set(run(expression(), targets=["product"]).durations) == \
            {"a", "b", "c", "sum", "product"}
        assert run(expression(), targets=[]).values == {}

    def test_unknown_target(self):
        with pytest.raises(KeyError, match="ghost"):
            run(expression(), targets=["ghost"])

    def test_targets_skip_failing_branches(self):
        g = expression()
        g.add_node("base.fail", "boom")
        with pytest.raises(NodeError):
            run(g)
        assert run(g, targets=["product"])["product"] == 10.0


class TestEvents:
    def test_sequence(self):
        events = []
        run(expression(), listener=events.append)
        kinds = [e.kind for e in events]
        assert kinds[0] == "graph-started" and kinds[-1] == "graph-finished"
        assert kinds.count("node-started") == kinds.count("node-finished") == 5
        started = [e.node for e in events if e.kind == "node-started"]
        finished = [e.node for e in events if e.kind == "node-finished"]
        assert started == finished == ["a", "b", "c", "sum", "product"]
        assert all(e.duration is not None and e.duration >= 0
                   for e in events if e.kind in ("node-finished", "graph-finished"))
        # a node starts only after everything it reads from has finished
        for i, event in enumerate(events):
            if event.kind == "node-started" and event.node == "product":
                done = {e.node for e in events[:i] if e.kind == "node-finished"}
                assert {"sum", "c"} <= done


class TestStructure:
    @pytest.mark.parametrize("seed", range(60))
    def test_random_expression_graphs_match_plain_python(self, seed):
        """Build a random DAG of arithmetic nodes and evaluate it both ways."""
        rng = random.Random(seed)
        ops = {"base.add": lambda a, b: a + b, "base.subtract": lambda a, b: a - b,
               "base.multiply": lambda a, b: a * b, "base.maximum": max,
               "base.minimum": min}
        g = Graph()
        expected = {}
        for i in range(rng.randint(1, 4)):
            value = rng.choice([rng.randint(-5, 5), round(rng.uniform(-5, 5), 3)])
            expected[const(g, value, f"c{i}").id] = value
        for i in range(rng.randint(1, 25)):
            spec_id = rng.choice(list(ops))
            node = g.add_node(spec_id, f"n{i}")
            a, b = rng.choice(list(expected)), rng.choice(list(expected))
            g.connect(a, "out", node, "a")
            g.connect(b, "out", node, "b")
            x, y = expected[a], expected[b]
            out = ops[spec_id](x, y)
            if spec_id in ("base.maximum", "base.minimum") and float in (type(x), type(y)):
                out = float(out)
            expected[node.id] = out
        result = run(g, check_outputs=True)
        for node_id, value in expected.items():
            assert result[node_id] == pytest.approx(value) and type(result[node_id]) is type(value)
        # ... and again after a trip through the file format
        assert run(Graph.loads(g.dumps())).values == result.values

    def test_deep_chain(self):
        g = Graph()
        const(g, 0, "start")
        previous = "start"
        for i in range(2000):
            node = g.add_node("base.add", f"n{i:04d}", values={"b": 1})
            g.connect(previous, "out", node, "a")
            previous = node.id
        assert run(g)[previous] == 2000

    def test_wide_graph(self):
        g = Graph()
        for i in range(500):
            g.add_node("base.negate", f"n{i}", values={"x": i})
        result = run(g)
        assert sum(result[f"n{i}"] for i in range(500)) == -sum(range(500))
