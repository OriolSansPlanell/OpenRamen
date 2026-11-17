"""Value semantics: what a node may and may not do to the data it is given."""

from dataclasses import replace

import numpy as np
import pytest

import openramen.runtime as runtime
from openramen import Graph, NodeError, analyze, compile_graph, run
from openramen.library import default_registry, spec_from_function
from openramen.library.importer import Override, import_module
from openramen.model import OUTPUTS

from helpers import const, links


@pytest.fixture(autouse=True)
def forget_what_was_learned():
    runtime._NEEDS_WRITABLE.clear()
    runtime._VERIFIED.clear()
    yield
    runtime._NEEDS_WRITABLE.clear()
    runtime._VERIFIED.clear()


@pytest.fixture
def registry():
    r = default_registry()

    @r.node("t.fill", passthrough=("a",))
    def fill(a: np.ndarray, value: float = 9.0) -> None:
        a[...] = value

    @r.node("t.scale", mutates=("a",))
    def scale(a: np.ndarray, factor: float) -> np.ndarray:
        a *= factor
        return a

    @r.node("t.sneaky")
    def sneaky(a: np.ndarray) -> float:
        a[0] = -1.0                                  # writes without declaring it
        return float(a.sum())

    @r.node("t.push", passthrough=("items",))
    def push(items: list, item: int) -> None:
        items.append(item)

    @r.node("t.sneaky_list")
    def sneaky_list(items: list) -> int:
        items.append(0)                              # undeclared mutation of a list
        return len(items)

    @r.node("t.honest")
    def honest(items: list) -> int:
        return len(items)

    @r.node("t.total")
    def total(a: np.ndarray) -> float:
        return float(a.sum())

    @r.node("t.writable")
    def writable(a: np.ndarray) -> bool:
        return bool(a.flags.writeable)

    @r.node("t.same_object")
    def same_object(a: np.ndarray) -> np.ndarray:
        return a

    return r


def fan_out(registry, spec, data, **literals):
    """One source feeding a node under test and, untouched, a witness."""
    g = Graph()
    const(g, data, "src")
    g.add_node(spec, "n", values=literals)
    g.add_node("base.identity", "witness")
    pin = registry.get(spec).inputs[0].name
    g.connect("src", "out", "n", pin)
    g.connect("src", "out", "witness", "value")
    return g


class TestDeclaredMutation:
    def test_a_mutating_node_gets_its_own_copy(self, registry):
        data = np.arange(4.0)
        g = fan_out(registry, "t.scale", data, factor=10.0)
        result = run(g, registry, check_outputs=True)
        assert result["n"].tolist() == [0, 10, 20, 30]
        assert result["witness"].tolist() == [0, 1, 2, 3]          # the branch is unharmed
        assert result["src"].tolist() == [0, 1, 2, 3]
        assert data.tolist() == [0, 1, 2, 3]                       # and so is the graph's constant
        assert run(g, registry)["n"].tolist() == [0, 10, 20, 30]   # running again starts clean

    def test_passthrough_hands_the_written_object_on(self, registry):
        spec = registry.get("t.fill")
        assert [(p.name, p.aliases, p.passthrough) for p in spec.outputs] == [("a", "a", True)]
        assert spec.input("a").access == "mutate"
        g = fan_out(registry, "t.fill", np.zeros((2, 2)), value=3.0)
        assert str(analyze(g, registry).out_type("n", "a")) == "Array[float64, 2]"
        result = run(g, registry, check_outputs=True)
        assert result["n"].tolist() == [[3, 3], [3, 3]] and result["witness"].sum() == 0
        # a list written in place may hold anything afterwards: the declared type stays
        g = fan_out(registry, "t.push", [1, 2], item=3)
        assert str(analyze(g, registry).out_type("n", "items")) == "List[Any]"

    def test_chained_in_place_nodes(self, registry):
        g = Graph()
        const(g, np.zeros(3), "src")
        g.add_node("t.fill", "one", values={"value": 1.0})
        g.add_node("t.fill", "two", values={"value": 2.0})
        g.add_node("t.total", "sum_one")
        links(g, "src.out > one.a", "one.a > two.a", "one.a > sum_one.a")
        result = run(g, registry, check_outputs=True)
        assert result["one"].tolist() == [1, 1, 1] and result["two"].tolist() == [2, 2, 2]
        assert result["sum_one"] == 3.0 and result["src"].sum() == 0

    def test_lists_and_nested_containers_are_copied_deeply(self, registry):
        r = registry

        @r.node("t.poke", passthrough=("table",))
        def poke(table: list) -> None:
            table[0].append(99)
            table[1]["k"][0] = -1.0

        value = [[1], {"k": np.zeros(2)}]
        g = fan_out(r, "t.poke", value)
        result = run(g, r)
        assert result["n"][0] == [1, 99] and result["n"][1]["k"].tolist() == [-1, 0]
        assert result["witness"][0] == [1] and result["witness"][1]["k"].tolist() == [0, 0]
        assert value[0] == [1]

    def test_literals_are_fresh_for_every_call(self, registry):
        g = Graph()
        loop = g.add_for("loop")
        loop.values["count"] = 3
        loop.add_output("sizes", mode="collect")
        f = loop.frame
        f.add_node("t.push", "p", values={"items": [1], "item": 5})
        f.add_node("base.list_length", "n")
        links(f, "p.items > n.items", "n.out > outputs.sizes")
        assert run(g, registry)["loop"] == [2, 2, 2]
        assert loop.frame.node("p").values["items"] == [1]

    def test_numpy_in_place_functions(self):
        registry = default_registry()
        registry.import_module("numpy")
        g = fan_out(registry, "numpy.fill_diagonal", np.zeros((3, 3)), val=7.0)
        result = run(g, registry, check_outputs=True)
        assert np.array_equal(result["n"], np.eye(3) * 7) and result["witness"].sum() == 0
        g = fan_out(registry, "numpy.put", np.zeros(4), ind=[1, 3], v=[5.0, 6.0])
        result = run(g, registry, check_outputs=True)
        assert result["n"].tolist() == [0, 5, 0, 6] and result["witness"].sum() == 0

    def test_passthrough_in_an_override(self):
        import types
        module = types.ModuleType("fakelib")

        def clear(buffer, value=0.0):
            """Overwrite a buffer.

            Parameters
            ----------
            buffer : ndarray
            value : float
            """
            buffer[...] = value

        clear.__module__ = "fakelib"
        module.clear = clear
        registry = default_registry()
        import_module(registry, module, overrides={
            "clear": Override(passthrough=("buffer",), outputs=[])})
        spec = registry.get("fakelib.clear")
        assert spec.input("buffer").access == "mutate"
        assert [(p.name, p.passthrough) for p in spec.outputs] == [("buffer", True)]
        a = np.ones(2)
        assert spec.call({"buffer": a, "value": 4.0}, {})["buffer"] is a and a[0] == 4.0
        direct = spec_from_function(lambda a, v: a.fill(v), "t.fill2", passthrough=("a",),
                                    types={"a": "Array", "v": "Float"})
        assert [p.name for p in direct.outputs] == ["a"] and direct.outputs[0].passthrough


class TestFrozenArrays:
    def test_reading_nodes_see_read_only_views(self, registry):
        g = fan_out(registry, "t.writable", np.zeros(3))
        result = run(g, registry)
        assert result["n"] is False

    def test_views_cost_no_copy(self, registry):
        data = np.zeros(1000)
        g = fan_out(registry, "t.same_object", data)
        result = run(g, registry)
        assert np.shares_memory(result["n"], result["src"])

    def test_an_undeclared_write_is_contained(self, registry):
        events = []
        g = fan_out(registry, "t.sneaky", np.ones(4))
        result = run(g, registry, listener=events.append)
        assert result["n"] == 2.0                                   # -1 + 1 + 1 + 1, on a copy
        assert result["witness"].tolist() == [1, 1, 1, 1]           # nobody else saw the write
        warnings = [e for e in events if e.kind == "node-warning"]
        assert len(warnings) == 1 and warnings[0].node == "n"
        assert "t.sneaky needs writable arrays" in warnings[0].message
        assert "private copies" in warnings[0].message
        # from now on the node is handed copies up front, silently
        events.clear()
        again = run(g, registry, listener=events.append)
        assert again["n"] == 2.0 and again["witness"].tolist() == [1, 1, 1, 1]
        assert not [e for e in events if e.kind == "node-warning"]
        assert "t.sneaky" in runtime._NEEDS_WRITABLE

    def test_other_errors_are_not_retried(self, registry):
        calls = []

        @registry.node("t.broken")
        def broken(a: np.ndarray) -> float:
            calls.append(1)
            raise ValueError("something else went wrong")

        with pytest.raises(NodeError, match="something else went wrong"):
            run(fan_out(registry, "t.broken", np.ones(2)), registry)
        assert calls == [1] and "t.broken" not in runtime._NEEDS_WRITABLE

    def test_a_retry_that_fails_reports_the_second_error(self, registry):
        @registry.node("t.twice")
        def twice(a: np.ndarray) -> float:
            a[0] = 1.0
            raise KeyError("after the write")

        with pytest.raises(NodeError, match="after the write"):
            run(fan_out(registry, "t.twice", np.ones(2)), registry)
        assert "t.twice" not in runtime._NEEDS_WRITABLE

    def test_results_inside_containers_stay_safe(self, registry):
        # the array travels inside a list; a declared mutation copies deeply
        @registry.node("t.first_to_zero", passthrough=("items",))
        def first_to_zero(items: list) -> None:
            items[0][...] = 0.0

        g = fan_out(registry, "t.first_to_zero", [np.ones(2)])
        result = run(g, registry)
        assert result["n"][0].tolist() == [0, 0] and result["witness"][0].tolist() == [1, 1]

    def test_frames_keep_the_protection(self, registry):
        g = Graph()
        const(g, np.ones((2, 3)), "src")
        loop = g.add_for("loop")
        loop.add_input("row", mode="index")
        loop.add_input("whole")
        loop.add_output("flags", mode="collect")
        loop.add_output("sums", mode="collect")
        f = loop.frame
        f.add_node("t.writable", "w")
        f.add_node("t.sneaky", "s")
        links(f, "inputs.whole > w.a", "inputs.row > s.a", "w.out > outputs.flags",
              "s.out > outputs.sums")
        links(g, "src.out > loop.row", "src.out > loop.whole")
        result = run(g, registry)
        assert result["loop"] == {"flags": [False, False], "sums": [1.0, 1.0]}
        assert result["src"].tolist() == [[1, 1, 1], [1, 1, 1]]


class TestMutationDetector:
    def test_an_undeclared_list_mutation_is_caught(self, registry):
        g = fan_out(registry, "t.sneaky_list", [1, 2, 3])
        # without the detector the write goes unnoticed and leaks to the other branch
        leaked = run(g, registry)
        assert leaked["witness"] == [1, 2, 3, 0]
        with pytest.raises(NodeError, match="modified its input 'items' in place") as info:
            run(g, registry, detect_mutation=True)
        error = info.value
        assert error.node == "n" and error.pin == "items" and error.spec == "t.sneaky_list"
        assert "mutates or passthrough" in str(error)

    def test_declared_and_honest_nodes_pass(self, registry):
        for spec, literals in (("t.honest", {}), ("t.push", {"item": 4})):
            g = fan_out(registry, spec, [1, 2, 3], **literals)
            result = run(g, registry, detect_mutation=True)
            assert result["witness"] == [1, 2, 3]
        assert ("t.honest", registry.get("t.honest").version) in runtime._VERIFIED

    @pytest.mark.parametrize("value, change", [
        ({"a": 1}, lambda v: v.update(b=2)),
        ({1, 2}, lambda v: v.add(3)),
        (bytearray(b"ab"), lambda v: v.extend(b"c")),
        ([[1, 2], [3]], lambda v: v[1].append(4)),
        ([{"k": [1.0]}], lambda v: v[0]["k"].__setitem__(0, float("nan"))),
        ([1.0], lambda v: v.__setitem__(0, 1)),
    ])
    def test_kinds_of_changes(self, registry, value, change):
        @registry.node("t.change")
        def changer(value) -> None:
            change(value)

        g = Graph()
        const(g, value, "src")
        g.add_node("t.change", "n")
        g.connect("src", "out", "n", "value")
        with pytest.raises(NodeError, match="modified its input 'value'"):
            run(g, registry, detect_mutation=True)

    def test_nan_is_not_a_change(self, registry):
        g = fan_out(registry, "t.honest", [float("nan"), np.array([np.nan])])
        assert run(g, registry, detect_mutation=True)["n"] == 2

    def test_found_inside_loops(self, registry):
        g = Graph()
        const(g, [1], "src")
        loop = g.add_for("loop")
        loop.values["count"] = 2
        loop.add_input("items")
        loop.frame.add_node("t.sneaky_list", "s")
        loop.frame.connect("inputs", "items", "s", "items")
        g.connect("src", "out", loop, "items")
        with pytest.raises(NodeError, match="modified its input") as info:
            run(g, registry, detect_mutation=True)
        assert info.value.node == "loop/s" and info.value.context == ["iteration 0 of 'loop'"]


class TestInPlaceRegisters:
    def accumulate(self, registry, count=4, extra=()):
        g = Graph()
        const(g, np.zeros(3), "start")
        loop = g.add_for("loop")
        loop.values["count"] = count
        loop.add_register("buffer")
        g.connect("start", "out", loop, "buffer")
        f = loop.frame
        f.add_node("t.add_one", "bump")
        links(f, "inputs.buffer > bump.a", "bump.a > outputs.buffer", *extra)
        return g, loop

    @pytest.fixture
    def counting(self, registry):
        seen = []

        @registry.node("t.add_one", passthrough=("a",))
        def add_one(a: np.ndarray) -> None:
            seen.append(a)
            a += 1.0

        return registry, seen

    def test_the_accumulate_pattern_uses_one_buffer(self, counting):
        registry, seen = counting
        g, loop = self.accumulate(registry, count=5)
        plan = compile_graph(g, registry)
        assert plan.steps[-1].structure.in_place == {"buffer"}
        result = run(plan)
        assert result["loop"].tolist() == [5, 5, 5]
        assert len({id(a) for a in seen}) == 1                                  # one buffer for the whole loop
        assert result["start"].tolist() == [0, 0, 0]                # copied once, on the way in
        assert not np.shares_memory(result["loop"], result["start"])
        seen.clear()
        assert run(plan)["loop"].tolist() == [5, 5, 5]              # and nothing carries over

    def test_zero_iterations(self, counting):
        registry, seen = counting
        g, loop = self.accumulate(registry, count=0)
        result = run(g, registry)
        assert result["loop"].tolist() == [0, 0, 0] and seen == []

    def test_a_second_reader_switches_the_optimisation_off(self, counting):
        registry, seen = counting
        g, loop = self.accumulate(registry, count=3)
        loop.add_output("before", mode="collect")
        loop.frame.add_node("t.total", "peek")
        links(loop.frame, "inputs.buffer > peek.a", "peek.out > outputs.before")
        plan = compile_graph(g, registry)
        assert plan.steps[-1].structure.in_place == set()
        result = run(plan)
        # the reader sees the value before this iteration's write, every time
        assert result["loop"]["before"] == [0.0, 3.0, 6.0]
        assert result["loop"]["buffer"].tolist() == [3, 3, 3]
        assert len({id(a) for a in seen}) == 3

    def test_a_reader_of_the_result_switches_it_off_too(self, counting):
        registry, seen = counting
        g, loop = self.accumulate(registry, count=3)
        loop.add_output("history", mode="collect")
        loop.frame.connect("bump", "a", OUTPUTS, "history")
        plan = compile_graph(g, registry)
        assert plan.steps[-1].structure.in_place == set()
        history = run(plan)["loop"]["history"]
        assert [h.tolist() for h in history] == [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

    def test_swapped_registers_are_not_optimised(self, counting):
        registry, seen = counting
        g = Graph()
        const(g, np.zeros(2), "start")
        loop = g.add_for("loop")
        loop.values["count"] = 3
        loop.add_register("a")
        loop.add_register("b")
        links(g, "start.out > loop.a", "start.out > loop.b")
        f = loop.frame
        f.add_node("t.add_one", "bump")
        links(f, "inputs.a > bump.a", "bump.a > outputs.b", "inputs.b > outputs.a")
        plan = compile_graph(g, registry)
        assert plan.steps[-1].structure.in_place == set()
        a, b = np.zeros(2), np.zeros(2)
        for _ in range(3):
            a, b = b, a + 1.0
        result = run(plan)["loop"]
        assert result["a"].tolist() == a.tolist() and result["b"].tolist() == b.tolist()

    def test_same_answer_with_and_without_the_optimisation(self, counting):
        registry, seen = counting
        fast, _ = self.accumulate(registry, count=7)
        slow, loop = self.accumulate(registry, count=7)
        loop.frame.add_node("t.total", "peek")
        loop.frame.connect("inputs", "buffer", "peek", "a")
        assert compile_graph(slow, registry).steps[-1].structure.in_place == set()
        assert run(fast, registry)["loop"].tolist() == run(slow, registry)["loop"].tolist()

    def test_list_registers(self, registry):
        g = Graph()
        loop = g.add_for("loop")
        loop.values.update(count=1000, items=[])
        loop.add_register("items")
        loop.frame.add_node("t.push", "p")
        links(loop.frame, "inputs.items > p.items", "inputs.i > p.item", "p.items > outputs.items")
        plan = compile_graph(g, registry)
        assert plan.steps[-1].structure.in_place == {"items"}
        assert run(plan)["loop"] == list(range(1000))
        assert loop.values["items"] == []

    def test_in_a_while_loop(self, counting):
        registry, seen = counting
        g = Graph()
        const(g, np.zeros(2), "start")
        loop = g.add_while("loop")
        loop.add_register("buffer")
        g.connect("start", "out", loop, "buffer")
        f = loop.frame
        f.add_node("t.add_one", "bump")
        f.add_node("base.greater_equal", "done", values={"b": 3})
        links(f, "inputs.buffer > bump.a", "bump.a > outputs.buffer", "inputs.i > done.a",
              "done.out > outputs.stop")
        plan = compile_graph(g, registry)
        assert plan.steps[-1].structure.in_place == {"buffer"}
        assert run(plan)["loop"].tolist() == [4, 4] and len({id(a) for a in seen}) == 1

    @pytest.mark.parametrize("reader", ["aa_peek", "zz_peek"])
    def test_any_other_reader_of_the_result_switches_it_off(self, counting, reader):
        registry, seen = counting
        g, loop = self.accumulate(registry, count=3)
        loop.add_output("sums", mode="collect")
        loop.frame.add_node("t.total", reader)
        links(loop.frame, f"bump.a > {reader}.a", f"{reader}.out > outputs.sums")
        plan = compile_graph(g, registry)
        assert plan.steps[-1].structure.in_place == set()
        result = run(plan)["loop"]
        assert result["sums"] == [3.0, 6.0, 9.0] and result["buffer"].tolist() == [3, 3, 3]

    def test_even_a_reader_that_feeds_nothing_switches_it_off(self, counting):
        registry, seen = counting
        g, loop = self.accumulate(registry, count=3)
        loop.frame.add_node("t.total", "zz_watch")       # runs after the frame's outputs
        loop.frame.connect("bump", "a", "zz_watch", "a")
        plan = compile_graph(g, registry)
        frame = plan.steps[-1].structure.frames[0]
        assert [s.node for s in frame.steps] == ["inputs", "bump", "outputs", "zz_watch"]
        assert plan.steps[-1].structure.in_place == set()
        assert run(plan)["loop"].tolist() == [3, 3, 3]

    def aliasing(self, registry, id, fn, **options):
        """Register a node whose output shares memory with its input ``a``."""
        spec = spec_from_function(fn, id, **options)
        spec = replace(spec, outputs=(replace(spec.outputs[0], aliases="a"),))
        return registry.register(spec)

    def test_a_node_that_returns_something_else_cannot_corrupt_it(self, registry):
        # The node claims to hand its buffer on but returns another input: an
        # array that other nodes can see.  The loop must not write into that.
        def swap(a: np.ndarray, other: np.ndarray) -> np.ndarray:
            a += 1.0
            return other

        self.aliasing(registry, "t.swap", swap, mutates=("a",))
        g = Graph()
        const(g, np.zeros(2), "start")
        const(g, np.full(2, 5.0), "shared")
        loop = g.add_for("loop")
        loop.values["count"] = 4
        loop.add_register("buffer")
        loop.add_input("shared")
        links(g, "start.out > loop.buffer", "shared.out > loop.shared")
        loop.frame.add_node("t.swap", "swap")
        links(loop.frame, "inputs.buffer > swap.a", "inputs.shared > swap.other",
              "swap.out > outputs.buffer")
        g.add_node("t.total", "after")
        g.connect("shared", "out", "after", "a")
        plan = compile_graph(g, registry)
        assert plan.steps[-1].structure.in_place == {"buffer"}
        events = []
        result = run(plan, listener=events.append)
        assert result["shared"].tolist() == [5, 5] and result["after"] == 10.0
        assert result["loop"].tolist() == [5, 5]
        assert not [e for e in events if e.kind == "node-warning"]

    def test_a_view_of_the_register_is_not_a_buffer_to_write_in(self, registry):
        flags = []

        def backwards(a: np.ndarray) -> np.ndarray:
            flags.append(a.flags.writeable)
            return a[::-1]

        self.aliasing(registry, "t.backwards", backwards)
        g = Graph()
        const(g, np.arange(3.0), "start")
        loop = g.add_for("loop")
        loop.values["count"] = 3
        loop.add_register("buffer")
        g.connect("start", "out", loop, "buffer")
        loop.frame.add_node("t.backwards", "flip")
        links(loop.frame, "inputs.buffer > flip.a", "flip.out > outputs.buffer")
        plan = compile_graph(g, registry)
        assert plan.steps[-1].structure.in_place == set()
        assert run(plan)["loop"].tolist() == [2, 1, 0] and flags == [False] * 3

