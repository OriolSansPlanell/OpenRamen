"""Running independent nodes at the same time."""

import os
import threading
import time

import numpy as np
import pytest

import openramen.runtime as runtime
from openramen import Graph, NodeError, compile_graph, run
from openramen.library import default_registry
from openramen.runtime import Cancelled

from helpers import const, links

NAP = 0.25
#: These tests are about choosing a thread; the stress mode overrides the choice.
adaptive = pytest.mark.skipif(bool(os.environ.get("OPENRAMEN_STRESS")),
                              reason="the stress mode dispatches everything")


class Probe:
    """Records which nodes ran when, on which thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = 0
        self.most = 0
        self.log = []
        self.threads = {}

    def enter(self, label):
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
            self.threads[label] = threading.current_thread().name

    def leave(self, label):
        with self.lock:
            self.running -= 1
            self.log.append(label)


@pytest.fixture
def probe():
    return Probe()


@pytest.fixture
def registry(probe):
    r = default_registry()

    @r.node("t.nap")
    def nap(x: float, label: str = "", seconds: float = NAP) -> float:
        probe.enter(label)
        time.sleep(seconds)
        probe.leave(label)
        return x

    @r.node("t.quick")
    def quick(x: float, label: str = "") -> float:
        probe.enter(label)
        probe.leave(label)
        return x + 1.0

    @r.node("t.boom")
    def boom(x: float, seconds: float = 0.0, busy: int = 0, label: str = "") -> float:
        """Fail: after a pause, or once ``busy`` other nodes are running."""
        if label:
            probe.enter(label)
        deadline = time.perf_counter() + 10.0
        while probe.running < busy + bool(label) and time.perf_counter() < deadline:
            time.sleep(0.001)
        time.sleep(seconds)
        raise ValueError(f"boom {x}")

    @r.node("t.write", pure=False)
    def write(x: float, label: str = "", seconds: float = 0.0) -> float:
        time.sleep(seconds)
        probe.leave(label)
        return x

    @r.node("legacy.step", thread_safe=False)
    def legacy(x: float, label: str = "") -> float:
        probe.enter(label)
        time.sleep(0.03)
        probe.leave(label)
        return x

    @r.node("t.sort")
    def sort(a: np.ndarray) -> np.ndarray:
        return np.sort(a)

    return r


def branches(count, spec="t.nap", **literals):
    """``count`` independent nodes fed by one constant, and their sum."""
    g = Graph()
    const(g, 1.0, "src")
    for k in range(count):
        g.add_node(spec, f"n{k}", values={"label": f"n{k}", **literals})
        g.connect("src", "out", f"n{k}", "x")
    return g


def timed(*args, **kwargs):
    t0 = time.perf_counter()
    result = run(*args, **kwargs)
    return result, time.perf_counter() - t0


class TestParallelBranches:
    def test_two_branches_take_the_time_of_one(self, registry, probe):
        g = branches(2)
        result, parallel = timed(g, registry)
        assert result["n0"] == result["n1"] == 1.0 and probe.most == 2
        _, serial = timed(g, registry, workers=1)
        assert serial > NAP * 1.9 and probe.most == 2
        assert parallel < serial * 0.8                 # ideally half

    def test_workers_limit_how_many_run_at_once(self, registry, probe):
        g = branches(6, seconds=0.15)
        run(g, registry, workers=3)
        assert probe.most == 3
        probe.most = 0
        run(g, registry, workers=1)
        assert probe.most == 1 and set(probe.threads.values()) == {"MainThread"}

    @adaptive
    def test_a_chain_needs_no_other_thread(self, registry, probe):
        g = Graph()
        const(g, 1.0, "src")
        previous = "src.out"
        for k in range(4):
            g.add_node("t.nap", f"n{k}", values={"label": f"n{k}", "seconds": 0.01})
            links(g, f"{previous} > n{k}.x")
            previous = f"n{k}.x" if False else f"n{k}.out"
        plan = compile_graph(g, registry)
        assert plan.schedule is None
        run(plan)
        assert set(probe.threads.values()) == {"MainThread"} and probe.most == 1

    @adaptive
    def test_a_single_ready_node_runs_in_the_calling_thread(self, registry, probe):
        # a diamond: n0 and n1 in parallel, then one node that waits for both
        g = branches(2, seconds=0.02)
        g.add_node("base.add", "sum")
        g.add_node("t.nap", "after", values={"label": "after", "seconds": 0.02})
        links(g, "n0.out > sum.a", "n1.out > sum.b", "sum.out > after.x")
        run(g, registry)
        assert probe.threads["after"] == "MainThread"
        assert {probe.threads["n0"], probe.threads["n1"]} != {"MainThread"}

    @adaptive
    def test_nodes_that_prove_cheap_stay_in_the_calling_thread(self, registry, probe,
                                                                monkeypatch):
        monkeypatch.setattr(runtime, "_CHEAP", 0.05)   # so a hiccup of the machine cannot matter
        plan = compile_graph(branches(3, spec="t.quick"), registry)
        assert plan.schedule is not None and all(s.cost < 0 for s in plan.steps)
        run(plan)                                    # unknown cost: they are dispatched
        assert "MainThread" not in probe.threads.values()
        assert all(0 <= s.cost < runtime._CHEAP for s in plan.steps if s.heavy)
        probe.threads.clear()
        run(plan)                                    # known to be cheap: no dispatch
        assert set(probe.threads.values()) == {"MainThread"}

    @adaptive
    def test_base_nodes_alone_are_never_scheduled(self):
        g = Graph()
        const(g, 2, "a")
        g.add_node("base.add", "x", values={"b": 1})
        g.add_node("base.multiply", "y", values={"b": 3})
        links(g, "a.out > x.a", "a.out > y.a")
        plan = compile_graph(g)
        assert plan.schedule is None and not plan.heavy and plan.pure
        assert run(plan)["x"] == 3

    def test_numpy_work_runs_on_two_threads(self, registry):
        g = Graph()
        rng = np.random.default_rng(0)
        for name in ("a", "b"):
            const(g, rng.random(200_000), name)
            g.add_node("t.sort", f"sorted_{name}")
            g.connect(name, "out", f"sorted_{name}", "a")
        result = run(g, registry, check_outputs=True)
        for name in ("a", "b"):
            assert np.array_equal(result[f"sorted_{name}"], np.sort(result[name]))

    def test_results_do_not_depend_on_workers(self, registry):
        g = branches(5, spec="t.quick")
        previous = "src"
        for k in range(5):                           # ((((1 + n0) + n1) + n2) ...
            g.add_node("base.add", f"sum{k}")
            links(g, f"{previous}.out > sum{k}.a", f"n{k}.out > sum{k}.b")
            previous = f"sum{k}"
        expected = run(g, registry, workers=1)
        for workers in (2, 4, 16):
            result = run(g, registry, workers=workers)
            assert result.values == expected.values and result["sum4"] == 11.0

    def test_durations_and_outputs_are_complete(self, registry):
        g = branches(3, seconds=0.01)
        result = run(g, registry)
        assert set(result.durations) == {"src", "n0", "n1", "n2"}
        assert all(result.durations[f"n{k}"] >= 0.009 for k in range(3))

    def test_targets(self, registry, probe):
        g = branches(4, seconds=0.01)
        result = run(g, registry, targets=["n0", "n2"])
        assert "n1" not in result and sorted(probe.log) == ["n0", "n2"] and probe.most == 2


class TestOrderAndLocks:
    def test_side_effects_keep_their_order(self, registry, probe):
        g = Graph()
        const(g, 1.0, "src")
        naps = [0.06, 0.0, 0.03, 0.0, 0.01]
        for k, seconds in enumerate(naps):
            g.add_node("t.write", f"w{k}", values={"label": f"w{k}", "seconds": seconds})
            g.connect("src", "out", f"w{k}", "x")
        g.add_node("t.nap", "pure", values={"label": "pure", "seconds": 0.01})
        g.connect("src", "out", "pure", "x")
        plan = compile_graph(g, registry)
        assert plan.schedule is not None and not plan.pure
        run(plan)
        assert [label for label in probe.log if label != "pure"] == [f"w{k}" for k in range(5)]

    def test_side_effects_inside_structures_count(self, registry, probe):
        g = Graph()
        const(g, 1.0, "src")
        for k, seconds in enumerate([0.05, 0.0]):
            sub = g.add_subgraph(f"s{k}")
            sub.add_input("x")
            sub.frame.add_node("t.write", "w", values={"label": f"s{k}", "seconds": seconds})
            sub.frame.connect("inputs", "x", "w", "x")
            g.connect("src", "out", sub, "x")
        plan = compile_graph(g, registry)
        assert [s.pure for s in plan.steps] == [True, False, False]
        run(plan)
        assert probe.log == ["s0", "s1"]

    def test_a_library_that_is_not_thread_safe_runs_one_node_at_a_time(self, registry, probe):
        g = branches(4, spec="legacy.step")
        run(g, registry)
        assert probe.most == 1 and len(probe.log) == 4
        assert len(set(probe.threads.values())) > 1          # they were on several threads

    def test_one_object_is_used_by_one_node_at_a_time(self, registry):
        class Counter:
            def __init__(self):
                self.inside = 0
                self.most = 0
                self.total = 0
                self.spans = []

        made = []

        @registry.node("t.make_counter")
        def make_counter() -> object:
            made.append(Counter())
            return made[-1]

        @registry.node("t.bump")
        def bump(counter: object, label: str = "") -> int:
            counter.inside += 1
            counter.most = max(counter.most, counter.inside)
            counter.spans.append([time.perf_counter(), 0.0])
            time.sleep(0.05)
            counter.spans[-1][1] = time.perf_counter()
            counter.total += 1
            counter.inside -= 1
            return counter.total

        g = Graph()
        g.add_node("t.make_counter", "shared")
        g.add_node("t.make_counter", "other")
        for k in range(4):
            g.add_node("t.bump", f"b{k}")
            g.connect("shared" if k < 3 else "other", "out", f"b{k}", "counter")
        result = run(g, registry, workers=4)
        shared, other = result["shared"], result["other"]
        assert shared.most == 1 and shared.total == 3 and other.total == 1
        assert sorted(result[f"b{k}"] for k in range(3)) == [1, 2, 3]
        spans = sorted(shared.spans)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))    # one after the other
        (began, ended), = other.spans                        # the other object did not wait:
        assert began < spans[-1][0]                          # it started before the last of them

    def test_the_listener_is_called_one_event_at_a_time(self, registry):
        inside, overlaps, events = [0], [], []

        def listener(event):
            inside[0] += 1
            overlaps.append(inside[0])
            time.sleep(0.001)
            events.append(event)
            inside[0] -= 1

        run(branches(6, seconds=0.0), registry, listener=listener, workers=6)
        assert max(overlaps) == 1
        kinds = [e.kind for e in events]
        assert kinds[0] == "graph-started" and kinds[-1] == "graph-finished"
        for k in range(6):
            started = events.index(next(e for e in events
                                        if e.kind == "node-started" and e.node == f"n{k}"))
            finished = events.index(next(e for e in events
                                         if e.kind == "node-finished" and e.node == f"n{k}"))
            assert started < finished


class TestFailures:
    def test_a_failure_stops_the_rest(self, registry, probe):
        g = branches(2, seconds=0.05)
        g.add_node("t.boom", "bad", values={"busy": 2})      # fails once both naps are running
        g.connect("src", "out", "bad", "x")
        g.add_node("t.nap", "later", values={"label": "later", "seconds": 0.0})
        g.connect("n0", "out", "later", "x")
        events = []
        with pytest.raises(NodeError, match="boom 1.0") as info:
            run(g, registry, listener=events.append, workers=4)
        assert info.value.node == "bad" and isinstance(info.value.original, ValueError)
        assert "later" not in probe.log                        # never started
        assert sorted(probe.log) == ["n0", "n1"]                # what was running finished
        assert events[-1].kind == "graph-failed" and events[-1].error is info.value
        assert [e.node for e in events if e.kind == "node-failed"] == ["bad"]

    def test_the_first_failing_node_in_order_is_reported(self, registry, probe):
        g = Graph()
        const(g, 1.0, "one")
        const(g, 2.0, "two")
        # 'b_fast' fails first, while 'a_slow' is running; then 'a_slow' fails too
        g.add_node("t.boom", "a_slow", values={"seconds": 0.05, "label": "a_slow"})
        g.add_node("t.boom", "b_fast", values={"busy": 1})
        links(g, "one.out > a_slow.x", "two.out > b_fast.x")
        for _ in range(3):
            probe.running = 0
            with pytest.raises(NodeError) as info:
                run(g, registry)
            assert info.value.node == "a_slow"

    def test_failure_inside_a_nested_frame(self, registry, probe):
        g = Graph()
        const(g, 1.0, "src")
        for k in range(2):
            sub = g.add_subgraph(f"s{k}")
            sub.add_input("x")
            sub.add_output("y")
            for j in range(2):
                sub.frame.add_node("t.nap", f"n{j}", values={"label": f"s{k}n{j}",
                                                           "seconds": 0.02})
                sub.frame.connect("inputs", "x", f"n{j}", "x")
            sub.frame.add_node("t.boom" if k else "base.add", "end")
            if k:
                sub.frame.connect("n0", "out", "end", "x")
            else:
                links(sub.frame, "n0.out > end.a", "n1.out > end.b")
            sub.frame.connect("end", "out", "outputs", "y")
            g.connect("src", "out", sub, "x")
        with pytest.raises(NodeError) as info:
            run(g, registry)
        assert info.value.node == "s1/end" and info.value.context == ["in 's1'"]

    def test_cancel_while_nodes_are_running(self, registry, probe):
        g = branches(8, seconds=0.03)
        with pytest.raises(Cancelled):
            run(g, registry, cancel=lambda: len(probe.log) >= 2, workers=2)
        assert 2 <= len(probe.log) <= 4                # running nodes finished, the rest never ran

    def test_nothing_leaks_out_of_an_abandoned_frame(self, registry):
        # 'boom' fails at once while a loop in the other branch is busy: the
        # loop must stop early instead of running all its iterations
        g = Graph()
        const(g, 1.0, "src")
        loop = g.add_for("loop")
        loop.values["count"] = 400                      # two seconds, if it ran to the end
        loop.add_input("x")
        loop.add_output("y")
        loop.frame.add_node("t.nap", "n", values={"seconds": 0.005})
        links(loop.frame, "inputs.x > n.x", "n.out > outputs.y")
        g.add_node("t.boom", "bad", values={"seconds": 0.03})
        links(g, "src.out > loop.x", "src.out > bad.x")
        t0 = time.perf_counter()
        with pytest.raises(NodeError, match="boom"):
            run(g, registry)
        assert time.perf_counter() - t0 < 1.0


    def test_an_abandoned_frame_stops_handing_out_work(self, registry, probe):
        # as above, but the loop body itself runs two nodes at a time
        g = Graph()
        const(g, 1.0, "src")
        loop = g.add_for("loop")
        loop.values["count"] = 100
        loop.add_input("x")
        loop.add_output("y")
        for j in range(2):
            loop.frame.add_node("t.nap", f"n{j}", values={"label": f"n{j}", "seconds": 0.01})
            loop.frame.connect("inputs", "x", f"n{j}", "x")
        loop.frame.add_node("base.add", "sum")
        links(loop.frame, "n0.out > sum.a", "n1.out > sum.b", "sum.out > outputs.y")
        g.add_node("t.boom", "bad", values={"seconds": 0.03})
        links(g, "src.out > loop.x", "src.out > bad.x")
        with pytest.raises(NodeError, match="boom"):
            run(g, registry, workers=4)
        ran = len(probe.log)
        assert ran < 40                                # of 200, had the loop gone on
        time.sleep(0.05)
        assert len(probe.log) == ran                   # and nothing is still running


class TestNesting:
    def test_parallel_work_inside_parallel_structures(self, registry, probe):
        g = Graph()
        const(g, 1.0, "src")
        for k in range(2):
            sub = g.add_subgraph(f"s{k}")
            sub.add_input("x")
            sub.add_output("y")
            for j in range(2):
                sub.frame.add_node("t.nap", f"n{j}", values={"label": f"s{k}n{j}"})
                sub.frame.connect("inputs", "x", f"n{j}", "x")
            sub.frame.add_node("base.add", "sum")
            links(sub.frame, "n0.out > sum.a", "n1.out > sum.b", "sum.out > outputs.y")
            g.connect("src", "out", sub, "x")
        result, elapsed = timed(g, registry, workers=4)
        assert result["s0"] == result["s1"] == 2.0 and probe.most == 4
        _, serial = timed(g, registry, workers=1)
        assert elapsed < serial * 0.6                  # four naps at once: ideally a quarter

    def test_no_deadlock_when_no_frame_thread_is_free(self, registry, probe, monkeypatch):
        monkeypatch.setattr(runtime, "_SLOTS", threading.BoundedSemaphore(1))
        g = Graph()
        const(g, 1.0, "src")
        frame = g
        source = ("src", "out")
        for depth in range(4):                        # structures inside structures, two wide
            subs = []
            for k in range(2):
                sub = frame.add_subgraph(f"d{depth}k{k}")
                sub.add_input("x")
                sub.add_output("y")
                frame.connect(*source, sub, "x")
                sub.frame.add_node("t.nap", "n", values={"label": f"{depth}{k}",
                                                        "seconds": 0.005})
                links(sub.frame, "inputs.x > n.x", "n.out > outputs.y")
                subs.append(sub)
            if frame is not g:
                frame.connect(subs[0], "y", "outputs", "y", replace=True)
            frame, source = subs[1].frame, ("inputs", "x")
        result = run(g, registry, workers=2)
        assert result["d0k0"] == 1.0 and len(probe.log) == 8

    def test_loops_with_parallel_bodies(self, registry, probe):
        g = Graph()
        loop = g.add_for("loop")
        loop.values["count"] = 3
        loop.add_output("sums", mode="collect")
        f = loop.frame
        f.add_node("base.to_float", "x")
        for j in range(2):
            f.add_node("t.nap", f"n{j}", values={"label": f"n{j}", "seconds": 0.1})
            f.connect("x", "out", f"n{j}", "x")
        f.add_node("base.add", "sum")
        links(f, "inputs.i > x.x", "n0.out > sum.a", "n1.out > sum.b", "sum.out > outputs.sums")
        result, elapsed = timed(g, registry)
        assert result["loop"] == [0.0, 2.0, 4.0] and probe.most == 2
        _, serial = timed(g, registry, workers=1)
        assert elapsed < serial * 0.8                  # six naps, two at a time: ideally half


class TestParallelFor:
    def loop(self, registry, count=6, seconds=0.03, parallel=True, spec="t.nap"):
        g = Graph()
        const(g, [float(k) for k in range(count)], "items")
        loop = g.add_for("loop", parallel=parallel)
        loop.add_input("x", mode="index")
        loop.add_output("y", mode="collect")
        loop.add_output("last", fallback=-1.0)
        loop.add_output("index", mode="stack")
        loop.frame.add_node(spec, "n", values={"seconds": seconds})
        links(loop.frame, "inputs.x > n.x", "n.out > outputs.y", "n.out > outputs.last",
              "inputs.i > outputs.index")
        g.connect("items", "out", loop, "x")
        return g, loop

    def test_iterations_overlap_and_results_keep_their_order(self, registry, probe):
        g, loop = self.loop(registry, seconds=0.08)
        result, elapsed = timed(g, registry, workers=3, check_outputs=True)
        out = result["loop"]
        assert out["y"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] and out["last"] == 5.0
        assert out["index"].tolist() == [0, 1, 2, 3, 4, 5] and probe.most == 3
        _, serial = timed(g, registry, workers=1)
        assert elapsed < serial * 0.7                  # three at a time: ideally a third

    def test_same_results_as_the_ordinary_loop(self, registry):
        parallel, _ = self.loop(registry, seconds=0.0)
        ordinary, _ = self.loop(registry, seconds=0.0, parallel=False)
        a, b = run(parallel, registry)["loop"], run(ordinary, registry)["loop"]
        assert a["y"] == b["y"] and a["last"] == b["last"]
        assert a["index"].tolist() == b["index"].tolist()
        assert run(parallel, registry, workers=1)["loop"]["y"] == b["y"]

    @pytest.mark.parametrize("count", [0, 1])
    def test_few_iterations(self, registry, count):
        g, loop = self.loop(registry, count=count, seconds=0.0)
        out = run(g, registry, check_outputs=True)["loop"]
        assert out["y"] == [float(k) for k in range(count)]
        assert out["last"] == (0.0 if count else -1.0)

    def test_count_and_indexing_together(self, registry):
        g, loop = self.loop(registry, seconds=0.0)
        loop.values["count"] = 4
        assert run(g, registry)["loop"]["y"] == [0.0, 1.0, 2.0, 3.0]
        loop.values["count"] = -2
        assert run(g, registry)["loop"]["y"] == []

    def test_count_alone(self, registry):
        g = Graph()
        loop = g.add_for("loop", parallel=True)
        loop.values["count"] = 5
        loop.add_output("squares", mode="collect")
        loop.frame.add_node("base.multiply", "sq")
        links(loop.frame, "inputs.i > sq.a", "inputs.i > sq.b", "sq.out > outputs.squares")
        assert run(g, registry)["loop"] == [0, 1, 4, 9, 16]

    def test_the_earliest_failing_iteration_is_reported(self, registry):
        g = Graph()
        loop = g.add_for("loop", parallel=True)
        loop.values["count"] = 8
        loop.add_output("y", mode="collect")
        f = loop.frame
        f.add_node("base.to_float", "x")
        f.add_node("base.greater_equal", "late", values={"b": 3})
        case = f.add_case("pick", cases=[[True], [False]])
        case.add_input("x")
        case.add_output("y")
        case.frames[0].add_node("t.boom", "bad", values={"seconds": 0.01})
        links(case.frames[0], "inputs.x > bad.x", "bad.out > outputs.y")
        case.frames[1].add_node("t.nap", "ok", values={"seconds": 0.02})
        links(case.frames[1], "inputs.x > ok.x", "ok.out > outputs.y")
        links(f, "inputs.i > x.x", "inputs.i > late.a", "late.out > pick.selector",
              "x.out > pick.x", "pick.y > outputs.y")
        with pytest.raises(NodeError, match="boom 3.0") as info:
            run(g, registry, workers=4)
        assert info.value.node == "loop/pick[0]/bad"
        assert info.value.context == ["frame 0 of 'loop/pick'", "iteration 3 of 'loop'"]

    def test_cancel(self, registry):
        g, loop = self.loop(registry, count=400, seconds=0.01)   # a second, uncancelled
        started = time.perf_counter()
        with pytest.raises(Cancelled):
            run(g, registry, cancel=lambda: time.perf_counter() - started > 0.03, workers=4)
        assert time.perf_counter() - started < 0.6

    def test_shift_registers_are_refused(self, registry):
        from openramen import GraphError, analyze
        g, loop = self.loop(registry)
        with pytest.raises(GraphError, match="cannot have shift registers"):
            loop.add_register("total")
        loop.registers.append("total")
        problems = [p.message for p in analyze(g, registry).problems]
        assert any("only a For loop without shift registers" in m for m in problems)
        with pytest.raises(TypeError):
            Graph().add_while(parallel=True)             # only a For loop takes the option

    def test_file_format(self, registry):
        g, loop = self.loop(registry)
        text = g.dumps()
        assert '"parallel": true' in text
        loaded = Graph.loads(text)
        assert loaded.node("loop").parallel and loaded == g and loaded.dumps() == text
        assert g.copy().node("loop").parallel
        loop.parallel = False
        assert '"parallel"' not in g.dumps() and loaded != g

    def test_nested_parallel_loops(self, registry, probe):
        g = Graph()
        outer = g.add_for("outer", parallel=True)
        outer.values["count"] = 3
        outer.add_output("table", mode="stack")
        inner = outer.frame.add_for("inner", parallel=True)
        inner.values["count"] = 4
        inner.add_input("row")
        inner.add_output("line", mode="stack")
        f = inner.frame
        f.add_node("base.multiply", "product")
        f.add_node("base.to_float", "as_float")
        f.add_node("t.nap", "n", values={"seconds": 0.002})
        links(f, "inputs.row > product.a", "inputs.i > product.b", "product.out > as_float.x",
              "as_float.out > n.x", "n.out > outputs.line")
        links(outer.frame, "inputs.i > inner.row", "inner.line > outputs.table")
        expected = np.array([[float(r * c) for c in range(4)] for r in range(3)])
        for workers in (1, 2, 8):
            assert np.array_equal(run(g, registry, workers=workers)["outer"], expected)
