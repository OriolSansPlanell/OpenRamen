"""Background runs and Stop."""

import threading
import time

import pytest

import openramen as ramen
from openramen import Graph, GraphCheckError, NodeError, Run, start
from openramen.library import default_registry
from openramen.model import OUTPUTS
from openramen.runtime import Cancelled

from helpers import const, links


@pytest.fixture
def registry():
    r = default_registry()
    r.calls = []

    @r.node("t.nap")
    def nap(x: float, seconds: float = 0.02) -> float:
        r.calls.append(x)
        time.sleep(seconds)
        return x

    return r


def endless(body=None):
    """A While loop that never stops by itself."""
    g = Graph("endless")
    loop = g.add_while("loop")
    loop.add_register("n")
    loop.values["n"] = 0
    f = loop.frame
    f.add_node("base.add", "next", values={"b": 1})
    links(f, "inputs.n > next.a", "next.out > outputs.n")
    f.node(OUTPUTS).values["stop"] = False
    if body:
        body(f)
    return g


class TestStop:
    def test_stop_works_inside_a_loop(self):
        events = []
        handle = start(endless(), listener=events.append)
        assert isinstance(handle, Run) and not handle.done and "running" in repr(handle)
        time.sleep(0.05)
        assert not handle.done                       # it really is still going
        t0 = time.perf_counter()
        handle.cancel()
        assert handle.wait(2.0)
        assert time.perf_counter() - t0 < 1.0
        assert handle.done and handle.cancelled and isinstance(handle.error, Cancelled)
        with pytest.raises(Cancelled):
            handle.result()
        assert events[-1].kind == "graph-failed" and isinstance(events[-1].error, Cancelled)
        assert "cancelled" in repr(handle)
        started = sum(e.kind == "node-started" and e.node == "loop/next" for e in events)
        assert started > 10                          # the loop had been running

    def test_stop_inside_nested_loops_and_cases(self):
        def body(f):
            inner = f.add_for("inner")
            inner.values["count"] = 1000
            inner.add_output("last", fallback=0)
            case = inner.frame.add_case("pick", cases=[[0], None])
            case.add_output("v")
            case.frames[0].node(OUTPUTS).values["v"] = 1
            case.frames[1].node(OUTPUTS).values["v"] = 2
            inner.frame.add_node("base.remainder", "odd", values={"b": 2})
            links(inner.frame, "inputs.i > odd.a", "odd.out > pick.selector",
                  "pick.v > outputs.last")

        handle = start(endless(body))
        time.sleep(0.05)
        handle.cancel()
        assert handle.wait(2.0) and handle.cancelled

    def test_a_running_node_finishes_first(self, registry):
        def body(f):
            f.add_node("t.nap", "nap", values={"seconds": 0.4})
            f.add_node("base.to_float", "x")
            links(f, "inputs.n > x.x", "x.out > nap.x")

        handle = start(endless(body), registry)
        time.sleep(0.05)
        t0 = time.perf_counter()
        handle.cancel()
        assert handle.wait(3.0) and handle.cancelled
        waited = time.perf_counter() - t0
        assert 0.1 < waited < 1.5                    # the nap in progress was not interrupted
        assert len(registry.calls) == 1

    def test_stop_with_nodes_on_several_threads(self, registry):
        g = Graph()
        const(g, 1.0, "src")
        for k in range(200):
            g.add_node("t.nap", f"n{k}", values={"seconds": 0.02})
            g.connect("src", "out", f"n{k}", "x")
        handle = start(g, registry, workers=4)
        while len(registry.calls) < 4:               # wait until the first nodes are running
            time.sleep(0.001)
        handle.cancel()
        assert handle.wait(3.0) and handle.cancelled
        assert 4 <= len(registry.calls) <= 12        # they finished; almost nothing else started

    def test_stop_in_a_parallel_loop(self, registry):
        g = Graph()
        loop = g.add_for("loop", parallel=True)
        loop.values["count"] = 2000
        loop.add_output("y", mode="collect")
        loop.frame.add_node("base.to_float", "x")
        loop.frame.add_node("t.nap", "nap", values={"seconds": 0.01})
        links(loop.frame, "inputs.i > x.x", "x.out > nap.x", "nap.out > outputs.y")
        handle = start(g, registry, workers=4)
        while len(registry.calls) < 8:
            time.sleep(0.001)
        handle.cancel()
        assert handle.wait(3.0) and handle.cancelled and len(registry.calls) < 40

    def test_the_callers_own_cancel_still_counts(self):
        flag = threading.Event()
        handle = start(endless(), cancel=flag.is_set)
        time.sleep(0.02)
        flag.set()
        assert handle.wait(2.0) and handle.cancelled


class TestBackgroundRuns:
    def test_result(self, registry):
        g = Graph("sum")
        const(g, 2.0, "a")
        g.add_node("t.nap", "n", values={"seconds": 0.05})
        g.add_node("base.add", "sum", values={"b": 1.0})
        links(g, "a.out > n.x", "n.out > sum.a")
        handle = start(g, registry)
        assert not handle.done and handle.error is None
        with pytest.raises(TimeoutError):
            handle.result(timeout=0.001)
        assert handle.wait(0.001) is False
        result = handle.result(timeout=2.0)
        assert result["sum"] == 3.0 and handle.done and not handle.cancelled
        assert handle.result() is result and "finished" in repr(handle)
        handle.cancel()                              # too late: nothing happens
        assert handle.result()["sum"] == 3.0

    def test_errors_come_out_of_result(self):
        g = Graph()
        g.add_node("base.fail", "bad", values={"message": "broken"})
        handle = start(g)
        assert handle.wait(2.0) and not handle.cancelled
        assert isinstance(handle.error, NodeError) and "failed" in repr(handle)
        with pytest.raises(NodeError, match="broken"):
            handle.result()

    def test_a_graph_with_errors_does_not_start(self):
        g = Graph()
        g.add_node("base.add", "sum")
        with pytest.raises(GraphCheckError):
            start(g)

    def test_a_plan_can_be_started(self, registry):
        g = Graph()
        const(g, 2.0, "a")
        plan = ramen.compile_graph(g)
        handles = [start(plan) for _ in range(5)]
        assert [h.result(2.0)["a"] for h in handles] == [2.0] * 5
        assert handles[0].plan is plan

    def test_several_runs_at_once(self, registry):
        def body(f):
            f.add_node("t.nap", "nap", values={"seconds": 0.005})
            f.add_node("base.to_float", "x")
            links(f, "inputs.n > x.x", "x.out > nap.x")

        handles = [start(endless(body), registry) for _ in range(4)]
        time.sleep(0.05)
        handles[0].cancel()
        assert handles[0].wait(2.0)
        assert not any(h.done for h in handles[1:])  # cancelling one leaves the others alone
        for handle in handles[1:]:
            handle.cancel()
        assert all(h.wait(2.0) and h.cancelled for h in handles)
