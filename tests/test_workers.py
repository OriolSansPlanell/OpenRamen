"""Nodes that run in worker processes."""

import os
import threading
import time

import numpy as np
import pytest

import openramen.runtime as runtime
import process_nodes
from openramen import Graph, NodeError, compile_graph, run, start
from openramen.library import default_registry, spec_from_function
from openramen.runtime import Cancelled, workers

from helpers import const, links


@pytest.fixture(autouse=True)
def tidy():
    runtime._LOCAL_ONLY.clear()
    runtime._NEEDS_WRITABLE.clear()
    yield
    runtime._LOCAL_ONLY.clear()
    runtime._NEEDS_WRITABLE.clear()


@pytest.fixture
def small(monkeypatch):
    """Send even small arrays through shared memory."""
    monkeypatch.setattr(workers, "SHARE_FROM", 64)


@pytest.fixture(scope="module")
def registry():
    r = default_registry()
    options = {
        "fill": {"passthrough": ("a",)},
        "split": {"outputs": ("first", "second"),
                  "types": {"first": "Array[float64, 1]", "second": "Array[float64, 1]"}},
        "crash": {"pure": False},
    }
    for name in ("count_primes", "pid", "double", "describe", "fill", "every_other",
                 "reversed_rows", "sneaky", "split", "bundle", "total_of", "crash", "nap",
                 "fail", "fail_awkwardly", "make_lock", "identity"):
        fn = getattr(process_nodes, name)
        r.register(spec_from_function(fn, f"p.{name}", executor="process",
                                      **options.get(name, {})))
        r.register(spec_from_function(fn, f"t.{name}", **options.get(name, {})))
    return r


def one(registry, spec, data=None, pin="a", **literals):
    g = Graph()
    g.add_node(spec, "n", values=literals)
    if data is not None:
        const(g, data, "src")
        g.connect("src", "out", "n", pin)
    return g


def part(a: np.ndarray) -> np.ndarray:
    return a[100:900:2]


def blocks():
    import gc
    gc.collect()
    workers._sweep()
    return workers.shared_blocks()


class TestProcessNodes:
    def test_runs_in_another_process(self, registry):
        result = run(one(registry, "p.pid"), registry)
        assert result["n"] != os.getpid()
        assert run(one(registry, "t.pid"), registry)["n"] == os.getpid()

    def test_same_values_as_in_this_process(self, registry):
        for spec, literals in (("count_primes", {"limit": 2000}), ("nap", {"seconds": 0.0})):
            remote = run(one(registry, f"p.{spec}", **literals), registry, check_outputs=True)
            local = run(one(registry, f"t.{spec}", **literals), registry)
            assert remote["n"] == local["n"]

    def test_plan_marks_them(self, registry):
        plan = compile_graph(one(registry, "p.pid"), registry)
        assert plan.steps[0].remote and plan.steps[0].heavy
        assert not compile_graph(one(registry, "t.pid"), registry).steps[0].remote

    def test_two_pure_python_branches_use_two_processes(self, registry):
        workers.start(2)
        g = Graph()
        for k in range(2):
            g.add_node("p.count_primes", f"n{k}", values={"limit": 250_000})
            g.add_node("p.pid", f"pid{k}")
        t0 = time.perf_counter()
        result = run(g, registry, targets=["n0", "n1"], workers=2)
        parallel = time.perf_counter() - t0
        t0 = time.perf_counter()
        expected = process_nodes.count_primes(250_000)
        one_alone = time.perf_counter() - t0
        assert result["n0"] == result["n1"] == expected
        if (os.cpu_count() or 1) >= 2:
            assert parallel < one_alone * 1.85         # two of them, in under twice the time

    def test_workers_are_kept(self, registry):
        workers.start(1)
        seen = {run(one(registry, "p.pid"), registry)["n"] for _ in range(5)}
        assert len(seen) <= workers.limit() and len(workers._LANES) <= workers.limit()

    def test_a_worker_is_sent_each_node_once(self, registry, monkeypatch):
        import pickle
        monkeypatch.setattr(workers, "limit", lambda: 1)
        workers.shutdown()
        sent = []
        exchange = workers._Lane.exchange

        def recording(lane, message, cancel):
            request = pickle.loads(message)
            if request[0] == "call":
                sent.append(request[2] is not None)        # was the node itself included?
            return exchange(lane, message, cancel)

        monkeypatch.setattr(workers._Lane, "exchange", recording)
        plan = compile_graph(one(registry, "p.count_primes", limit=50), registry)
        assert [run(plan)["n"] for _ in range(3)] == [15, 15, 15]
        assert sent == [True, False, False]
        workers.shutdown()

    def test_errors_come_back(self, registry):
        with pytest.raises(NodeError, match="ValueError: no good") as info:
            run(one(registry, "p.fail", message="no good"), registry)
        assert isinstance(info.value.original, ValueError) and info.value.node == "n"
        cause = info.value.__cause__
        assert isinstance(cause, workers.RemoteError) and "process_nodes.py" in \
            cause.remote_traceback

    def test_errors_that_cannot_be_rebuilt(self, registry):
        with pytest.raises(NodeError, match="Awkward: 1 and 2") as info:
            run(one(registry, "p.fail_awkwardly"), registry)
        assert isinstance(info.value.original, RuntimeError)
        assert run(one(registry, "p.pid"), registry)["n"] > 0      # the worker is fine

    def test_a_crash_takes_only_the_worker_down(self, registry):
        with pytest.raises(NodeError, match="worker process died.*exit code 3") as info:
            run(one(registry, "p.crash"), registry)
        assert isinstance(info.value.original, workers.WorkerDied)
        assert run(one(registry, "p.count_primes", limit=100), registry)["n"] == 25

    def test_results_that_cannot_be_sent_back(self, registry):
        registry.register(spec_from_function(process_nodes.identity, "p.any_lock",
                                             executor="process", types={"x": "Any"}),
                          replace=True)
        registry.register(spec_from_function(process_nodes.make_lock, "p.hidden_lock",
                                             executor="process", types={"out": "Any"}),
                          replace=True)
        with pytest.raises(NodeError, match="cannot be sent back.*executor 'thread'"):
            run(one(registry, "p.hidden_lock"), registry)

    def test_nodes_with_object_pins_stay_here(self, registry):
        registry.register(spec_from_function(
            process_nodes.make_lock, "p.typed_lock", executor="process",
            types={"out": "Object[threading:Thread]"}), replace=True)
        plan = compile_graph(one(registry, "p.typed_lock"), registry)
        assert not plan.steps[0].remote                # declared to return an object
        lock = run(plan)["n"]
        assert hasattr(lock, "acquire")

    def test_arguments_that_cannot_be_sent(self, registry):
        registry.register(spec_from_function(process_nodes.identity, "p.any",
                                             executor="process", types={"x": "Any"}),
                          replace=True)
        registry.node("t.lock", replace=True)(lambda: threading.Lock())
        g = Graph()
        g.add_node("t.lock", "lock")
        g.add_node("p.any", "n")
        g.connect("lock", "out", "n", "x")
        events = []
        result = run(g, registry, listener=events.append)
        assert result["n"] is result["lock"]           # it ran here, on the real object
        warnings = [e.message for e in events if e.kind == "node-warning"]
        assert len(warnings) == 1 and "runs in this process, not in a worker" in warnings[0]
        events.clear()
        run(g, registry, listener=events.append)       # remembered: no second attempt
        assert not [e for e in events if e.kind == "node-warning"]

    def test_functions_a_worker_cannot_import(self, registry):
        def local_function(x: int) -> int:
            return x + 1

        registry.register(spec_from_function(local_function, "p.local", executor="process"),
                          replace=True)
        events = []
        result = run(one(registry, "p.local", x=1), registry, listener=events.append)
        assert result["n"] == 2
        assert any(e.kind == "node-warning" and "cannot be sent" in e.message for e in events)


class TestSharedMemory:
    def test_arrays_arrive_and_come_back(self, registry, small):
        data = np.arange(1000.0)
        result = run(one(registry, "p.double", data), registry, check_outputs=True)
        assert np.array_equal(result["n"], data * 2.0)
        assert not result["n"].flags.writeable
        assert workers._segment_of(result["n"]) is not None      # it lives in a shared block
        assert data.sum() == np.arange(1000.0).sum()

    @pytest.mark.parametrize("array", [
        np.arange(24.0).reshape(2, 3, 4), np.arange(100, dtype="uint8"),
        np.arange(60.0).reshape(6, 10)[::2, ::3], np.arange(50.0)[::-1],
        np.asfortranarray(np.arange(30.0).reshape(5, 6)), np.zeros((0, 4)),
        np.array(3.5), np.arange(10) > 4, np.arange(8, dtype="complex64"),
        np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "i4"), ("b", "f8")]),
    ])
    def test_every_kind_of_array(self, registry, small, array):
        registry.register(spec_from_function(process_nodes.identity, "p.same",
                                             executor="process",
                                             types={"x": "Array", "out": "Array"}),
                          replace=True)
        result = run(one(registry, "p.same", array, pin="x"), registry)["n"]
        assert result.dtype == array.dtype and result.shape == array.shape
        assert np.array_equal(result, array)

    def test_small_arrays_take_the_pipe(self, registry):
        result = run(one(registry, "p.double", np.arange(10.0)), registry)["n"]
        assert workers._segment_of(result) is None and result.tolist()[:3] == [0, 2, 4]

    def test_a_result_goes_into_the_next_worker_without_a_copy(self, registry, small,
                                                               monkeypatch):
        g = one(registry, "p.double", np.arange(1000.0))
        g.add_node("p.double", "again")
        g.add_node("p.describe", "look")
        links(g, "n.out > again.a", "again.out > look.a")
        created = []
        original = workers._create
        monkeypatch.setattr(workers, "_create", lambda n: created.append(n) or original(n))
        result = run(g, registry)
        assert np.array_equal(result["again"], np.arange(1000.0) * 4.0)
        assert result["look"]["sum"] == float(np.arange(1000.0).sum() * 4)
        assert result["look"]["writable"] is False      # shared, so the worker may only read
        # one block for the constant on its way in, one per result; nothing for the hand-overs
        assert created == [8000, 8000, 8000]

    @pytest.fixture
    def created(self, monkeypatch):
        """The sizes of the shared blocks made during the test."""
        made = []
        original = workers._create
        monkeypatch.setattr(workers, "_create", lambda n: made.append(n) or original(n))
        return made

    def test_in_place_functions(self, registry, small, created):
        data = np.zeros(500)
        g = one(registry, "p.fill", data, value=4.0)
        g.add_node("base.identity", "witness")
        g.connect("src", "out", "witness", "value")
        result = run(g, registry, check_outputs=True)
        assert result["n"].tolist() == [4.0] * 500 and result["witness"].sum() == 0
        assert data.sum() == 0
        assert created == [4000]                       # written where it was sent: one block

    def test_views_of_the_input_come_back_as_views(self, registry, small, created):
        data = np.arange(1000.0)
        result = run(one(registry, "p.every_other", data), registry)["n"]
        assert np.array_equal(result, data[::2]) and created == [8000]
        grid = np.arange(600.0).reshape(20, 30)
        result = run(one(registry, "p.reversed_rows", grid), registry)["n"]
        assert np.array_equal(result, grid[::-1, ::-1]) and created == [8000, 4800]

    def test_views_of_results_can_be_sent_on(self, registry, small):
        g = one(registry, "p.double", np.arange(1000.0))
        registry.node("t.part", replace=True)(part)
        g.add_node("t.part", "part")
        g.add_node("p.describe", "look")
        links(g, "n.out > part.a", "part.out > look.a")
        result = run(g, registry)
        assert result["look"]["sum"] == float((np.arange(1000.0) * 2)[100:900:2].sum())
        assert result["look"]["shape"] == [400]
        assert np.shares_memory(result["part"], result["n"])

    def test_an_undeclared_write_is_contained(self, registry, small):
        data = np.ones(400)
        g = one(registry, "p.double", data)
        g.add_node("p.sneaky", "bad")
        g.add_node("p.describe", "witness")
        links(g, "n.out > bad.a", "n.out > witness.a")
        events = []
        result = run(g, registry, listener=events.append, workers=1)
        assert result["bad"] == 2.0 * 400 - 2.0 - 1.0
        assert result["witness"]["sum"] == 800.0 and result["n"].sum() == 800.0
        assert any(e.kind == "node-warning" and "needs writable arrays" in e.message
                   for e in events)
        assert "p.sneaky" in runtime._NEEDS_WRITABLE

    def test_arrays_inside_containers(self, registry, small):
        data = np.arange(300.0)
        g = one(registry, "p.bundle", data)
        g.add_node("p.split", "halves")
        g.connect("src", "out", "halves", "a")
        result = run(g, registry)
        first, second, third = result["n"]
        assert np.array_equal(first, data + 1.0) and np.array_equal(second["twice"], data * 2)
        assert second["label"] == "x" and third == ((300,), 3)
        assert np.array_equal(result["halves", "first"], data[:150])
        assert np.array_equal(result["halves", "second"], data[150:])
        g2 = Graph()
        const(g2, [np.ones(200), np.full(100, 2.0), 5.0], "items")
        g2.add_node("p.total_of", "sum")
        g2.connect("items", "out", "sum", "items")
        assert run(g2, registry)["sum"] == 405.0

    def test_blocks_are_freed(self, registry, small):
        before = blocks()
        result = run(one(registry, "p.double", np.arange(5000.0)), registry)
        assert blocks() == before + 1                  # the result is still alive
        kept = result["n"][10:20]                      # a view keeps the block
        del result
        assert blocks() == before + 1 and kept[0] == 20.0
        del kept
        assert blocks() == before

    def test_nothing_is_left_behind_after_failures(self, registry, small):
        before = blocks()
        for spec in ("p.fail", "p.crash"):
            g = Graph()
            const(g, np.arange(3000.0), "src")
            g.add_node("p.double", "ok")
            g.add_node(spec, "bad", values={"message": "x"} if spec == "p.fail" else {})
            g.add_node("base.to_text", "after")
            links(g, "src.out > ok.a", "bad.out > after.x")
            with pytest.raises(NodeError):
                run(g, registry)
        assert blocks() == before

    def test_many_calls_from_many_threads(self, registry, small):
        before = blocks()
        g = Graph()
        rng = np.random.default_rng(0)
        expected = {}
        for k in range(12):
            data = rng.random(200 + 50 * k)
            const(g, data, f"d{k}")
            g.add_node("p.double", f"n{k}")
            g.connect(f"d{k}", "out", f"n{k}", "a")
            expected[f"n{k}"] = data * 2.0
        result = run(g, registry, workers=6)
        for name, value in expected.items():
            assert np.array_equal(result[name], value)
        del result
        assert blocks() == before


class TestStop:
    def test_stop_kills_a_running_process_node(self, registry):
        workers.start(1)
        handle = start(one(registry, "p.nap", seconds=30.0), registry)
        time.sleep(0.3)
        assert not handle.done
        t0 = time.perf_counter()
        handle.cancel()
        assert handle.wait(3.0) and handle.cancelled
        assert time.perf_counter() - t0 < 2.0          # it did not wait for the nap to end
        assert run(one(registry, "p.pid"), registry)["n"] > 0

    def test_cancel_before_a_worker_is_free(self, registry, monkeypatch):
        monkeypatch.setattr(workers, "limit", lambda: 1)
        workers.shutdown()
        g = Graph()
        for k in range(3):
            g.add_node("p.nap", f"n{k}", values={"seconds": 5.0})
        started = time.perf_counter()
        with pytest.raises(Cancelled):
            run(g, registry, cancel=lambda: time.perf_counter() - started > 0.5, workers=3)
        assert time.perf_counter() - started < 3.0
        workers.shutdown()


class TestInLoops:
    def test_a_process_node_in_a_parallel_loop(self, registry, small):
        g = Graph()
        const(g, np.arange(12.0).reshape(4, 3).repeat(100, axis=1), "rows")
        loop = g.add_for("loop", parallel=True)
        loop.add_input("row", mode="index")
        loop.add_output("out", mode="stack")
        loop.frame.add_node("p.double", "n")
        links(loop.frame, "inputs.row > n.a", "n.out > outputs.out")
        g.connect("rows", "out", loop, "row")
        result = run(g, registry, check_outputs=True)
        assert np.array_equal(result["loop"], result["rows"] * 2.0)

    def test_a_shift_register_through_a_worker(self, registry, small):
        g = Graph()
        const(g, np.ones(300), "start")
        loop = g.add_for("loop")
        loop.values["count"] = 5
        loop.add_register("acc")
        loop.frame.add_node("p.double", "n")
        links(loop.frame, "inputs.acc > n.a", "n.out > outputs.acc")
        g.connect("start", "out", loop, "acc")
        before = blocks()
        result = run(g, registry, check_outputs=True)
        assert result["loop"].tolist() == [32.0] * 300 and result["start"].sum() == 300
        del result
        assert blocks() == before

    def test_in_place_register_through_a_worker(self, registry, small):
        g = Graph()
        const(g, np.zeros(300), "start")
        loop = g.add_for("loop")
        loop.values["count"] = 3
        loop.add_register("buffer")
        loop.frame.add_node("p.fill", "n")
        loop.frame.add_node("base.to_float", "v")
        links(loop.frame, "inputs.buffer > n.a", "inputs.i > v.x", "v.out > n.value",
              "n.a > outputs.buffer")
        g.connect("start", "out", loop, "buffer")
        result = run(g, registry, check_outputs=True)
        assert result["loop"].tolist() == [2.0] * 300 and result["start"].sum() == 0
