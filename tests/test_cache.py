"""Pure nodes run once per distinct input."""

import enum
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from openramen import Cache, Graph, compile_graph, run
from openramen.library import default_registry
from openramen.model import OUTPUTS
from openramen.runtime.cache import content_token, fingerprint, function_identity, size_of

from helpers import const, links


class Colour(enum.Enum):
    RED = 1
    GREEN = 2


@pytest.fixture
def registry():
    r = default_registry()
    r.calls = []
    r.clock = [0.0]

    @r.node("t.square")
    def square(x: np.ndarray) -> np.ndarray:
        r.calls.append("square")
        return x * x

    @r.node("t.total")
    def total(x: np.ndarray) -> float:
        r.calls.append("total")
        return float(x.sum())

    @r.node("t.shift")
    def shift(x: np.ndarray, by: float = 1.0) -> np.ndarray:
        r.calls.append("shift")
        return x + by

    @r.node("t.read", pure=False)
    def read() -> float:
        r.calls.append("read")
        return r.clock[0]

    @r.node("t.read_array", pure=False)
    def read_array() -> np.ndarray:
        r.calls.append("read_array")
        return np.full(3, r.clock[0])

    @r.node("t.negate")
    def negate(x: float) -> float:
        r.calls.append("negate")
        return -x

    @r.node("t.scale", mutates=("x",))
    def scale(x: np.ndarray, factor: float = 2.0) -> np.ndarray:
        r.calls.append("scale")
        x *= factor
        return x

    return r


def pipeline(data=None):
    """data -> square -> shift -> total"""
    g = Graph("pipeline")
    const(g, np.arange(4.0) if data is None else data, "data")
    g.add_node("t.square", "square")
    g.add_node("t.shift", "shift", values={"by": 1.0})
    g.add_node("t.total", "total")
    links(g, "data.out > square.x", "square.out > shift.x", "shift.out > total.x")
    return g


class TestFingerprint:
    @pytest.mark.parametrize("a, b", [
        (1, 1), (1.5, 1.5), ("text", "text"), (None, None), (True, True), (b"ab", b"ab"),
        ([1, [2.0, "x"]], [1, [2.0, "x"]]), ({"a": 1}, {"a": 1}), ({1, 2, 3}, {3, 2, 1}),
        ((1, 2), (1, 2)), (float("nan"), float("nan")), (Path("a/b"), Path("a/b")),
        (Colour.RED, Colour.RED), (np.arange(3), np.arange(3)), (1 + 2j, 1 + 2j),
        (np.arange(6).reshape(2, 3)[:, ::2], np.array([[0, 2], [3, 5]])),
        (np.float32(1.5), np.float32(1.5)),
    ])
    def test_equal_values_have_equal_fingerprints(self, a, b):
        assert fingerprint(a) == fingerprint(b) and len(fingerprint(a)) == 16

    def test_different_values_differ(self):
        values = [0, 1, True, False, None, 0.0, 1.0, -0.0, "", "0", "1", b"", b"0", [], (), {},
                  [0], (0,), {0: 0}, {0}, [[]], [0, 1], [1, 0], "ab", ["a", "b"], ["ab"],
                  ["a", ["b"]], [["a"], "b"], 1 + 0j, Colour.RED, Colour.GREEN, Path("a"),
                  np.arange(3), np.arange(3.0), np.arange(3).reshape(3, 1),
                  np.arange(3, dtype="int32"), np.zeros(0), np.zeros((0, 2)),
                  np.float32(1.0), np.int8(1), 10, "10", 1.5, "1.5", {"a": 1}, {"a": 2},
                  {"b": 1}, [1, 2, 3], [12, 3], [1, 23]]
        prints = [fingerprint(v) for v in values]
        assert None not in prints and len(set(prints)) == len(values)

    def test_things_without_a_fingerprint(self):
        assert fingerprint(object()) is None and fingerprint([1, object()]) is None
        assert fingerprint({"k": threading.Lock()}) is None
        assert fingerprint(np.array([object()])) is None
        assert fingerprint({object()}) is None

    def test_content_tokens_only_for_cheap_values(self):
        assert content_token(3) == fingerprint(3) and content_token("ab") == fingerprint("ab")
        assert content_token(None) == fingerprint(None)
        assert content_token(np.arange(3)) is None and content_token([1]) is None
        assert content_token("x" * 1000) is None

    def test_function_identity_follows_the_code(self):
        def one(x):
            return x + 1

        def also_one(x):
            return x + 1

        def two(x):
            return x + 2

        also_one.__qualname__ = two.__qualname__ = one.__qualname__
        assert function_identity(one) == function_identity(also_one)
        assert function_identity(one) != function_identity(two)
        assert function_identity(np.sin) == function_identity(np.sin) != function_identity(np.cos)

    def test_sizes(self):
        assert size_of(np.zeros(1000)) >= 8000
        assert size_of([np.zeros(100), "abc"]) >= 800
        assert size_of({"k": np.zeros(10)}) >= 80 and size_of(5) > 0


class TestCaching:
    def test_a_second_run_computes_nothing(self, registry):
        cache = Cache()
        g = pipeline()
        first = run(g, registry, cache=cache)
        assert registry.calls == ["square", "shift", "total"] and first["total"] == 18.0
        assert (cache.hits, cache.misses) == (0, 4) and len(cache) == 4
        registry.calls.clear()
        events = []
        second = run(g, registry, cache=cache, listener=events.append)
        assert registry.calls == [] and second["total"] == 18.0
        assert np.array_equal(second["shift"], first["shift"])
        assert cache.hits == 4
        assert [e.node for e in events if e.kind == "node-cached"] == [
            "data", "square", "shift", "total"]

    def test_without_a_cache_everything_runs(self, registry):
        g = pipeline()
        run(g, registry)
        run(g, registry)
        assert registry.calls == ["square", "shift", "total"] * 2

    def test_only_what_is_downstream_of_a_change_runs_again(self, registry):
        cache = Cache()
        g = pipeline()
        run(g, registry, cache=cache)
        registry.calls.clear()
        g.node("shift").values["by"] = 2.0
        assert run(g, registry, cache=cache)["total"] == 22.0
        assert registry.calls == ["shift", "total"]
        registry.calls.clear()
        g.node("data").config["value"] = np.arange(5.0)
        assert run(g, registry, cache=cache)["total"] == 40.0
        assert registry.calls == ["square", "shift", "total"]
        registry.calls.clear()
        g.node("data").config["value"] = np.arange(4.0)      # back to something seen before
        g.node("shift").values["by"] = 1.0
        assert run(g, registry, cache=cache)["total"] == 18.0 and registry.calls == []

    def test_equal_content_is_enough(self, registry):
        cache = Cache()
        run(pipeline(np.arange(4.0)), registry, cache=cache)
        registry.calls.clear()
        run(pipeline(np.arange(4.0)), registry, cache=cache)  # a new graph, the same data
        assert registry.calls == []
        run(pipeline(np.arange(4.0, dtype="float32")), registry, cache=cache)
        assert registry.calls == ["square", "shift", "total"]

    def test_the_same_work_twice_in_one_graph_is_done_once(self, registry):
        g = pipeline()
        g.add_node("t.square", "again")
        g.connect("data", "out", "again", "x")
        result = run(g, registry, cache=Cache(), workers=1)
        assert registry.calls.count("square") == 1
        assert np.array_equal(result["again"], result["square"])

    def test_side_effects_always_run(self, registry):
        cache = Cache()
        g = Graph()
        g.add_node("t.read", "now")
        g.add_node("t.negate", "minus")
        g.connect("now", "out", "minus", "x")
        assert run(g, registry, cache=cache)["minus"] == 0.0
        assert run(g, registry, cache=cache)["minus"] == 0.0
        # the reading ran twice; it gave the same number, so what follows was reused
        assert registry.calls == ["read", "negate", "read"]
        registry.clock[0] = 5.0
        assert run(g, registry, cache=cache)["minus"] == -5.0
        assert registry.calls[3:] == ["read", "negate"]

    def test_a_new_array_from_a_side_effect_is_never_assumed_equal(self, registry):
        cache = Cache()
        g = Graph()
        g.add_node("t.read_array", "now")
        g.add_node("t.total", "sum")
        g.connect("now", "out", "sum", "x")
        run(g, registry, cache=cache)
        run(g, registry, cache=cache)
        assert registry.calls == ["read_array", "total"] * 2

    def test_targets(self, registry):
        cache = Cache()
        g = pipeline()
        run(g, registry, cache=cache, targets=["square"])
        assert registry.calls == ["square"]
        run(g, registry, cache=cache)
        assert registry.calls == ["square", "shift", "total"]

    def test_plans_and_parallel_runs(self, registry):
        cache = Cache()
        g = Graph()
        const(g, np.arange(4.0), "data")
        for k in range(6):
            g.add_node("t.shift", f"s{k}", values={"by": float(k % 3)})
            g.connect("data", "out", f"s{k}", "x")
        plan = compile_graph(g, registry)
        first = run(plan, cache=cache, workers=4)
        again = run(plan, cache=cache, workers=4)
        assert all(np.array_equal(first[f"s{k}"], again[f"s{k}"]) for k in range(6))
        assert 3 <= registry.calls.count("shift") <= 6      # three distinct pieces of work
        count = len(registry.calls)
        run(plan, cache=cache, workers=1)
        assert len(registry.calls) == count


    def test_two_outputs_of_one_node_are_two_values(self, registry):
        @registry.node("t.halves", outputs=("low", "high"))
        def halves(x: np.ndarray):
            return x[:2].copy(), x[2:].copy()

        cache = Cache()
        g = Graph()
        const(g, np.arange(4.0), "data")
        g.add_node("t.halves", "halves")
        g.add_node("t.total", "low")
        g.add_node("t.total", "high")
        links(g, "data.out > halves.x", "halves.low > low.x", "halves.high > high.x")
        for _ in range(2):
            result = run(g, registry, cache=cache, workers=1)
            assert result["low"] == 1.0 and result["high"] == 5.0


class TestSharedValues:
    def test_cached_arrays_are_read_only(self, registry):
        cache = Cache()
        result = run(pipeline(), registry, cache=cache)
        assert not result["square"].flags.writeable
        with pytest.raises(ValueError):
            result["square"][0] = 99.0
        assert run(pipeline(), registry)["square"].flags.writeable   # no cache: your own array

    def test_a_node_that_writes_still_gets_its_own_copy(self, registry):
        cache = Cache()
        g = pipeline()
        g.add_node("t.scale", "scaled", values={"factor": 10.0})
        g.connect("square", "out", "scaled", "x")
        for _ in range(2):
            result = run(g, registry, cache=cache)
            assert result["scaled"].tolist() == [0, 10, 40, 90]
            assert result["square"].tolist() == [0, 1, 4, 9] and result["total"] == 18.0
        assert registry.calls.count("scale") == 1


class TestStructures:
    def loop(self, registry, impure=False):
        g = Graph()
        const(g, [1.0, 2.0, 3.0], "items")
        loop = g.add_for("loop")
        loop.add_input("x", mode="index")
        loop.add_output("y", mode="collect")
        loop.frame.add_node("t.negate", "minus")
        links(loop.frame, "inputs.x > minus.x", "minus.out > outputs.y")
        if impure:
            loop.frame.add_node("t.read", "now")
            loop.frame.add_node("base.constant", "inner", config={"value": 1.0})
        g.connect("items", "out", loop, "x")
        return g, loop

    def test_a_pure_structure_is_cached_as_a_whole(self, registry):
        cache = Cache()
        g, loop = self.loop(registry)
        assert run(g, registry, cache=cache)["loop"] == [-1.0, -2.0, -3.0]
        assert run(g, registry, cache=cache)["loop"] == [-1.0, -2.0, -3.0]
        assert registry.calls == ["negate"] * 3
        loop.frame.add_node("t.negate", "twice")             # the loop itself changed
        loop.frame.connect("minus", "out", "twice", "x")
        loop.frame.connect("twice", "out", OUTPUTS, "y", replace=True)
        assert run(g, registry, cache=cache)["loop"] == [1.0, 2.0, 3.0]
        assert len(registry.calls) == 9

    def test_a_structure_with_side_effects_is_not(self, registry):
        cache = Cache()
        g, loop = self.loop(registry, impure=True)
        run(g, registry, cache=cache)
        run(g, registry, cache=cache)
        assert registry.calls.count("read") == 6 and registry.calls.count("negate") == 6
        assert len(cache) == 1                  # the constant outside; nothing from inside


class TestLimits:
    def test_least_recently_used_goes_first(self):
        cache = Cache(max_bytes=20_000)
        block = lambda: {"out": np.zeros(1000)}              # 8 kB each
        cache.put(b"a", block(), {"out": b"1"})
        cache.put(b"b", block(), {"out": b"2"})
        assert cache.get(b"a") is not None                   # 'a' was used after 'b'
        cache.put(b"c", block(), {"out": b"3"})
        assert cache.get(b"b") is None and cache.get(b"a") and cache.get(b"c")
        assert len(cache) == 2 and 16_000 < cache.nbytes <= 20_000
        cache.put(b"a", {"out": 1}, {"out": b"4"})           # replacing frees the old size
        assert cache.nbytes < 9_000 and cache.get(b"a")[0] == {"out": 1}
        outputs, tokens = cache.get(b"c")
        assert tokens == {"out": b"3"} and not outputs["out"].flags.writeable
        outputs["extra"] = 1                                 # the caller's own dict
        assert "extra" not in cache.get(b"c")[0]

    def test_the_budget_holds_during_runs(self, registry):
        g = pipeline(np.zeros(1000))
        cache = Cache(max_bytes=20_000)
        for _ in range(3):
            assert run(g, registry, cache=cache)["total"] == 1000.0
            assert cache.nbytes <= 20_000 and len(cache) < 4

    def test_a_dropped_value_is_computed_again_and_the_rest_still_fits(self, registry):
        cache = Cache()
        g = pipeline()
        first = run(g, registry, cache=cache)
        victim = next(key for key, entry in cache._entries.items() if "out" in entry[0]
                      and np.array_equal(entry[0]["out"], first["square"]))
        del cache._entries[victim]
        registry.calls.clear()
        run(g, registry, cache=cache)
        assert registry.calls == ["square"]                  # shift and total were found again

    def test_too_big_for_the_budget(self, registry):
        cache = Cache(max_bytes=100)
        g = pipeline(np.zeros(1000))
        run(g, registry, cache=cache)
        run(g, registry, cache=cache)
        assert registry.calls.count("square") == 2 and registry.calls.count("total") == 1

    def test_no_room_at_all(self, registry):
        for cache in (Cache(max_bytes=0), Cache(max_entries=0)):
            registry.calls.clear()
            run(pipeline(), registry, cache=cache)
            run(pipeline(), registry, cache=cache)
            assert len(cache) == 0 and registry.calls == ["square", "shift", "total"] * 2
        with pytest.raises(ValueError):
            Cache(max_bytes=-1)

    def test_entry_limit(self, registry):
        cache = Cache(max_entries=2)
        run(pipeline(), registry, cache=cache)
        assert len(cache) == 2

    def test_clear_and_repr(self, registry):
        cache = Cache()
        run(pipeline(), registry, cache=cache)
        assert "4 entries" in repr(cache) and cache.nbytes > 0
        cache.clear()
        assert len(cache) == 0 and cache.nbytes == 0
        run(pipeline(), registry, cache=cache)
        assert registry.calls == ["square", "shift", "total"] * 2


class TestWhatIsNotCached:
    def test_values_without_a_fingerprint(self, registry):
        @registry.node("t.use", config=("thing",))
        def use(thing: object = None) -> int:
            registry.calls.append("use")
            return 1

        cache = Cache()
        g = Graph()
        g.add_node("t.use", "u", config={"thing": threading.Lock()})
        g.add_node("t.negate", "n")
        g.add_node("base.to_float", "f")
        links(g, "u.out > f.x", "f.out > n.x")
        assert run(g, registry, cache=cache)["n"] == -1.0
        assert run(g, registry, cache=cache)["n"] == -1.0
        assert registry.calls == ["use", "negate", "use"]    # the number it gave is known

    def test_objects_on_wires(self, registry):
        @registry.node("t.make")
        def make() -> object:
            registry.calls.append("make")
            return threading.Lock()

        @registry.node("t.check")
        def check(lock: object) -> bool:
            registry.calls.append("check")
            return lock.locked()

        cache = Cache()
        g = Graph()
        g.add_node("t.make", "lock")
        g.add_node("t.check", "check")
        g.connect("lock", "out", "check", "lock")
        first = run(g, registry, cache=cache)
        second = run(g, registry, cache=cache)
        assert second["lock"] is first["lock"] and registry.calls == ["make", "check"]

    def test_a_replaced_function_is_not_mistaken_for_the_old_one(self, registry):
        cache = Cache()
        g = Graph()
        const(g, 3.0, "x")
        g.add_node("t.negate", "n")
        g.connect("x", "out", "n", "x")
        assert run(g, registry, cache=cache)["n"] == -3.0

        @registry.node("t.negate", replace=True)
        def negate(x: float) -> float:
            return -x - 100.0

        assert run(g, registry, cache=cache)["n"] == -103.0

    def test_nan_and_special_values(self, registry):
        cache = Cache()
        g = Graph()
        const(g, float("nan"), "x")
        g.add_node("t.negate", "n")
        g.connect("x", "out", "n", "x")
        run(g, registry, cache=cache)
        run(g, registry, cache=cache)
        assert registry.calls == ["negate"]


def test_cache_shared_by_threads(registry):
    cache = Cache(max_bytes=50_000)
    errors = []

    def work(seed):
        try:
            rng = np.random.default_rng(seed)
            for _ in range(30):
                data = np.arange(float(rng.integers(1, 6)) * 100)
                result = run(pipeline(data), registry, cache=cache)
                assert result["total"] == float((data * data + 1.0).sum())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and cache.nbytes <= 50_000 and cache.hits > 100
