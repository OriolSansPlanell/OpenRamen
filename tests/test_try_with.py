"""The Try and With structures."""

import contextlib
import threading

import pytest

from openramen import Graph, GraphError, NodeError, analyze, compile_graph, run
from openramen import types as T
from openramen.check import entered_type, interface
from openramen.library import default_registry
from openramen.model import OUTPUTS, UNSET

from helpers import const, links


def safe_divide(a, b, fallback=None):
    """try: q = a // b  except: q = -1 (or the fallback)."""
    g = Graph()
    const(g, a, "a")
    const(g, b, "b")
    attempt = g.add_try("attempt")
    attempt.add_input("a")
    attempt.add_input("b")
    attempt.add_output("q") if fallback is None else attempt.add_output("q", fallback=fallback)
    body, handler = attempt.frames
    body.add_node("base.floor_divide", "div")
    links(body, "inputs.a > div.a", "inputs.b > div.b", "div.out > outputs.q")
    if fallback is None:
        handler.node(OUTPUTS).values["q"] = -1
    links(g, "a.out > attempt.a", "b.out > attempt.b")
    return g, attempt


class TestTryModel:
    def test_two_frames(self):
        g = Graph()
        attempt = g.add_try()
        assert attempt.id == "try_1" and attempt.spec == "structure.try"
        assert [f.name for f in attempt.frames] == ["try_1[0]", "try_1[1]"]
        assert attempt.reserved == ("i", "count", "stop", "selector",
                                    "error", "error_type", "failed")
        for name in ("error", "error_type", "failed"):
            with pytest.raises(GraphError, match="uses itself"):
                attempt.add_output(name)
        g.add_for().add_input("error")                    # only reserved in a Try
        with pytest.raises(GraphError, match="only loops"):
            attempt.add_register("r")
        with pytest.raises(GraphError, match="only loops"):
            attempt.add_output("y", mode="collect")

    def test_interface(self):
        g, attempt = safe_divide(1, 1)
        spec = interface(attempt)
        assert [p.name for p in spec.inputs] == ["a", "b"]
        assert [(p.name, str(p.type)) for p in spec.outputs] == [
            ("q", "Any"), ("failed", "Bool"), ("error", "Str")]

    def test_file_format(self):
        g, attempt = safe_divide(7, 0)
        text = g.dumps()
        loaded = Graph.loads(text)
        assert loaded == g and loaded.dumps() == text and len(loaded.node("attempt").frames) == 2
        assert run(loaded)["attempt"]["q"] == -1
        data = g.to_dict()
        next(n for n in data["nodes"] if n["id"] == "attempt")["structure"]["frames"].pop()
        with pytest.raises(GraphError, match="exactly two frames"):
            Graph.from_dict(data)

    def test_flaws(self):
        g, attempt = safe_divide(1, 1)
        attempt.frames.pop()
        problems = analyze(g).problems
        assert [p.code for p in problems] == ["bad-frame"]
        assert "exactly two frames, not 1" in problems[0].message


class TestTry:
    @pytest.mark.parametrize("a, b", [(7, 2), (7, 0), (0, 5), (-9, 4)])
    def test_matches_python(self, a, b):
        try:
            q, failed = a // b, False
        except ZeroDivisionError:
            q, failed = -1, True
        g, _ = safe_divide(a, b)
        result = run(g, check_outputs=True)["attempt"]
        assert result["q"] == q and result["failed"] is failed
        assert bool(result["error"]) is failed

    def test_the_error_reaches_the_handler(self):
        g, attempt = safe_divide(7, 0)
        attempt.add_output("message")
        attempt.add_output("kind")
        body, handler = attempt.frames
        body.node(OUTPUTS).values.update(message="fine", kind="none")
        links(handler, "inputs.error > outputs.message", "inputs.error_type > outputs.kind")
        result = run(g, check_outputs=True)["attempt"]
        assert result["kind"] == "ZeroDivisionError" and result["failed"] is True
        assert result["message"] == result["error"]
        assert "node 'attempt[0]/div' (base.floor_divide)" in result["error"]
        assert "in the body of 'attempt'" in result["error"]

    def test_types(self):
        g, attempt = safe_divide(7, 2)
        analysis = analyze(g)
        assert analysis.ok
        assert {k: str(v) for k, v in analysis.nodes["attempt"].out_types.items()} == {
            "q": "Int", "failed": "Bool", "error": "Str"}
        attempt.frames[1].node(OUTPUTS).values["q"] = 0.5          # Int joins Float
        assert str(analyze(g).out_type("attempt", "q")) == "Float"
        value = run(g, check_outputs=True)["attempt"]["q"]
        assert value == 3.0 and type(value) is float
        attempt.frames[1].node(OUTPUTS).values["q"] = "none"
        assert [p.code for p in analyze(g).problems] == ["tunnel-conflict"]

    def test_handler_inputs_have_types(self):
        g, attempt = safe_divide(7, 2)
        handler = analyze(g).nodes["attempt"].frames[1]
        assert handler.nodes["inputs"].out_types == {
            "a": T.INT, "b": T.INT, "error": T.STR, "error_type": T.STR}
        body = analyze(g).nodes["attempt"].frames[0]
        assert "error" not in body.nodes["inputs"].out_types
        attempt.frames[0].add_node("base.text_length", "n")
        attempt.frames[0].connect("inputs", "error", "n", "text")
        assert [(p.code, p.node) for p in analyze(g).problems] == [
            ("unknown-pin", "attempt[0]/inputs")]

    def test_fallback_instead_of_a_handler(self):
        g, attempt = safe_divide(7, 0, fallback=0)
        assert analyze(g).ok
        assert run(g)["attempt"]["q"] == 0
        g.node("b").config["value"] = 7
        assert run(g)["attempt"]["q"] == 1

    def test_outputs_must_come_from_both_frames(self):
        g, attempt = safe_divide(7, 0)
        del attempt.frames[1].node(OUTPUTS).values["q"]
        problems = analyze(g).problems
        assert [p.code for p in problems] == ["unwired-tunnel"]
        assert "frame 1" in problems[0].message

    def test_a_failing_handler_fails_the_structure(self):
        g, attempt = safe_divide(7, 0)
        handler = attempt.frames[1]
        del handler.node(OUTPUTS).values["q"]
        handler.add_node("base.fail", "give_up", values={"message": "no way"})
        handler.add_node("base.constant", "zero", config={"value": 0})
        handler.connect("zero", "out", OUTPUTS, "q")
        with pytest.raises(NodeError, match="no way") as info:
            run(g)
        assert info.value.node == "attempt[1]/give_up"
        assert info.value.context == ["in the error handler of 'attempt'"]

    def test_nothing_after_the_failure_runs_in_the_body(self):
        registry = default_registry()
        seen = []

        @registry.node("t.note", pure=False)
        def note(x: int) -> int:
            seen.append(x)
            return x

        g, attempt = safe_divide(7, 0)
        body = attempt.frames[0]
        body.add_node("t.note", "before")
        body.add_node("t.note", "after")
        links(body, "inputs.a > before.x", "div.out > after.x")
        assert run(g, registry, workers=1)["attempt"]["failed"] is True and seen == [7]
        seen.clear()
        assert run(g, registry, workers=4)["attempt"]["failed"] is True
        assert seen in ([], [7])       # 'before' may not have started yet; 'after' never runs

    def test_try_inside_a_loop_and_a_loop_inside_try(self):
        g = Graph()
        const(g, [4, 0, 2, 0, 1], "divisors")
        loop = g.add_for("loop")
        loop.add_input("d", mode="index")
        loop.add_output("q", mode="collect")
        loop.add_register("failures")
        loop.values["failures"] = 0
        f = loop.frame
        attempt = f.add_try("attempt")
        attempt.add_input("d")
        attempt.add_output("q")
        body, handler = attempt.frames
        inner = body.add_for("twice")
        inner.values["count"] = 2
        inner.add_input("d")
        inner.add_register("acc")
        inner.values["acc"] = 100
        inner.frame.add_node("base.floor_divide", "div")
        links(inner.frame, "inputs.acc > div.a", "inputs.d > div.b", "div.out > outputs.acc")
        links(body, "inputs.d > twice.d", "twice.acc > outputs.q")
        handler.node(OUTPUTS).values["q"] = -1
        f.add_node("base.to_int", "one")
        f.add_node("base.add", "count")
        links(f, "inputs.d > attempt.d", "attempt.q > outputs.q", "attempt.failed > one.x",
              "inputs.failures > count.a", "one.out > count.b", "count.out > outputs.failures")
        g.connect("divisors", "out", loop, "d")
        expected, failures = [], 0
        for d in [4, 0, 2, 0, 1]:
            try:
                acc = 100
                for _ in range(2):
                    acc = acc // d
                expected.append(acc)
            except ZeroDivisionError:
                expected.append(-1)
                failures += 1
        for workers in (1, 4):
            result = run(g, check_outputs=True, workers=workers)["loop"]
            assert result == {"q": expected, "failures": failures}

    def test_cancel_is_not_caught(self):
        from openramen.runtime import Cancelled
        g, attempt = safe_divide(7, 1)
        with pytest.raises(Cancelled):
            run(g, cancel=lambda: True)

    def test_purity(self):
        g, attempt = safe_divide(7, 1)
        assert compile_graph(g).pure


class Resource:
    """A context manager that records what happens to it."""

    def __init__(self, enters="self", swallow=False, fail_on=""):
        self.events = []
        self.enters = enters
        self.swallow = swallow
        self.fail_on = fail_on

    def __enter__(self):
        if self.fail_on == "enter":
            raise OSError("cannot open")
        self.events.append("enter")
        return self if self.enters == "self" else self.enters

    def __exit__(self, kind, error, traceback):
        self.events.append(("exit", None if kind is None else kind.__name__))
        if self.fail_on == "exit":
            raise OSError("cannot close")
        return self.swallow


@pytest.fixture
def registry():
    r = default_registry()
    made = []
    r.made = made

    @r.node("t.resource", pure=False, config=("enters", "swallow", "fail_on"))
    def resource(enters: object = "self", swallow: bool = False, fail_on: str = "") -> Resource:
        made.append(Resource(enters, swallow, fail_on))
        return made[-1]

    @r.node("t.use", pure=False)
    def use(resource: object, value: int = 1) -> int:
        events = getattr(resource, "events", None)
        if events is not None:
            events.append(("use", value))
        return value * 2

    @r.node("t.lock")
    def lock() -> object:
        return threading.Lock()

    @r.node("t.locked")
    def locked(lock: object) -> bool:
        return lock.locked()

    return r


def with_graph(registry, **config):
    g = Graph()
    g.add_node("t.resource", "res", config=config)
    w = g.add_with("w")
    w.add_input("value")
    w.add_output("y")
    w.values["value"] = 21
    w.frame.add_node("t.use", "use")
    links(w.frame, "inputs.resource > use.resource", "inputs.value > use.value",
          "use.out > outputs.y")
    g.connect("res", "out", w, "resource")
    return g, w


class TestWith:
    def test_enter_run_exit(self, registry):
        g, w = with_graph(registry)
        assert analyze(g, registry).ok
        assert run(g, registry, check_outputs=True)["w"] == 42
        assert registry.made[0].events == ["enter", ("use", 21), ("exit", None)]

    def test_the_frame_gets_what_enter_returns(self, registry):
        g, w = with_graph(registry, enters=5)
        w.add_output("entered")
        w.frame.connect("inputs", "resource", OUTPUTS, "entered")
        assert run(g, registry)["w"] == {"y": 42, "entered": 5}

    def test_exit_runs_when_the_frame_fails(self, registry):
        g, w = with_graph(registry)
        w.frame.add_node("base.fail", "bad", values={"message": "inside"})
        w.frame.add_node("t.use", "never")
        links(w.frame, "use.out > bad.message" if False else "inputs.resource > never.resource")
        w.frame.connect("use", "out", "never", "value")
        with pytest.raises(NodeError, match="inside") as info:
            run(g, registry, workers=1)
        assert info.value.context == ["in 'w'"]
        events = registry.made[0].events
        assert events[0] == "enter" and events[-1] == ("exit", "RuntimeError")

    def test_a_context_manager_may_swallow_the_error(self, registry):
        g, w = with_graph(registry, swallow=True)
        w.frame.add_node("base.fail", "bad", values={"message": "inside"})
        w.outputs[0].fallback = -1
        assert run(g, registry)["w"] == -1
        assert registry.made[0].events[-1] == ("exit", "RuntimeError")
        w.outputs[0].fallback = UNSET
        with pytest.raises(NodeError, match="output 'y' received no value"):
            run(g, registry)                              # nothing to give, no fallback

    @pytest.mark.parametrize("fail_on, message", [("enter", "cannot open"),
                                                  ("exit", "cannot close")])
    def test_failing_to_open_or_close(self, registry, fail_on, message):
        g, w = with_graph(registry, fail_on=fail_on)
        with pytest.raises(NodeError, match=message) as info:
            run(g, registry)
        assert info.value.node == "w" and info.value.pin == "resource"
        assert isinstance(info.value.original, OSError)

    def test_not_a_context_manager(self, registry):
        g, w = with_graph(registry)
        g.remove_node("res")
        w.values["resource"] = 5
        problems = analyze(g, registry).problems
        assert [(p.code, p.pin) for p in problems] == [("type-mismatch", "resource")]
        assert "must be a context manager, not Int" in problems[0].message
        registry.node("t.anything")(lambda: 5)
        del w.values["resource"]
        g.add_node("t.anything", "res")
        g.connect("res", "out", w, "resource")
        assert analyze(g, registry).ok                    # Any: checked when it runs
        with pytest.raises(NodeError, match="'resource' is a Int, not a context manager"):
            run(g, registry)

    def test_resource_is_required_and_reserved(self, registry):
        g = Graph()
        w = g.add_with("w")
        assert [(p.code, p.pin) for p in analyze(g).problems] == [("missing-input", "resource")]
        with pytest.raises(GraphError, match="uses itself"):
            w.add_input("resource")
        with pytest.raises(GraphError, match="only loops"):
            w.add_register("r")

    def test_real_context_managers(self, registry):
        g = Graph()
        g.add_node("t.lock", "lock")
        w = g.add_with("w")
        w.add_input("lock")
        w.add_output("held")
        w.frame.add_node("t.locked", "check")
        links(w.frame, "inputs.lock > check.lock", "check.out > outputs.held")
        links(g, "lock.out > w.resource", "lock.out > w.lock")
        g.add_node("t.locked", "after")
        g.connect("lock", "out", "after", "lock")
        result = run(g, registry, targets=["w"])
        assert result["w"] is True and not result["lock"].locked()

    def test_a_file(self, registry, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("first line\nsecond line\n")

        @registry.node("t.open", pure=False)
        def open_file(path: str) -> object:
            return open(path)

        @registry.node("t.first_line", pure=False)
        def first_line(file: object) -> str:
            return file.readline().strip()

        g = Graph()
        g.add_node("t.open", "file", values={"path": str(path)})
        w = g.add_with("reading")
        w.add_output("line")
        w.frame.add_node("t.first_line", "read")
        links(w.frame, "inputs.resource > read.file", "read.out > outputs.line")
        g.connect("file", "out", w, "resource")
        result = run(g, registry)
        assert result["reading"] == "first line" and result["file"].closed

    def test_file_format_and_purity(self, registry):
        g, w = with_graph(registry)
        text = g.dumps()
        assert Graph.loads(text) == g and Graph.loads(text).dumps() == text
        plan = compile_graph(g, registry)
        assert not plan.steps[-1].pure                    # opening and closing is a side effect

    def test_with_inside_try(self, registry):
        g = Graph()
        g.add_node("t.resource", "res")
        attempt = g.add_try("attempt")
        attempt.add_input("res")
        attempt.add_output("y", fallback=-1)
        w = attempt.frames[0].add_with("w")
        w.add_output("y")
        w.frame.add_node("base.fail", "bad", values={"message": "inside"})
        w.frame.node(OUTPUTS).values["y"] = 1
        links(attempt.frames[0], "inputs.res > w.resource", "w.y > outputs.y")
        g.connect("res", "out", attempt, "res")
        result = run(g, registry)["attempt"]
        assert result["y"] == -1 and result["failed"] and "inside" in result["error"]
        assert registry.made[0].events == ["enter", ("exit", "RuntimeError")]


class TestEnteredType:
    def test_by_type(self):
        assert entered_type(T.ANY) == T.ANY
        assert entered_type(T.INT) is None and entered_type(T.parse_type("List[Int]")) is None
        assert entered_type(T.parse_type("Array[float64, 1]")) is None
        assert entered_type(T.parse_type("Optional[Int]")) == T.ANY      # checked at run time
        lock = T.from_annotation(type(threading.Lock()))
        assert entered_type(lock) == T.ANY                 # no annotation on __enter__
        assert entered_type(T.ObjectT("nowhere:Missing")) == T.ANY
        assert entered_type(T.from_annotation(Resource)) == T.ANY
        assert entered_type(T.from_annotation(complex)) is None
        assert entered_type(T.from_annotation(object)) == T.ANY

    def test_annotated_enter(self):
        class Typed(contextlib.AbstractContextManager):
            def __enter__(self) -> int:
                return 1

            def __exit__(self, *args):
                return None

        class Plain:
            pass

        globals()["Typed"], globals()["Plain"] = Typed, Plain
        Typed.__qualname__, Plain.__qualname__ = "Typed", "Plain"
        assert entered_type(T.from_annotation(Typed)) == T.INT
        assert entered_type(T.from_annotation(Plain)) is None
