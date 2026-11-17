"""Running structures: every program here is compared with plain Python."""

import enum
import math

import numpy as np
import pytest

import openramen as ramen
from openramen import Graph, NodeError, analyze, compile_graph, run
from openramen.library import default_registry
from openramen.model import OUTPUTS
from openramen.runtime import Cancelled, _same

from helpers import const, links


class Colour(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def checked(graph, registry=None, **options):
    """Run with every output verified against its static type, twice: as built
    and after a trip through the file format."""
    first = run(graph, registry, check_outputs=True, **options)
    again = run(Graph.loads(graph.dumps()), registry, check_outputs=True, **options)
    for key, value in first.values.items():
        assert _same(value, again.values[key]), key
    return first


def for_loop(source=None, mode="index", count=None):
    g = Graph()
    loop = g.add_for("loop")
    if source is not None:
        const(g, source, "src")
        loop.add_input("x", mode=mode)
        g.connect("src", "out", loop, "x")
    if count is not None:
        loop.values["count"] = count
    return g, loop, loop.frame


class TestForLoop:
    @pytest.mark.parametrize("values", [[1, 2, 3], [0.5, -1.5], [7], list(range(50))])
    def test_sum_of_squares(self, values):
        g, loop, f = for_loop(values)
        loop.add_register("total")
        loop.values["total"] = 0
        loop.add_output("squares", mode="collect")
        f.add_node("base.multiply", "square")
        f.add_node("base.add", "add")
        links(f, "inputs.x > square.a", "inputs.x > square.b", "inputs.total > add.a",
              "square.out > add.b", "add.out > outputs.total", "square.out > outputs.squares")
        total, squares = 0, []
        for x in values:
            squares.append(x * x)
            total = total + x * x
        result = checked(g)["loop"]
        assert result == {"squares": squares, "total": total}
        assert type(result["total"]) is type(total + 0.0 if isinstance(values[0], float) else total)

    @pytest.mark.parametrize("count", [0, 1, 5])
    def test_count_and_iteration_number(self, count):
        g, loop, f = for_loop(count=count)
        loop.add_output("numbers", mode="collect")
        loop.add_output("array", mode="stack")
        loop.add_output("last", fallback=-1)
        for name in ("numbers", "array", "last"):
            f.connect("inputs", "i", OUTPUTS, name)
        result = checked(g)["loop"]
        assert result["numbers"] == list(range(count))
        assert result["array"].tolist() == list(range(count)) and result["array"].dtype == "int64"
        assert result["last"] == (count - 1 if count else -1)

    def test_count_from_a_wire_and_negative_count(self):
        g, loop, f = for_loop()
        const(g, -3, "n")
        g.connect("n", "out", loop, "count")
        loop.add_output("numbers", mode="collect")
        f.connect("inputs", "i", OUTPUTS, "numbers")
        assert checked(g)["loop"] == []

    def test_shortest_input_decides(self):
        g, loop, f = for_loop([1, 2, 3, 4])
        const(g, ["a", "b"], "names")
        loop.add_input("name", mode="index")
        g.connect("names", "out", loop, "name")
        loop.add_output("pairs", mode="collect")
        f.add_node("base.format_value", "fmt")
        links(f, "inputs.x > fmt.value", "inputs.name > fmt.template", "fmt.out > outputs.pairs")
        const(g, "{}", "unused")
        expected = [name.format(x) if "{" in name else name for x, name in zip([1, 2, 3, 4], "ab")]
        assert len(checked(g)["loop"]) == len(expected) == 2
        loop.values["count"] = 1                 # a count caps it further
        assert len(checked(g)["loop"]) == 1
        loop.values["count"] = 10
        assert len(checked(g)["loop"]) == 2

    def test_rows_of_an_array(self):
        image = np.arange(12, dtype="float32").reshape(3, 4)
        g, loop, f = for_loop(image)
        loop.add_output("sums", mode="stack")
        loop.add_output("doubled", mode="stack")
        loop.add_output("flat", mode="concat")
        loop.add_output("rows", mode="collect")
        f.add_node("base.array_sum", "sum")
        f.add_node("base.add", "twice")
        links(f, "inputs.x > sum.array", "inputs.x > twice.a", "inputs.x > twice.b",
              "sum.out > outputs.sums", "twice.out > outputs.doubled",
              "twice.out > outputs.flat", "inputs.x > outputs.rows")
        result = checked(g)["loop"]
        assert np.array_equal(result["sums"], np.array([row.sum() for row in image]))
        assert np.array_equal(result["doubled"], image * 2) and result["doubled"].dtype == "float32"
        assert np.array_equal(result["flat"], (image * 2).ravel())
        assert len(result["rows"]) == 3 and np.array_equal(result["rows"][1], image[1])

    def test_elements_of_an_array_are_plain_scalars(self):
        g, loop, f = for_loop(np.array([1.5, 2.5]))
        loop.add_output("items", mode="collect")
        f.connect("inputs", "x", OUTPUTS, "items")
        result = checked(g)["loop"]
        assert result == [1.5, 2.5] and all(type(v) is float for v in result)

    def test_tuples_can_be_indexed(self):
        g, loop, f = for_loop((1, 2.5, 4))
        loop.add_output("items", mode="collect")
        f.connect("inputs", "x", OUTPUTS, "items")
        assert checked(g)["loop"] == [1, 2.5, 4]

    @pytest.mark.parametrize("mode, element, expected_type, expected", [
        ("collect", 1.5, "List[Float]", []),
        ("stack", 1.5, "Array[float64, 1]", np.empty(0)),
        ("stack", True, "Array[bool, 1]", np.empty(0, dtype=bool)),
        ("stack", np.zeros((2, 3), dtype="uint8"), "Array[uint8, 3]",
         np.empty((0, 0, 0), dtype="uint8")),
        ("concat", np.zeros(4, dtype="int16"), "Array[int16, 1]", np.empty(0, dtype="int16")),
        ("concat", [1, 2], "List[Int]", []),
    ])
    def test_zero_iterations(self, mode, element, expected_type, expected):
        g, loop, f = for_loop(count=0)
        const(g, element, "e")
        loop.add_input("e")
        g.connect("e", "out", loop, "e")
        loop.add_output("y", mode=mode)
        f.connect("inputs", "e", OUTPUTS, "y")
        assert str(analyze(g).out_type("loop", "y")) == expected_type
        result = checked(g)["loop"]
        if isinstance(expected, np.ndarray):
            assert result.shape == expected.shape and result.dtype == expected.dtype
        else:
            assert result == expected

    def test_zero_iterations_and_a_plain_output(self):
        g, loop, f = for_loop([])
        loop.add_register("r")
        loop.values["r"] = "start"
        loop.add_output("y")
        f.add_node("base.to_text", "t")
        links(f, "inputs.x > t.x", "t.out > outputs.y", "t.out > outputs.r")
        with pytest.raises(NodeError, match="ran zero times.*fallback or use a shift register"):
            run(g)
        loop.outputs[0].fallback = "nothing"
        assert checked(g)["loop"] == {"y": "nothing", "r": "start"}

    def test_nested_loops(self):
        table = [[1, 2, 3], [4, 5, 6]]
        g, outer, f = for_loop(table)
        outer.add_register("total")
        outer.values["total"] = 0
        outer.add_output("row_sums", mode="collect")
        inner = f.add_for("inner")
        inner.add_input("v", mode="index")
        inner.add_register("s")
        inner.values["s"] = 0
        inner.frame.add_node("base.add", "add")
        links(inner.frame, "inputs.s > add.a", "inputs.v > add.b", "add.out > outputs.s")
        f.add_node("base.add", "grand")
        links(f, "inputs.x > inner.v", "inner.s > outputs.row_sums", "inputs.total > grand.a",
              "inner.s > grand.b", "grand.out > outputs.total")
        result = checked(g)["loop"]
        assert result == {"row_sums": [sum(row) for row in table],
                          "total": sum(map(sum, table))}

    def test_multiplication_table_stacks_into_a_matrix(self):
        g, outer, f = for_loop(count=3)
        outer.add_output("table", mode="stack")
        inner = f.add_for("inner")
        inner.values["count"] = 4
        inner.add_input("row")
        inner.add_output("line", mode="stack")
        inner.frame.add_node("base.multiply", "mul")
        links(inner.frame, "inputs.row > mul.a", "inputs.i > mul.b", "mul.out > outputs.line")
        links(f, "inputs.i > inner.row", "inner.line > outputs.table")
        expected = np.array([[r * c for c in range(4)] for r in range(3)])
        result = checked(g)["loop"]
        assert np.array_equal(result, expected) and result.dtype == expected.dtype
        assert str(analyze(g).out_type("loop", "table")) == "Array[int64, 2]"

    def test_concat_of_lists(self):
        g, loop, f = for_loop([[1], [2, 3], []])
        loop.add_output("all", mode="concat")
        f.connect("inputs", "x", OUTPUTS, "all")
        assert checked(g)["loop"] == [1, 2, 3]
        g.node("src").config["value"] = [[1], [2.5]]
        assert str(analyze(g).out_type("loop", "all")) == "List[Union[Float, Int]]"
        assert checked(g)["loop"] == [1, 2.5]

    def test_concat_of_arrays_of_different_kinds(self):
        g, loop, f = for_loop((np.zeros(2, dtype="uint8"), np.ones(3)))
        loop.add_output("all", mode="concat")
        f.connect("inputs", "x", OUTPUTS, "all")
        assert str(analyze(g).out_type("loop", "all")) == "Array[_, 1]"
        assert checked(g)["loop"].tolist() == [0, 0, 1, 1, 1]
        loop.values["count"] = 0
        empty = checked(g)["loop"]
        assert isinstance(empty, np.ndarray) and empty.shape == (0,)

    def test_stack_of_ragged_arrays_fails_clearly(self):
        g, loop, f = for_loop([2, 3])
        loop.add_output("y", mode="stack")
        f.add_node("base.array_range", "r")
        links(f, "inputs.x > r.count", "r.out > outputs.y")
        with pytest.raises(NodeError) as info:
            run(g)
        assert info.value.node == "loop" and info.value.pin == "y"


class TestShiftRegisters:
    def test_fibonacci(self):
        g, loop, f = for_loop(count=30)
        loop.add_register("a")
        loop.add_register("b")
        loop.values.update(a=0, b=1)
        f.add_node("base.add", "sum")
        links(f, "inputs.a > sum.a", "inputs.b > sum.b", "inputs.b > outputs.a",
              "sum.out > outputs.b")
        a, b = 0, 1
        for _ in range(30):
            a, b = b, a + b
        assert checked(g)["loop"] == {"a": a, "b": b}

    def test_register_value_is_coerced_to_the_register_type(self):
        g, loop, f = for_loop([0.5, 0.25])
        loop.add_register("acc")
        loop.values["acc"] = 1                       # Int, but the body makes Floats
        loop.add_output("seen", mode="collect")
        f.add_node("base.add", "add")
        links(f, "inputs.acc > add.a", "inputs.x > add.b", "add.out > outputs.acc",
              "inputs.acc > outputs.seen")
        result = checked(g)["loop"]
        assert result == {"seen": [1.0, 1.5], "acc": 1.75}
        assert [type(v) for v in result["seen"]] == [float, float]
        g.node("src").config["value"] = []           # zero iterations: still a Float
        loop.inputs[0].type = "List[Float]"
        del loop.outputs[0]
        f.disconnect(OUTPUTS, "seen")
        out = run(g)["loop"]
        assert out == 1.0 and type(out) is float

    def test_growing_a_list(self):
        g, loop, f = for_loop(count=4)
        loop.add_register("items")
        loop.values["items"] = []
        f.add_node("base.list_append", "push")
        links(f, "inputs.items > push.items", "inputs.i > push.item", "push.out > outputs.items")
        assert checked(g)["loop"] == [0, 1, 2, 3]
        assert loop.values["items"] == []            # the graph's own literal is untouched

    def test_untyped_feedback_is_verified_while_running(self):
        registry = default_registry()

        @registry.node("t.flaky")
        def flaky(i: int):
            return i if i < 2 else "oops"

        g, loop, f = for_loop(count=5)
        loop.add_register("r")
        loop.values["r"] = 0
        f.add_node("t.flaky", "n")
        links(f, "inputs.i > n.i", "n.out > outputs.r")
        with pytest.raises(NodeError, match="shift register 'r'") as info:
            run(g, registry)
        assert info.value.node == "loop" and info.value.pin == "r"

    def test_array_register_running_maximum(self):
        frames = np.random.default_rng(1).random((5, 3, 3))
        g, loop, f = for_loop(frames)
        const(g, np.zeros((3, 3)), "start")
        loop.add_register("best")
        g.connect("start", "out", loop, "best")
        f.add_node("base.maximum", "max")
        links(f, "inputs.best > max.a", "inputs.x > max.b", "max.out > outputs.best")
        assert np.array_equal(checked(g)["loop"], frames.max(axis=0))


class TestWhileLoop:
    def collatz(self, start, max_iterations=None):
        g = Graph()
        loop = g.add_while("loop", max_iterations=max_iterations)
        loop.add_register("n")
        loop.values["n"] = start
        loop.add_output("steps")
        loop.add_output("trail", mode="collect")
        f = loop.frame
        f.add_node("base.remainder", "odd", values={"b": 2})
        f.add_node("base.to_bool", "is_odd")
        case = f.add_case("step", cases=[[True], [False]])
        case.add_input("n")
        case.add_output("next")
        up, down = case.frames
        up.add_node("base.multiply", "triple", values={"b": 3})
        up.add_node("base.add", "plus", values={"b": 1})
        links(up, "inputs.n > triple.a", "triple.out > plus.a", "plus.out > outputs.next")
        down.add_node("base.floor_divide", "half", values={"b": 2})
        links(down, "inputs.n > half.a", "half.out > outputs.next")
        f.add_node("base.less_equal", "done", values={"b": 1})
        f.add_node("base.add", "count", values={"b": 1})
        links(f, "inputs.n > odd.a", "odd.out > is_odd.x", "is_odd.out > step.selector",
              "inputs.n > step.n", "step.next > outputs.n", "step.next > outputs.trail",
              "step.next > done.a", "done.out > outputs.stop",
              "inputs.i > count.a", "count.out > outputs.steps")
        return g

    @pytest.mark.parametrize("start", [1, 2, 6, 7, 27, 97])
    def test_collatz(self, start):
        n, trail = start, []
        while True:                                  # a While loop runs at least once
            n = 3 * n + 1 if n % 2 else n // 2
            trail.append(n)
            if n <= 1:
                break
        result = checked(self.collatz(start))["loop"]
        assert result == {"steps": len(trail), "trail": trail, "n": n}

    def test_iteration_limit(self):
        g = self.collatz(27, max_iterations=10)
        with pytest.raises(NodeError, match="did not stop within 10 iterations") as info:
            run(g)
        assert info.value.node == "loop"
        assert run(self.collatz(6, max_iterations=8))["loop"]["steps"] == 8
        with pytest.raises(NodeError, match="did not stop within 8 iterations"):
            run(self.collatz(12, max_iterations=8))                 # it needs nine

    def test_newton_square_root(self):
        g = Graph()
        const(g, 2.0, "target")
        loop = g.add_while("loop", max_iterations=100)
        loop.add_input("a")
        loop.add_register("x")
        loop.values["x"] = 1.0
        g.connect("target", "out", loop, "a")
        f = loop.frame
        f.add_node("base.divide", "quotient")
        f.add_node("base.add", "sum")
        f.add_node("base.divide", "next", values={"b": 2.0})
        f.add_node("base.subtract", "change")
        f.add_node("base.absolute", "size")
        f.add_node("base.less", "done", values={"b": 1e-12})
        links(f, "inputs.a > quotient.a", "inputs.x > quotient.b", "inputs.x > sum.a",
              "quotient.out > sum.b", "sum.out > next.a", "next.out > change.a",
              "inputs.x > change.b", "change.out > size.x", "size.out > done.a",
              "next.out > outputs.x", "done.out > outputs.stop")
        x = 1.0
        while True:
            new = (x + 2.0 / x) / 2.0
            done = abs(new - x) < 1e-12
            x = new
            if done:
                break
        result = checked(g)["loop"]
        assert result == x and math.isclose(result, math.sqrt(2.0))

    def test_stop_is_checked_while_running_when_untyped(self):
        registry = default_registry()
        registry.node("t.word")(lambda: "yes")
        g = Graph()
        loop = g.add_while("loop")
        loop.frame.add_node("t.word", "w")
        loop.frame.connect("w", "out", OUTPUTS, "stop")
        with pytest.raises(NodeError, match="'stop' must be a Bool"):
            run(g, registry)

    def test_constant_stop_runs_once(self):
        g = Graph()
        loop = g.add_while("loop")
        loop.add_output("n", mode="collect")
        loop.frame.connect("inputs", "i", OUTPUTS, "n")
        loop.frame.node(OUTPUTS).values["stop"] = True
        assert checked(g)["loop"] == [0]


class TestCase:
    def grade(self, selector, cases, results, fallback=ramen.model.UNSET):
        g = Graph()
        const(g, selector, "sel")
        case = g.add_case("case", cases=cases)
        case.add_output("y", fallback=fallback)
        g.connect("sel", "out", case, "selector")
        for frame, value in zip(case.frames, results):
            if value is not None:
                frame.node(OUTPUTS).values["y"] = value
        return g

    @pytest.mark.parametrize("selector, expected", [(True, "yes"), (False, "no")])
    def test_bool(self, selector, expected):
        assert checked(self.grade(selector, [[True], [False]], ["yes", "no"]))["case"] == expected

    @pytest.mark.parametrize("score", [-5, 0, 49, 50, 69, 70, 89, 90, 100, 101, 1000])
    def test_int_ranges(self, score):
        g = self.grade(score, [[(90, 100)], [(70, 89)], [(50, 69)], [0, 101], None],
                       ["A", "B", "C", "edge", "F"])
        if 90 <= score <= 100:
            expected = "A"
        elif 70 <= score <= 89:
            expected = "B"
        elif 50 <= score <= 69:
            expected = "C"
        elif score in (0, 101):
            expected = "edge"
        else:
            expected = "F"
        assert checked(g)["case"] == expected

    @pytest.mark.parametrize("value, expected", [(1, "low"), (5, "five"), (9, "other")])
    def test_the_default_frame_can_come_first(self, value, expected):
        g = self.grade(value, [None, [(0, 3)], [5]], ["other", "low", "five"])
        assert checked(g)["case"] == expected

    @pytest.mark.parametrize("word, expected", [("add", 1), ("plus", 1), ("sub", 2), ("?", 0)])
    def test_strings(self, word, expected):
        assert checked(self.grade(word, [["add", "plus"], ["sub"], None], [1, 2, 0]))["case"] == \
            expected

    @pytest.mark.parametrize("colour", list(Colour))
    def test_enum(self, colour):
        g = self.grade(colour, [["RED"], ["GREEN", "BLUE"]], ["warm", "cool"])
        assert run(g, check_outputs=True)["case"] == ("warm" if colour is Colour.RED else "cool")

    def test_true_is_not_one(self):
        registry = default_registry()
        registry.node("t.any", config=("value",))(lambda value: value)
        for value, expected in [(1, "int"), (True, "bool"), ("1", "other"), (1.0, "other")]:
            g = Graph()
            g.add_node("t.any", "sel", config={"value": value})
            case = g.add_case("case", cases=[[1], [True], None])
            case.add_output("y")
            g.connect("sel", "out", case, "selector")
            for frame, text in zip(case.frames, ["int", "bool", "other"]):
                frame.node(OUTPUTS).values["y"] = text
            assert run(g, registry)["case"] == expected

    def test_fallback_for_an_unwired_frame(self):
        g = self.grade(False, [[True], [False]], [1.5, None], fallback=0.0)
        assert checked(g)["case"] == 0.0
        g.node("sel").config["value"] = True
        assert checked(g)["case"] == 1.5

    def test_only_the_chosen_frame_runs(self):
        g = self.grade(True, [[True], [False]], ["fine", "never"])
        bad = g.node("case").frames[1]
        bad.add_node("base.fail", "boom", values={"message": "should not run"})
        assert checked(g)["case"] == "fine"
        g.node("sel").config["value"] = False
        with pytest.raises(NodeError, match="should not run") as info:
            run(g)
        assert info.value.node == "case[1]/boom" and info.value.context == ["frame 1 of 'case'"]

    def test_selector_and_inputs_inside_the_frame(self):
        g = Graph()
        const(g, 7, "sel")
        const(g, 10, "base")
        case = g.add_case("case", cases=[[(0, 9)], None])
        case.add_input("base")
        case.add_output("y")
        links(g, "sel.out > case.selector", "base.out > case.base")
        small, other = case.frames
        small.add_node("base.add", "add")
        links(small, "inputs.selector > add.a", "inputs.base > add.b", "add.out > outputs.y")
        links(other, "inputs.base > outputs.y")
        assert checked(g)["case"] == 17
        g.node("sel").config["value"] = 12
        assert checked(g)["case"] == 10

    def test_output_is_converted_to_the_joined_type(self):
        g = self.grade(True, [[True], [False]], [1, 2.5])
        value = checked(g)["case"]
        assert value == 1.0 and type(value) is float


class TestSubgraph:
    def test_runs_like_inline_nodes(self):
        g = Graph()
        const(g, 3.0, "x")
        sub = g.add_subgraph("hyp")
        sub.add_input("a")
        sub.add_input("b")
        sub.add_output("c")
        sub.values["b"] = 4.0
        f = sub.frame
        for name in ("aa", "bb"):
            f.add_node("base.multiply", name)
        f.add_node("base.add", "sum")
        f.add_node("base.sqrt", "root")
        links(f, "inputs.a > aa.a", "inputs.a > aa.b", "inputs.b > bb.a", "inputs.b > bb.b",
              "aa.out > sum.a", "bb.out > sum.b", "sum.out > root.x", "root.out > outputs.c")
        g.connect("x", "out", sub, "a")
        g.add_node("base.to_text", "show")
        g.connect(sub, "c", "show", "x")
        result = checked(g)
        assert result["hyp"] == 5.0 and result["show"] == "5.0"

    def test_empty_subgraph_and_constant_output(self):
        g = Graph()
        sub = g.add_subgraph("s")
        assert run(g).outputs("s") == {}
        sub.add_output("k")
        sub.frame.node(OUTPUTS).values["k"] = [1, 2]
        assert checked(g)["s"] == [1, 2]


class TestRunning:
    def failing(self):
        g, loop, f = for_loop([1.0, 2.0, 0.0, 4.0])
        loop.add_output("y", mode="collect")
        f.add_node("base.to_text", "text")
        f.add_node("base.to_float", "back")
        f.add_node("base.constant", "bad", config={"value": "nan?"})
        f.add_node("base.select", "pick")
        f.add_node("base.equal", "zero", values={"b": 0.0})
        links(f, "inputs.x > text.x", "inputs.x > zero.a", "zero.out > pick.condition",
              "bad.out > pick.if_true", "text.out > pick.if_false", "pick.out > back.x",
              "back.out > outputs.y")
        return g

    def test_errors_say_where_and_when(self):
        with pytest.raises(NodeError) as info:
            run(self.failing())
        error = info.value
        assert error.node == "loop/back" and error.spec == "base.to_float"
        assert isinstance(error.original, ValueError)
        assert error.context == ["iteration 2 of 'loop'"]
        assert str(error).endswith("[iteration 2 of 'loop']")

    def test_context_through_nested_structures(self):
        g, outer, f = for_loop(count=3)
        inner = f.add_for("inner")
        inner.values["count"] = 2
        inner.frame.add_node("base.divide", "div", values={"a": 1})
        inner.frame.add_node("base.subtract", "sub", values={"b": 1})
        inner.add_input("row")
        links(inner.frame, "inputs.row > sub.a", "sub.out > div.b")
        f.connect("inputs", "i", "inner", "row")
        with pytest.raises(NodeError) as info:
            run(g)
        assert info.value.node == "loop/inner/div"
        assert info.value.context == ["iteration 0 of 'loop/inner'", "iteration 1 of 'loop'"]

    def test_events_name_nodes_by_path(self):
        events = []
        with pytest.raises(NodeError):
            run(self.failing(), listener=events.append)
        kinds = [(e.kind, e.node) for e in events]
        assert kinds[0] == ("graph-started", None) and kinds[-1][0] == "graph-failed"
        assert ("node-started", "loop") in kinds and ("node-finished", "loop/text") in kinds
        failed = [node for kind, node in kinds if kind == "node-failed"]
        assert failed == ["loop/back", "loop"]
        assert kinds.count(("node-started", "loop/text")) == 3
        assert not any(node and node.endswith(("inputs", "outputs")) for _, node in kinds)

    def test_cancel(self):
        g = Graph()
        loop = g.add_while("loop")
        loop.frame.node(OUTPUTS).values["stop"] = False
        loop.frame.add_node("base.current_time", "tick")
        calls = []

        def cancel():
            calls.append(1)
            return len(calls) > 25

        events = []
        with pytest.raises(Cancelled):
            run(g, cancel=cancel, listener=events.append)
        assert events[-1].kind == "graph-failed" and isinstance(events[-1].error, Cancelled)
        assert not issubclass(Cancelled, NodeError)

    def test_a_plan_can_run_many_times(self):
        g, loop, f = for_loop(count=3)
        loop.add_register("items")
        loop.values["items"] = ["start"]
        f.add_node("base.to_text", "t")
        f.add_node("base.list_append", "push")
        links(f, "inputs.i > t.x", "inputs.items > push.items", "t.out > push.item",
              "push.out > outputs.items")
        plan = compile_graph(g)
        expected = ["start", "0", "1", "2"]
        assert run(plan)["loop"] == expected and run(plan)["loop"] == expected

    def test_targets_work_with_structures(self):
        g, loop, f = for_loop([1, 2])
        loop.add_output("y", mode="collect")
        f.connect("inputs", "x", OUTPUTS, "y")
        g.add_node("base.fail", "boom", values={"message": "not needed"})
        result = run(g, targets=["loop"])
        assert result["loop"] == [1, 2] and "boom" not in result

    def test_wrong_outputs_inside_a_frame_are_caught(self):
        registry = default_registry()

        @registry.node("t.liar")
        def liar(x: int) -> int:
            return "text"                           # type: ignore[return-value]

        g, loop, f = for_loop([1])
        loop.add_output("y")
        f.add_node("t.liar", "n")
        links(f, "inputs.x > n.x", "n.out > outputs.y")
        with pytest.raises(NodeError, match="declared Int") as info:
            run(g, registry, check_outputs=True)
        assert info.value.node == "loop/n"

    def test_declared_tunnel_types_convert(self):
        g, loop, f = for_loop([1, 2])
        loop.inputs[0].type = "List[Float]"
        loop.add_output("y", mode="collect")
        loop.add_output("z", type="Float")
        links(f, "inputs.x > outputs.y", "inputs.i > outputs.z")
        result = checked(g)["loop"]
        assert result == {"y": [1.0, 2.0], "z": 1.0}
        assert [type(v) for v in result["y"]] == [float, float] and type(result["z"]) is float

    def test_tunnels_take_literals(self):
        g, loop, f = for_loop()
        loop.add_input("x", mode="index")
        loop.add_input("offset")
        loop.values.update(x=[1, 2, 3], offset=10)
        loop.add_output("y", mode="collect")
        f.add_node("base.add", "add")
        links(f, "inputs.x > add.a", "inputs.offset > add.b", "add.out > outputs.y")
        assert checked(g)["loop"] == [11, 12, 13]
        assert loop.values["x"] == [1, 2, 3]

    def test_deep_nesting(self):
        g = Graph()
        const(g, 1, "one")
        frame, owner = g, None
        for depth in range(12):
            sub = frame.add_subgraph(f"level{depth}")
            sub.add_input("v")
            sub.add_output("v")
            frame.connect("one" if depth == 0 else "inputs", "out" if depth == 0 else "v", sub, "v")
            if depth:
                frame.connect(sub, "v", OUTPUTS, "v")
            frame = sub.frame
        frame.add_node("base.add", "add", values={"b": 41})
        links(frame, "inputs.v > add.a", "add.out > outputs.v")
        assert checked(g)["level0"] == 42

    def test_dynamic_indexing_fails_clearly(self):
        registry = default_registry()
        registry.node("t.any")(lambda: 5)
        g = Graph()
        g.add_node("t.any", "src")
        loop = g.add_for("loop")
        loop.add_input("x", mode="index")
        g.connect("src", "out", loop, "x")
        with pytest.raises(NodeError, match="cannot be indexed") as info:
            run(g, registry)
        assert info.value.pin == "x"
