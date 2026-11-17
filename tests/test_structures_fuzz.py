"""Random programs with loops and cases, run as graphs and as plain Python.

Each program is generated once as a small syntax tree.  One interpreter turns
the tree into a graph, another executes it directly; every variable must end
up with the same value both ways.
"""

import operator
import random

import pytest

from openramen import Graph, analyze, run
from openramen.model import UNSET

OPS = {"base.add": operator.add, "base.subtract": operator.sub,
       "base.maximum": max, "base.minimum": min}


class Generator:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.counter = 0

    def fresh(self, prefix="v"):
        self.counter += 1
        return f"{prefix}{self.counter}"

    def body(self, ints, lists, depth):
        """Statements that may use the given variables; returns them together
        with the variables that exist afterwards."""
        rng = self.rng
        ints, lists = list(ints), list(lists)
        statements = []
        for _ in range(rng.randint(1, 4)):
            roll = rng.random()
            if depth < 3 and roll < 0.22:
                statements.append(self.for_loop(ints, lists, depth))
            elif depth < 3 and roll < 0.32:
                statements.append(self.while_loop(ints, lists, depth))
            elif depth < 3 and roll < 0.47:
                statements.extend(self.case(ints, lists, depth))
            elif lists and roll < 0.55:
                out = self.fresh()
                statements.append(("append", out, rng.choice(lists), rng.choice(ints)))
                lists.append(out)
                continue
            elif roll < 0.65:
                out = self.fresh()
                statements.append(("scale", out, rng.choice(ints), rng.randint(-3, 3)))
            elif roll < 0.72:
                out = self.fresh()
                statements.append(("mod", out, rng.choice(ints), rng.randint(1, 9)))
            else:
                out = self.fresh()
                b = rng.choice(ints) if rng.random() < 0.7 else rng.randint(-5, 5)
                statements.append(("op", out, rng.choice(list(OPS)), rng.choice(ints), b))
            produced = statements[-1]
            if produced[0] in ("for", "while"):
                details = produced[1]
                for register, _init, kind in details["regs"]:
                    (ints if kind == "int" else lists).append(register)
                if details["collect"]:
                    lists.append(details["collect"][0])
                if details.get("last"):
                    ints.append(details["last"][0])
            elif produced[0] == "case":
                ints.append(produced[1]["out"])
            else:
                ints.append(produced[1])
        return statements, ints, lists

    def registers(self, ints, lists):
        rng = self.rng
        regs = []
        for _ in range(rng.randint(0, 2)):
            if lists and rng.random() < 0.3:
                regs.append((self.fresh("r"), rng.choice(lists), "list"))
            else:
                regs.append((self.fresh("r"), rng.choice(ints), "int"))
        return regs

    def loop_body(self, ints, lists, regs, depth, extra_ints):
        inner_ints = list(ints) + extra_ints + [r for r, _, kind in regs if kind == "int"]
        inner_lists = list(lists) + [r for r, _, kind in regs if kind == "list"]
        body, body_ints, body_lists = self.body(inner_ints, inner_lists, depth + 1)
        feed = {r: self.rng.choice(body_ints if kind == "int" else body_lists)
                for r, _, kind in regs}
        collect = (self.fresh(), self.rng.choice(body_ints)) if self.rng.random() < 0.6 else None
        return body, feed, collect, body_ints

    def for_loop(self, ints, lists, depth):
        rng = self.rng
        details = {"id": self.fresh("s"), "index": self.fresh(), "item": None, "over": None,
                   "count": None}
        extra = [details["index"]]
        if lists and rng.random() < 0.6:
            details["over"], details["item"] = rng.choice(lists), self.fresh()
            extra.append(details["item"])
            if rng.random() < 0.3:
                details["count"] = rng.randint(0, 3)
        else:
            details["count"] = rng.choice(ints) if rng.random() < 0.2 else rng.randint(0, 4)
        if isinstance(details["count"], str):
            # a wired count could be huge: bound it first
            bounded = self.fresh()
            details["bound"] = ("mod", bounded, details["count"], 4)
            details["count"] = bounded
        details["regs"] = self.registers(ints, lists)
        body, feed, collect, body_ints = self.loop_body(ints, lists, details["regs"], depth, extra)
        details.update(body=body, feed=feed, collect=collect, last=None)
        if rng.random() < 0.4:
            details["last"] = (self.fresh(), rng.choice(body_ints), rng.randint(-9, 9))
        return ("for", details)

    def while_loop(self, ints, lists, depth):
        details = {"id": self.fresh("s"), "index": self.fresh(), "stop": self.fresh("stop"),
                   "turns": self.rng.randint(1, 4)}
        details["regs"] = self.registers(ints, lists)
        body, feed, collect, _ = self.loop_body(ints, lists, details["regs"], depth,
                                                [details["index"]])
        details.update(body=body, feed=feed, collect=collect)
        return ("while", details)

    def case(self, ints, lists, depth):
        rng = self.rng
        key = self.fresh()
        prepare = ("mod", key, rng.choice(ints), 7)
        values = list(range(7))
        rng.shuffle(values)
        first, second = sorted(values[:2]), values[2:4]
        matches = [[(first[0], first[1])] if first[1] - first[0] == 1 else list(first),
                   list(second), None]
        rng.shuffle(matches)
        if rng.random() < 0.3:
            matches = [m for m in matches if m is None or rng.random() < 0.5]
        fallback = rng.randint(-9, 9) if rng.random() < 0.3 else UNSET
        frames = []
        for index, match in enumerate(matches):
            body, body_ints, _ = self.body(ints + [key], lists, depth + 1)
            result = rng.choice(body_ints)
            if fallback is not UNSET and index == 0:
                result = None                       # this frame leaves the output unwired
            frames.append((match, body, result))
        details = {"id": self.fresh("s"), "sel": key, "frames": frames, "out": self.fresh(),
                   "fallback": fallback}
        return [prepare, ("case", details)]


# -- plain Python -------------------------------------------------------------

def execute(statements, env):
    for statement in statements:
        kind = statement[0]
        if kind == "op":
            _, out, spec, a, b = statement
            env[out] = OPS[spec](env[a], env[b] if isinstance(b, str) else b)
        elif kind == "scale":
            env[statement[1]] = env[statement[2]] * statement[3]
        elif kind == "mod":
            env[statement[1]] = env[statement[2]] % statement[3]
        elif kind == "append":
            env[statement[1]] = env[statement[2]] + [env[statement[3]]]
        elif kind == "for":
            d = statement[1]
            if "bound" in d:
                execute([d["bound"]], env)
            sequence = env[d["over"]] if d["over"] else None
            count = env[d["count"]] if isinstance(d["count"], str) else d["count"]
            turns = len(sequence) if count is None else count
            if sequence is not None:
                turns = min(turns, len(sequence))
            regs = {r: env[init] for r, init, _ in d["regs"]}
            collected, last = [], d["last"][2] if d["last"] else None
            for i in range(turns):
                inner = dict(env)
                inner.update(regs)
                inner[d["index"]] = i
                if sequence is not None:
                    inner[d["item"]] = sequence[i]
                execute(d["body"], inner)
                regs = {r: inner[d["feed"][r]] for r in regs}
                if d["collect"]:
                    collected.append(inner[d["collect"][1]])
                if d["last"]:
                    last = inner[d["last"][1]]
            env.update(regs)
            if d["collect"]:
                env[d["collect"][0]] = collected
            if d["last"]:
                env[d["last"][0]] = last
        elif kind == "while":
            d = statement[1]
            regs = {r: env[init] for r, init, _ in d["regs"]}
            collected, i = [], 0
            while True:
                inner = dict(env)
                inner.update(regs)
                inner[d["index"]] = i
                execute(d["body"], inner)
                regs = {r: inner[d["feed"][r]] for r in regs}
                if d["collect"]:
                    collected.append(inner[d["collect"][1]])
                if i >= d["turns"] - 1:
                    break
                i += 1
            env.update(regs)
            if d["collect"]:
                env[d["collect"][0]] = collected
        else:
            d = statement[1]
            key, chosen = env[d["sel"]], None
            for match, body, result in d["frames"]:
                if match is None:
                    chosen = chosen or (body, result)
                elif any(e[0] <= key <= e[1] if isinstance(e, tuple) else e == key
                         for e in match):
                    chosen = (body, result)
                    break
            assert chosen is not None, "the generator always adds a default frame"
            inner = dict(env)
            execute(chosen[0], inner)
            env[d["out"]] = d["fallback"] if chosen[1] is None else inner[chosen[1]]


# -- as a graph ------------------------------------------------------------------

def enter(structure, graph, src):
    """Tunnel every variable in scope into a structure."""
    for name, source in src.items():
        structure.add_input(name)
        graph.connect(*source, structure, name)
    return {name: ("inputs", name) for name in src}


def build(statements, graph, src):
    for statement in statements:
        kind = statement[0]
        if kind in ("op", "scale", "mod", "append"):
            out = statement[1]
            if kind == "append":
                graph.add_node("base.list_append", out)
                graph.connect(*src[statement[2]], out, "items")
                graph.connect(*src[statement[3]], out, "item")
            else:
                spec = {"scale": "base.multiply", "mod": "base.remainder"}.get(kind) or statement[2]
                a, b = statement[-2], statement[-1]
                graph.add_node(spec, out)
                graph.connect(*src[a], out, "a")
                if isinstance(b, str):
                    graph.connect(*src[b], out, "b")
                else:
                    graph.node(out).values["b"] = b
            src[out] = (out, "out")
        elif kind in ("for", "while"):
            d = statement[1]
            if "bound" in d:
                build([d["bound"]], graph, src)
            loop = graph.add_structure(kind, d["id"])
            inner = enter(loop, graph, src)
            inner[d["index"]] = ("inputs", "i")
            if kind == "for":
                if d["over"]:
                    loop.add_input(d["item"], mode="index")
                    graph.connect(*src[d["over"]], loop, d["item"])
                    inner[d["item"]] = ("inputs", d["item"])
                if isinstance(d["count"], str):
                    graph.connect(*src[d["count"]], loop, "count")
                elif d["count"] is not None:
                    loop.values["count"] = d["count"]
            for register, init, _ in d["regs"]:
                loop.add_register(register)
                graph.connect(*src[init], loop, register)
                inner[register] = ("inputs", register)
            frame = loop.frame
            build(d["body"], frame, inner)
            for register, _, _ in d["regs"]:
                frame.connect(*inner[d["feed"][register]], "outputs", register)
                src[register] = (loop.id, register)
            if d["collect"]:
                out, var = d["collect"]
                loop.add_output(out, mode="collect")
                frame.connect(*inner[var], "outputs", out)
                src[out] = (loop.id, out)
            if d.get("last"):
                out, var, fallback = d["last"]
                loop.add_output(out, fallback=fallback)
                frame.connect(*inner[var], "outputs", out)
                src[out] = (loop.id, out)
            if kind == "while":
                frame.add_node("base.greater_equal", d["stop"], values={"b": d["turns"] - 1})
                frame.connect("inputs", "i", d["stop"], "a")
                frame.connect(d["stop"], "out", "outputs", "stop")
        else:
            d = statement[1]
            case = graph.add_case(d["id"], cases=[match for match, _, _ in d["frames"]])
            graph.connect(*src[d["sel"]], case, "selector")
            case.add_output(d["out"], fallback=d["fallback"])
            first = True
            for frame, (_match, body, result) in zip(case.frames, d["frames"]):
                inner = enter(case, graph, src) if first else {n: ("inputs", n) for n in src}
                first = False
                build(body, frame, inner)
                if result is not None:
                    frame.connect(*inner[result], "outputs", d["out"])
            src[d["out"]] = (case.id, d["out"])


def program(seed):
    generator = Generator(seed)
    rng = generator.rng
    env = {generator.fresh("c"): rng.randint(-20, 20) for _ in range(3)}
    lists = {generator.fresh("k"): [rng.randint(-9, 9) for _ in range(rng.randint(1, 4))]}
    statements, _, _ = generator.body(list(env), list(lists), 0)
    env.update(lists)
    graph, src = Graph(f"program {seed}"), {}
    for name, value in env.items():
        graph.add_node("base.constant", name, config={"value": value})
        src[name] = (name, "out")
    build(statements, graph, src)
    execute(statements, env)
    return graph, src, env


@pytest.mark.parametrize("seed", range(400))
def test_random_programs_match_plain_python(seed):
    graph, src, expected = program(seed)
    analysis = analyze(graph)
    assert analysis.ok, analysis.problems
    for attempt in (graph, Graph.loads(graph.dumps())):
        result = run(attempt, check_outputs=True, detect_mutation=attempt is graph)
        for name, source in src.items():
            got = result.values[source]
            assert got == expected[name] and type(got) is type(expected[name]), (name, source)


def test_the_programs_are_worth_running():
    kinds = {"for": 0, "while": 0, "case": 0, "nested": 0}

    def count(statements, depth):
        for statement in statements:
            if statement[0] in ("for", "while"):
                kinds[statement[0]] += 1
                kinds["nested"] += depth > 0
                count(statement[1]["body"], depth + 1)
            elif statement[0] == "case":
                kinds["case"] += 1
                kinds["nested"] += depth > 0
                for _, body, _ in statement[1]["frames"]:
                    count(body, depth + 1)

    for seed in range(400):
        generator = Generator(seed)
        rng = generator.rng
        ints = [generator.fresh("c") for _ in range(3)]
        [rng.randint(-20, 20) for _ in range(3)]
        count(generator.body(ints, [generator.fresh("k")], 0)[0], 0)
    assert min(kinds.values()) > 100, kinds


# -- damaged programs ---------------------------------------------------------------

def diagrams(graph):
    """Every diagram of a program: the graph itself and all frames, any depth."""
    yield graph
    for node in graph:
        for frame in getattr(node, "frames", ()):
            yield from diagrams(frame)


def damage(graph, rng):
    """Break a graph the way careless editing would."""
    where = rng.choice(list(diagrams(graph)))
    structures = [n for d in diagrams(graph) for n in d if hasattr(n, "frames")]
    roll = rng.randrange(12)
    if roll == 0 and where.wires:
        wire = rng.choice(where.wires)
        where.disconnect(wire.dst, wire.dst_pin)
    elif roll == 1 and len(where) > 0:
        where.remove_node(rng.choice([n.id for n in where]))
    elif roll == 2 and structures:
        structure = rng.choice(structures)
        if structure.inputs:
            rng.choice(structure.inputs).mode = rng.choice(["plain", "index"])
    elif roll == 3 and structures:
        structure = rng.choice(structures)
        if structure.outputs:
            rng.choice(structure.outputs).mode = rng.choice(["last", "collect", "stack", "concat"])
    elif roll == 4 and structures:
        structure = rng.choice(structures)
        if structure.cases:
            structure.cases[rng.randrange(len(structure.cases))] = rng.choice(
                [None, [True], ["x"], [0, 1], [(0, 6)], [2.5], []])
    elif roll == 5 and structures:
        structure = rng.choice(structures)
        structure.values[rng.choice(["count", "selector", "stop", "i", "nope"])] = rng.choice(
            [0, 3, -1, "text", None, True, 2.5, [1]])
    elif roll == 6 and structures:
        structure = rng.choice(structures)
        if structure.registers:
            structure.registers.remove(rng.choice(structure.registers))
    elif roll == 7 and structures:
        structure = rng.choice(structures)
        structure.kind = rng.choice(["for", "while", "case", "subgraph"])
    elif roll == 8 and structures:
        structure = rng.choice(structures)
        if structure.outputs:
            rng.choice(structure.outputs).type = rng.choice(["Str", "Float", "List[Int]", "Nope["])
    elif roll == 9 and structures:
        structure = rng.choice(structures)
        if structure.inputs and rng.random() < 0.5:
            del structure.inputs[rng.randrange(len(structure.inputs))]
        elif structure.frames:
            del structure.frames[rng.randrange(len(structure.frames))]
    elif roll == 10 and where.wires:
        wire = rng.choice(where.wires)
        nodes = [n.id for n in where]
        try:
            where.connect(rng.choice(nodes), rng.choice(["out", "i", "nope", wire.src_pin]),
                          wire.dst, wire.dst_pin, replace=True)
        except Exception:
            pass                                    # the model refused: nothing was damaged
    elif roll == 11:
        for node in where:
            if node.values and rng.random() < 0.5:
                key = rng.choice(list(node.values))
                node.values[key] = rng.choice(["text", None, [1, "a"], 2.5, {"k": 1}])


@pytest.mark.parametrize("seed", range(300))
def test_damaged_programs_are_reported_not_crashed_on(seed):
    from openramen import NodeError
    graph, _, _ = program(seed)
    rng = random.Random(seed * 7919)
    for _ in range(rng.randint(1, 4)):
        damage(graph, rng)
    analysis = analyze(graph)                       # must never raise
    for problem in analysis.problems:
        assert problem.code and problem.message and str(problem)
    if analysis.ok:
        try:
            run(graph, check_outputs=True)
        except NodeError as error:
            assert error.node and str(error)
