"""Random graphs over the whole base palette.

Whatever the checker accepts must either run to completion with every output
conforming to its statically resolved type, or fail with an honest run-time
error of the operation itself.  A node returning something other than the
checker promised would be a hole in the type system.
"""

import random
import warnings

import numpy as np
import pytest

import openramen as ramen
from openramen import Graph, analyze, run
from openramen import types as T

SKIP = {"base.fail", "base.random_uniform", "base.current_time", "base.constant"}
SPECS = [s for s in ramen.default_registry() if s.id not in SKIP]
VALUES = [
    0, 3, -2, 7, 2.5, -0.5, 1 + 2j, True, False, "ramen", "a b", "42", "",
    [1, 2, 3], [1.5, 2.5], ["a", "b"], [], [[1, 2], [3, 4]], [1, 2.5], (1, 2),
    np.array([1, 2, 3], dtype="uint8"), np.array([[1.5, 2.5]], dtype="float32"),
    np.arange(4, dtype="int64"), np.array(2.0), np.array([True, False]),
    np.array([1j, 2]), np.zeros((2, 2, 2)),
]


def build(rng: random.Random) -> Graph:
    graph = Graph(f"fuzz")
    for i in range(rng.randint(2, 6)):
        graph.add_node("base.constant", f"c{i}", config={"value": rng.choice(VALUES)})
    for i in range(rng.randint(3, 18)):
        spec = rng.choice(SPECS)
        analysis = analyze(graph)
        sources = [(nid, pin, t) for nid, info in analysis.nodes.items()
                   for pin, t in info.out_types.items()]
        node = graph.add_node(spec, f"n{i}")
        for pin in spec.inputs:
            if not pin.required and rng.random() < 0.5:
                continue
            fitting = [(nid, name) for nid, name, t in sources
                       if T.check(t, T.substitute(pin.type, {})).ok]
            if fitting:
                graph.connect(*rng.choice(fitting), node, pin.name)
        if not analyze(graph).ok:        # e.g. a missing input or a type conflict
            graph.remove_node(node)
    return graph


@pytest.mark.parametrize("seed", range(400))
def test_checked_graphs_never_lie_about_types(seed):
    rng = random.Random(seed)
    graph = build(rng)
    analysis = analyze(graph)
    assert analysis.ok, analysis.errors
    before = graph.dumps() if all(
        not isinstance(n.config.get("value"), np.ndarray) or n.config["value"].dtype.kind in "biufc"
        for n in graph) else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = run(graph, check_outputs=True)
        except ramen.NodeError as exc:
            lied = exc.pin is not None and "output is declared" in str(exc)
            assert not lied, f"seed {seed}: {exc}\n{graph.dumps() if before else ''}"
            return
    for (node, pin), value in result.values.items():
        assert T.conforms(value, analysis.out_type(node, pin)), (seed, node, pin)
        assert not isinstance(value, np.generic), (seed, node, type(value))
    if before is not None:
        assert graph.dumps() == before                    # running never edits the graph
        again = run(Graph.loads(before), check_outputs=True)
        assert again.values.keys() == result.values.keys()
