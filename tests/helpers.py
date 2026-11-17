"""Small helpers shared by the tests."""

from __future__ import annotations

from typing import Any, Optional

from openramen import Graph, Registry, analyze, run


def const(graph: Graph, value: Any, id: Optional[str] = None):
    return graph.add_node("base.constant", id, config={"value": value})


def single(spec_id: str, registry: Optional[Registry] = None, *,
           config: Optional[dict] = None, literals: Optional[dict] = None,
           **wired: Any) -> Graph:
    """A graph with one node ``n`` whose inputs come from constant nodes."""
    graph = Graph(spec_id)
    graph.add_node(spec_id, "n", config=config, values=literals)
    for pin, value in wired.items():
        source = const(graph, value, f"in_{pin}")
        graph.connect(source, "out", "n", pin)
    return graph


def call(spec_id: str, registry: Optional[Registry] = None, **kwargs: Any) -> Any:
    """Run one node with every output checked against its inferred type."""
    graph = single(spec_id, registry, **kwargs)
    return run(graph, registry, check_outputs=True)["n"]


def out_type(spec_id: str, registry: Optional[Registry] = None, pin: str = "out",
             **kwargs: Any) -> str:
    graph = single(spec_id, registry, **kwargs)
    analysis = analyze(graph, registry)
    assert analysis.ok, analysis.problems
    return str(analysis.out_type("n", pin))


def links(graph: Graph, *wires: str) -> None:
    """Connect pins written as ``"source.pin > destination.pin"``."""
    for text in wires:
        source, destination = (part.strip() for part in text.split(">"))
        src, src_pin = source.rsplit(".", 1)
        dst, dst_pin = destination.rsplit(".", 1)
        graph.connect(src, src_pin, dst, dst_pin)
