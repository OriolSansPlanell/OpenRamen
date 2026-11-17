"""Per-node overhead of the headless runtime.

Run with:  PYTHONPATH=src python benchmarks/overhead.py

A chain of trivial ``base.add`` nodes is checked, compiled and run; the time
per node is what OpenRamen itself costs on top of the work a node does.
"""

from __future__ import annotations

import time

from openramen import Graph, analyze, compile_graph, run


def chain(length: int) -> Graph:
    graph = Graph("chain")
    graph.add_node("base.constant", "start", config={"value": 0})
    previous = "start"
    for i in range(length):
        node = graph.add_node("base.add", f"n{i:06d}", values={"b": 1})
        graph.connect(previous, "out", node, "a")
        previous = node.id
    return graph


def best(fn, repeat: int = 5) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> None:
    print(f"{'nodes':>8} {'build':>10} {'check':>10} {'run (plan)':>12} {'per node':>10}")
    for length in (100, 1_000, 10_000):
        t0 = time.perf_counter()
        graph = chain(length)
        build = time.perf_counter() - t0
        check = best(lambda: analyze(graph), 3)
        plan = compile_graph(graph)
        running = best(lambda: run(plan))
        assert run(plan)[f"n{length - 1:06d}"] == length
        print(f"{length:>8} {build * 1e3:>8.1f}ms {check * 1e3:>8.1f}ms "
              f"{running * 1e3:>10.2f}ms {running / length * 1e6:>8.2f}us")


if __name__ == "__main__":
    main()
