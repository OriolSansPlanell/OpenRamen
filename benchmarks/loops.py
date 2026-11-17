"""Cost of loops, and what the in-place shift register saves.

Run with:  PYTHONPATH=src python benchmarks/loops.py
"""

from __future__ import annotations

import time

import numpy as np

from openramen import Graph, compile_graph, default_registry, run


def best(fn, repeat: int = 5) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def counting_loop(count: int) -> Graph:
    """total = total + i * 2, ``count`` times: two nodes per iteration."""
    graph = Graph("counting")
    loop = graph.add_for("loop")
    loop.values.update(count=count, total=0)
    loop.add_register("total")
    frame = loop.frame
    frame.add_node("base.multiply", "double", values={"b": 2})
    frame.add_node("base.add", "add")
    frame.connect("inputs", "i", "double", "a")
    frame.connect("inputs", "total", "add", "a")
    frame.connect("double", "out", "add", "b")
    frame.connect("add", "out", "outputs", "total")
    return graph


def buffer_loop(registry, size: int, count: int, watched: bool) -> Graph:
    """Add 1 to a big array ``count`` times.  With ``watched`` a second node
    reads the register, so the buffer must be copied every iteration."""
    graph = Graph("buffer")
    graph.add_node("base.constant", "start", config={"value": np.zeros(size)})
    loop = graph.add_for("loop")
    loop.values["count"] = count
    loop.add_register("buffer")
    graph.connect("start", "out", loop, "buffer")
    frame = loop.frame
    frame.add_node("bench.add_one", "bump")
    frame.connect("inputs", "buffer", "bump", "a")
    frame.connect("bump", "a", "outputs", "buffer")
    if watched:
        frame.add_node("base.array_shape", "watch")
        frame.connect("inputs", "buffer", "watch", "array")
    return graph


def main() -> None:
    print("For loop with two nodes per iteration")
    print(f"{'iterations':>12} {'run':>10} {'per iteration':>15} {'plain Python':>14}")
    for count in (1_000, 100_000):
        plan = compile_graph(counting_loop(count))
        running = best(lambda: run(plan), 3)

        def plain() -> int:
            total = 0
            for i in range(count):
                total = total + i * 2
            return total

        assert run(plan)["loop"] == plain()
        python = best(plain, 3)
        print(f"{count:>12} {running * 1e3:>8.1f}ms {running / count * 1e6:>13.2f}us "
              f"{python * 1e3:>12.2f}ms")

    registry = default_registry()

    @registry.node("bench.add_one", passthrough=("a",))
    def add_one(a: np.ndarray) -> None:
        a += 1.0

    size, count = 4_000_000, 50
    print(f"\nAdding 1 to a {size * 8 // 1_000_000} MB array {count} times")
    for watched, label in ((False, "in place (one buffer)"), (True, "copy per iteration")):
        plan = compile_graph(buffer_loop(registry, size, count, watched), registry)
        assert bool(plan.steps[-1].structure.in_place) is not watched
        running = best(lambda: run(plan), 3)
        print(f"  {label:<24} {running * 1e3:>8.1f}ms")


if __name__ == "__main__":
    main()
