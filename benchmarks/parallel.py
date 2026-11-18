"""What parallel execution, worker processes and the cache buy.

Run with:  PYTHONPATH=src python benchmarks/parallel.py
"""

from __future__ import annotations

import os
import time

import numpy as np

from openramen import Cache, Graph, compile_graph, default_registry, run
from openramen.library import spec_from_function
from openramen.runtime import workers


def sort_values(a: np.ndarray) -> np.ndarray:
    """Heavy, and NumPy releases the interpreter lock while it sorts."""
    return np.sort(a)


def count_primes(limit: int) -> int:
    """Heavy, pure Python: holds the interpreter lock all the way."""
    count = 0
    for n in range(2, limit):
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                break
        else:
            count += 1
    return count


def brighten(image: np.ndarray) -> np.ndarray:
    return image + 1.0


def timed(plan, **options) -> float:
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        run(plan, **options)
        best = min(best, time.perf_counter() - t0)
    return best


def branches(spec: str, count: int, pin: str, value) -> Graph:
    graph = Graph(spec)
    for k in range(count):
        graph.add_node("base.constant", f"in{k}", config={"value": value(k)})
        graph.add_node(spec, f"n{k}")
        graph.connect(f"in{k}", "out", f"n{k}", pin)
    return graph


def main() -> None:
    registry = default_registry()
    registry.register(spec_from_function(sort_values, "bench.sort"))
    registry.register(spec_from_function(count_primes, "bench.primes_thread"))
    registry.register(spec_from_function(count_primes, "bench.primes_process",
                                         executor="process"))
    registry.register(spec_from_function(brighten, "bench.brighten", executor="process"))
    cores = os.cpu_count() or 1
    print(f"{cores} processors\n")

    rng = np.random.default_rng(0)
    data = [rng.random(3_000_000) for _ in range(2)]
    one = compile_graph(branches("bench.sort", 1, "a", lambda k: data[k]), registry)
    two = compile_graph(branches("bench.sort", 2, "a", lambda k: data[k]), registry)
    print("Two independent branches, each sorting 3 million numbers (NumPy, on threads)")
    print(f"  one branch alone          {timed(one) * 1e3:7.0f} ms")
    print(f"  two branches, one worker  {timed(two, workers=1) * 1e3:7.0f} ms")
    print(f"  two branches, in parallel {timed(two) * 1e3:7.0f} ms")

    workers.start(2)
    limit = 300_000
    one = compile_graph(branches("bench.primes_thread", 1, "limit", lambda k: limit), registry)
    threads = compile_graph(branches("bench.primes_thread", 2, "limit", lambda k: limit),
                            registry)
    processes = compile_graph(branches("bench.primes_process", 2, "limit", lambda k: limit),
                              registry)
    print("\nTwo independent branches of pure Python (counting primes)")
    print(f"  one branch alone          {timed(one) * 1e3:7.0f} ms")
    print(f"  two branches on threads   {timed(threads) * 1e3:7.0f} ms")
    print(f"  two branches in processes {timed(processes) * 1e3:7.0f} ms")

    image = rng.random((4000, 4000))                       # 128 MB
    there = compile_graph(branches("bench.brighten", 1, "image", lambda k: image), registry)
    print(f"\nA {image.nbytes // 1_000_000} MB array to a worker process and back")
    print(f"  through shared memory     {timed(there) * 1e3:7.0f} ms")
    threshold, workers.SHARE_FROM = workers.SHARE_FROM, 1 << 62
    print(f"  through the pipe          {timed(there) * 1e3:7.0f} ms")
    workers.SHARE_FROM = threshold
    t0 = time.perf_counter()
    brighten(image)
    print(f"  the work itself           {(time.perf_counter() - t0) * 1e3:7.0f} ms")

    cache = Cache()
    t0 = time.perf_counter()
    run(two, cache=cache)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    run(two, cache=cache)
    print("\nThe sorting graph with a cache")
    print(f"  first run                 {first * 1e3:7.0f} ms")
    print(f"  second run                {(time.perf_counter() - t0) * 1e3:7.1f} ms")


if __name__ == "__main__":
    main()
