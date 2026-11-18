"""Functions for the worker-process tests.  They live in a module of their own
so that a worker process can import them."""

import os
import threading
import time

import numpy as np


def count_primes(limit: int) -> int:
    """Pure Python and slow on purpose."""
    count = 0
    for n in range(2, limit):
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                break
        else:
            count += 1
    return count


def pid() -> int:
    return os.getpid()


def double(a: np.ndarray) -> np.ndarray:
    return a * 2.0


def describe(a: np.ndarray) -> dict:
    return {"sum": float(a.sum()), "writable": bool(a.flags.writeable), "pid": os.getpid(),
            "shape": list(a.shape)}


def fill(a: np.ndarray, value: float = 7.0) -> None:
    a[...] = value


def every_other(a: np.ndarray) -> np.ndarray:
    return a[::2]                      # a view of its input


def reversed_rows(a: np.ndarray) -> np.ndarray:
    return a[::-1, ::-1]


def sneaky(a: np.ndarray) -> float:
    a[0] = -1.0                        # writes without declaring it
    return float(a.sum())


def split(a: np.ndarray) -> tuple:
    return a[: len(a) // 2].copy(), a[len(a) // 2:].copy()


def bundle(a: np.ndarray) -> list:
    return [a + 1.0, {"twice": a * 2.0, "label": "x"}, (a.shape, 3)]


def total_of(items: list) -> float:
    return float(sum(np.asarray(item).sum() for item in items))


def crash(code: int = 3) -> int:
    os._exit(code)


def nap(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


def fail(message: str) -> int:
    raise ValueError(message)


class Awkward(Exception):
    def __init__(self, a, b):
        super().__init__(f"{a} and {b}")


def fail_awkwardly() -> int:
    raise Awkward(1, 2)                # cannot be rebuilt from its message alone


def make_lock() -> object:
    return threading.Lock()            # cannot be sent back


def identity(x):
    return x
