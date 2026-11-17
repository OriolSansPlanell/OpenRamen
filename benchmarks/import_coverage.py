"""How much of real libraries the importer handles.  Prints docs/coverage.md.

Run with:  PYTHONPATH=src:tests python benchmarks/import_coverage.py > docs/coverage.md
"""

from __future__ import annotations

import collections
import importlib
import platform
import sys

import numpy as np

import openramen as ramen
from openramen.library import report

from numpy_cases import CASES, outcome, registry

LIBRARIES = ["numpy", "numpy.linalg", "numpy.fft", "numpy.random", "scipy.ndimage",
             "scipy.signal", "scipy.optimize", "skimage.filters", "skimage.measure",
             "skimage.morphology", "skimage.transform", "math"]


def share(part: int, whole: int) -> str:
    return "n/a" if not whole else f"{100 * part / whole:.0f}%"


def main() -> None:
    print("# Importer coverage\n")
    print("What the import engine makes of real libraries, measured by "
          "`benchmarks/import_coverage.py`.")
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, "
          f"OpenRamen {ramen.__version__}.\n")

    print("## Running right without manual fixes\n")
    print(f"{len(CASES)} real calls to {len({c[0] for c in CASES})} NumPy functions are run "
          "as nodes and compared with calling NumPy directly: same values, same dtypes, "
          "and every output checked against the type the importer declared.\n")
    print("| | Correct | Share |")
    print("| --- | --- | --- |")
    problems = []
    for pack, label in ((False, "From signatures and docstrings alone"),
                        (True, "With the bundled NumPy node pack")):
        reg = registry(pack=pack)
        outcomes = [(case, outcome(reg, *case)) for case in CASES]
        good = sum(result[0] == "ok" for _, result in outcomes)
        print(f"| {label} | {good} of {len(CASES)} | {share(good, len(CASES))} |")
        if not pack:
            problems = [(case[0], result) for case, result in outcomes if result[0] != "ok"]
    if problems:
        print("\nWhat introspection alone gets wrong:\n")
        for name, (kind, detail) in problems:
            print(f"- `numpy.{name}`: {kind}. {detail}")
    print("\nThe cases are in `tests/numpy_cases.py`. They cover creation, shape, arithmetic, "
          "reductions, sorting, linear algebra and FFT, plus deliberately awkward functions: "
          "flags that change the number of results, `*args`, nested results.\n")

    print("## How much is understood statically\n")
    print("| Module | Functions | Imported | Inputs typed | Outputs typed | Fully typed nodes |")
    print("| --- | --- | --- | --- | --- | --- |")
    failures = collections.OrderedDict()
    for name in LIBRARIES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        library = ramen.Registry().import_module(name)
        summary = report(library)
        wanted = summary.candidates - summary.skipped
        version = f" {summary.version}" if summary.version else ""
        print(f"| `{name}`{version} | {wanted} | {share(summary.imported, wanted)} "
              f"| {share(summary.typed_inputs, summary.inputs)} "
              f"| {share(summary.typed_outputs, summary.outputs)} "
              f"| {share(summary.fully_typed, summary.imported)} |")
        for function, reason in summary.failures.items():
            failures[f"{name}.{function}"] = reason
    print("\n\"Typed\" means the pin has a type other than `Any`. An untyped pin still works: "
          "its wires are checked when the graph runs instead of when it is drawn.")
    if failures:
        print("\nNot imported:\n")
        for function, reason in failures.items():
            print(f"- `{function}`: {reason}")
    print("\nClasses and their methods are not imported yet; only functions are counted.")


if __name__ == "__main__":
    sys.exit(main())
