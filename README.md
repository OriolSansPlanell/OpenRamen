# OpenRamen

A LabVIEW-style visual dataflow framework on top of Python: typed wires,
loops and cases as structures, a front panel generated from pin types, and a
palette built by importing ordinary Python libraries.

The name is a reminder of what the code looks like if nobody keeps it tidy.

## Status

Early. The **headless core** and the **importer** work and are tested; there
is no editor yet.

| Milestone | State |
| --- | --- |
| 0. Specification | Done: [docs/spec.md](docs/spec.md) |
| 1. Headless core: types, graph model, base palette, checker, runtime | Done |
| 2. Importer: turn any Python function into a node | Done: [coverage](docs/coverage.md) |
| 3. Structures (For, While, Case, subgraphs) and copy-on-branch rules | Next |
| 4. Scheduler: thread and process pools, caching, cancel | Planned |
| 5. Editor and front panel (Qt) | Planned |

## Try it

Requires Python 3.10 or newer and NumPy 2.

```bash
pip install -e .
python -m openramen run examples/arithmetic.json
python -m openramen check examples/array_statistics.json
python -m openramen nodes
python -m openramen run examples/numpy_histogram.json   # nodes imported from NumPy
python -m openramen import scipy.ndimage                # what importing a module gives
```

Building a graph in code:

```python
import openramen as ramen

g = ramen.Graph("demo")
a = g.add_node("base.constant", config={"value": 2})
b = g.add_node("base.add", values={"b": 0.5})
g.connect(a, "out", b, "a")

print(ramen.analyze(g).out_type(b.id, "out"))   # Float
print(ramen.run(g)[b.id])                       # 2.5
g.save("demo.json")
```

Turning your own function into a node:

```python
registry = ramen.default_registry()

@registry.node("my.scale", category="Demo")
def scale(image: "Array[float64, 2]", factor: float = 2.0) -> "Array[float64, 2]":
    """Multiply an image by a factor."""
    return image * factor
```

Using a whole library as nodes, without writing any wrapper:

```python
g = ramen.Graph("spectrum")
g.imports = ["numpy", "numpy.fft"]          # saved with the graph
g.add_node("numpy.linspace", "t", values={"start": 0.0, "stop": 1.0, "num": 64})
g.add_node("numpy.sin", "wave")
g.add_node("numpy.fft.rfft", "spectrum")
g.add_node("numpy.abs", "magnitude")
g.connect("t", "out", "wave", "x")
g.connect("wave", "out", "spectrum", "a")
g.connect("spectrum", "out", "magnitude", "x")

print(ramen.run(g)["magnitude"].shape)      # (33,)
```

Pins, types and help come from each function's signature and docstring. Of
166 real NumPy calls used as a benchmark, 165 run correctly from that alone;
a small bundled override file fixes the rest. See [docs/coverage.md](docs/coverage.md).

## Layout

```
src/openramen/
  types.py      type algebra: parsing, wire compatibility, unification
  model.py      Graph, Node, Wire and the JSON file format
  serial.py     JSON encoding of literal values
  library/      NodeSpec, the registry, the base palette, the importer
  packs/        override files for libraries ("node packs")
  check.py      static analysis: pins, types, missing inputs
  runtime/      headless execution
docs/spec.md    the contracts everything builds on
examples/       hand-written graphs, run by the test suite
tests/          the test suite
benchmarks/     per-node overhead and importer coverage measurements
```

## Licence

MIT. See [LICENSE](LICENSE).

## Tests

```bash
pip install -e ".[test]"
pytest
```

No milestone is considered done until its tests pass. The suite includes
property-style tests that compare the static type checker against what
actually happens at run time on thousands of random cases.
