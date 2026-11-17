# OpenRamen

A LabVIEW-style visual dataflow framework on top of Python: typed wires,
loops and cases as structures, a front panel generated from pin types, and a
palette built by importing ordinary Python libraries.

The name is a reminder of what the code looks like if nobody keeps it tidy.

## Status

Early. The **headless core** works and is tested; there is no editor yet.

| Milestone | State |
| --- | --- |
| 0. Specification | Done: [docs/spec.md](docs/spec.md) |
| 1. Headless core: types, graph model, base palette, checker, runtime | Done |
| 2. Importer: turn any Python function into a node | Next |
| 3. Structures (For, While, Case, subgraphs) and copy-on-branch rules | Planned |
| 4. Scheduler: thread and process pools, caching, cancel | Planned |
| 5. Editor and front panel (Qt) | Planned |

## Try it

Requires Python 3.10 or newer and NumPy 2.

```bash
pip install -e .
python -m openramen run examples/arithmetic.json
python -m openramen check examples/array_statistics.json
python -m openramen nodes
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

## Layout

```
src/openramen/
  types.py      type algebra: parsing, wire compatibility, unification
  model.py      Graph, Node, Wire and the JSON file format
  serial.py     JSON encoding of literal values
  library/      NodeSpec, the registry, the base palette
  check.py      static analysis: pins, types, missing inputs
  runtime/      headless execution
docs/spec.md    the contracts everything builds on
examples/       hand-written graphs, run by the test suite
tests/          the test suite
benchmarks/     per-node overhead measurement
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

No milestone is considered done until its tests pass. The suite includes
property-style tests that compare the static type checker against what
actually happens at run time on thousands of random cases.
