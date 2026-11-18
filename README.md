# OpenRamen

A LabVIEW-style visual dataflow framework on top of Python: typed wires,
loops and cases as structures, a front panel generated from pin types, and a
palette built by importing ordinary Python libraries.

The name is a reminder of what the code looks like if nobody keeps it tidy.

## Status

Early. The **headless core**, the **importer**, **structures** (loops,
cases, subgraphs, Try and With) and the **scheduler** (parallel branches,
worker processes, caching, Stop) work and are tested; there is no editor yet.

| Milestone | State |
| --- | --- |
| 0. Specification | Done: [docs/spec.md](docs/spec.md) |
| 1. Headless core: types, graph model, base palette, checker, runtime | Done |
| 2. Importer: turn any Python function into a node | Done: [coverage](docs/coverage.md) |
| 3. Structures (For, While, Case, subgraphs) and copy-on-branch rules | Done |
| 4. Scheduler: thread and process pools, caching, cancel, Try and With | Done |
| 5. Editor and front panel (Qt) | Next |

## Try it

Requires Python 3.10 or newer and NumPy 2. The tests run on Linux, macOS and
Windows on every push.

```bash
pip install -e .
python -m openramen run examples/arithmetic.json
python -m openramen check examples/array_statistics.json
python -m openramen nodes
python -m openramen run examples/numpy_histogram.json   # nodes imported from NumPy
python -m openramen import scipy.ndimage                # what importing a module gives
python -m openramen run examples/collatz.json           # a While loop with a Case inside
python -m openramen run examples/safe_division.json     # a Try structure inside a For loop
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

Loops and cases are structures: nodes that hold a diagram of their own.
Values cross their border through tunnels and shift registers, as in LabVIEW:

```python
g = ramen.Graph("sum of squares")
g.add_node("base.constant", "numbers", config={"value": [1, 2, 3, 4]})

loop = g.add_for("loop")
loop.add_input("x", mode="index")           # one element per iteration
loop.add_register("total")                  # carried from one iteration to the next
loop.add_output("squares", mode="collect")  # one value per iteration, as a list
loop.values["total"] = 0

body = loop.frame
body.add_node("base.multiply", "square")
body.add_node("base.add", "add")
body.connect("inputs", "x", "square", "a")
body.connect("inputs", "x", "square", "b")
body.connect("inputs", "total", "add", "a")
body.connect("square", "out", "add", "b")
body.connect("add", "out", "outputs", "total")
body.connect("square", "out", "outputs", "squares")
g.connect("numbers", "out", loop, "x")

print(ramen.run(g)["loop"])                 # {'squares': [1, 4, 9, 16], 'total': 30}
print(ramen.analyze(g).out_type("loop", "squares"))   # List[Int]
```

Wires carry values, not references. A node that writes to its input gets a
private copy, every other node gets a read-only view, so a branch can never
be changed behind your back, and nothing is copied that does not need to be.
`run(g, detect_mutation=True)` catches nodes that are wrong about what they write.

Branches that do not depend on each other run at the same time, without
anyone asking:

```python
import time
import openramen as ramen

palette = ramen.default_registry()

@palette.node("demo.slow_double")
def slow_double(x: float) -> float:
    time.sleep(0.2)                         # stands for real work
    return 2 * x

g = ramen.Graph("two branches")
g.add_node("base.constant", "x", config={"value": 1.5})
for name in ("left", "right"):
    g.add_node("demo.slow_double", name)
    g.connect("x", "out", name, "x")

running = ramen.start(g, palette)           # returns at once: the graph runs in the background
print(running.result()["left"])             # 3.0, after 0.2 s and not 0.4
# running.cancel() would have stopped it: that is the Stop button

cache = ramen.Cache()
ramen.run(g, palette, cache=cache)          # computes
ramen.run(g, palette, cache=cache)          # nothing changed, so nothing runs
```

- NumPy and SciPy calls overlap on threads. A node marked `executor="process"`
  runs in a worker process, so pure-Python work uses another processor too,
  and large arrays reach it through shared memory.
- Nodes with side effects keep their order, and libraries that are not
  thread-safe are kept to one node at a time.
- A `Cache` re-runs a pure node only when its inputs changed.
- A Try structure catches a failing node; a With structure opens and closes
  a resource around its frame; a For loop can run its iterations in parallel.

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
  runtime/      execution: scheduler, worker processes, cache
docs/spec.md    the contracts everything builds on
examples/       hand-written graphs, run by the test suite
tests/          the test suite
benchmarks/     overhead per node and iteration, parallel speed-up, importer coverage
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
actually happens at run time on thousands of random cases. Loops and cases are
tested the same way: hundreds of random programs are run both as graphs and
as plain Python, and must agree on every value. They are then run again
with every node on a different thread. `OPENRAMEN_STRESS=1 pytest` runs the
whole suite that way.
