# OpenRamen core specification

Version 2 of the three contracts everything else builds on: the **type
syntax**, the **NodeSpec** record, and the **graph file format**. The code in
`src/openramen` implements exactly this, and the tests in `tests/` hold it to
it.

## 1. Types

Every pin has a type. Types are written as text in files and documentation,
and parsed by `openramen.types.parse_type`.

| Family | Syntax | Notes |
| --- | --- | --- |
| Scalar | `Bool` `Int` `Float` `Complex` `Str` `Bytes` `Path` `None` | Scalars travel as Python objects. NumPy scalars are converted when a node returns them |
| Array | `Array[float64, 2]` | dtype name, number of dimensions. `_` means "not known statically": `Array[_, 2]`, `Array[uint8, _]`. Plain `Array` is `Array[_, _]` |
| List | `List[Int]` | `List` is `List[Any]` |
| Tuple | `Tuple[Int, Str]`, `Tuple[Int, ...]`, `Tuple[]` | Fixed length, or any length of one type |
| Dict | `Dict[Str, Float]` | |
| Union | `Union[Int, Str]`, `Optional[Int]` | Order does not matter; nested unions flatten |
| Enum | `Enum[pkg.Color: red, green]` | Qualified class name and member names |
| Record | `Record[pkg.Point: x=Float, y=Float]` | Dataclass, NamedTuple or TypedDict |
| Object | `Object[numpy.random:Generator]` | Opaque class instance, reference semantics. `module:QualName` |
| Callable | `Callable[[Int, Str], Bool]`, `Callable[..., Bool]` | |
| Iterator | `Iterator[Str]` | |
| Variable | `~T`, also in array slots: `Array[~D, ~N]` | Solved per node from what is wired |
| Dynamic | `Any` | Unknown until run time |

### Connecting two pins

`check(src, dst)` gives one of four verdicts.

| Verdict | Meaning | At run time |
| --- | --- | --- |
| `EXACT` | Fits as it is | Value is passed through |
| `COERCION` | Fits after a lossless conversion | The conversion is applied on the wire |
| `DYNAMIC` | Cannot be decided statically | `accept(value, dst)` checks, and converts if a coercion applies |
| `INCOMPATIBLE` | Never fits | The graph does not run |

The complete list of coercions:

- `Int` to `Float`, `Int` to `Complex`, `Float` to `Complex`
- `Path` to `Str` and `Str` to `Path`
- A tuple to `List[T]`, when every element fits `T`
- `List[T]` or `Tuple[T, ...]` to `Iterator[T]`
- An array to an array of a wider dtype, where NumPy calls the cast *safe*
- Any of the above inside `List`, `Tuple`, `Dict` values and `Iterator`

Deliberately **not** coercions: `Bool` to `Int`, `Float` to `Int`, a scalar to
an array, a list to an array. Each needs a visible conversion node.

Further rules:

- `Any` flows into everything as `DYNAMIC`; everything flows into `Any` as `EXACT`.
- A union source is `EXACT` if every option is, `INCOMPATIBLE` if none fits, `DYNAMIC` otherwise.
- A NamedTuple record flows into `Tuple` and `List` pins; a TypedDict record into `Dict` pins.
- Structural types flow into an `Object` pin when their Python class is a subclass of the pin's class.

### Generic nodes

A node whose pins mention `~T` is solved by unification. A variable seen
twice binds to the *join* of both types: `Int` and `Float` give `Float`,
`Int` and `None` give `Optional[Int]`, `Int` and `Str` are a `type-conflict`
error. Unbound variables become `Any`.

Nodes whose output type follows no simple pattern (arithmetic on scalars and
arrays, array reductions) carry an `infer` hook. The base palette's hooks
apply the real operation to tiny sample values, so the static type is what
Python and NumPy actually return.

## 2. NodeSpec

One immutable record per kind of node (`openramen.library.NodeSpec`).

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | dotted name | Unique, e.g. `base.add` |
| `inputs` | tuple of PinSpec | Wirable inputs, in call order |
| `outputs` | tuple of PinSpec | One output takes the whole return value; several unpack a tuple or a mapping |
| `config` | tuple of PinSpec | Parameters set on the node, never wired |
| `fn` or `target` | callable, or `"module:qualname"` | What runs. A target is imported on first use |
| `summary`, `description`, `category` | text | For the palette and context help |
| `pure` | bool | Same inputs always give the same outputs |
| `thread_safe` | bool | May run concurrently with other nodes |
| `executor` | `inline`, `thread`, `process` | Where it should run |
| `tier` | `auto`, `annotated`, `curated` | How much the spec can be trusted. Outputs of non-curated nodes are checked against their declared type on every run |
| `infer` | callable | Optional type hook: `infer(input_types, config) -> {output: Type}` |

PinSpec fields: `name`, `type`, `default`, `doc`, `access` (`read` or
`mutate`), `aliases` (outputs: the input it shares memory with),
`passthrough` (outputs: the value *is* the aliased input, as the call left
it), `widget`, `kind`, `hidden`, and `source` (where the type came from:
`annotation`, `docstring`, `default`, `override` or `observed`).

`default` is a value, or one of two markers: `MISSING` (the input is
required) and `DEFAULT` (optional, but the default value is not known, so the
input is only passed when it is wired).

`kind` says how an input reaches the function, in the order Python allows:

| Kind | Python parameter | On a node |
| --- | --- | --- |
| `positional` | positional-only | A pin |
| `keyword` | ordinary | A pin |
| `var_positional` | `*args` | Grows: pins `args_0`, `args_1`, ... numbered without gaps |
| `keyword_only` | after `*` | A pin |
| `var_keyword` | `**kwargs` | Grows: any other identifier becomes a keyword pin |

Arguments are passed by position for as long as no pin is left out, then by
keyword. The type of a growable pin is the type of one element.

`access`, `aliases` and `passthrough` drive the copy rules (section 4).
`thread_safe`, `executor` and `pure` decide how a node is scheduled (section 4).
`widget` and `hidden` are recorded now and acted on by the front panel.

## 3. Graph file

A graph is one UTF-8 JSON file, written with one node and one wire per line
and everything sorted, so that files diff and merge under git.

```json
{
  "format": "openramen.graph",
  "version": 1,
  "name": "demo",
  "nodes": [
    {"id": "a", "spec": "base.constant", "config": {"value": 2}},
    {"id": "sum", "spec": "base.add", "values": {"b": 0.5}, "pos": [160.0, 40.0], "label": "a + 0.5"}
  ],
  "wires": [
    ["a", "out", "sum", "a"]
  ]
}
```

- `imports` (optional): Python modules whose functions the graph uses as
  nodes, e.g. `["numpy", "scipy.ndimage"]`. They are imported when the graph
  is checked or run, never when the file is loaded.
- `nodes[].id`: an identifier, unique in the graph. `spec`: a NodeSpec id.
- `values`: literals for unwired inputs. `config`: configuration values.
- `pos` and `label` are optional and only matter to the editor.
- `wires[]`: `[source node, output pin, destination node, input pin]`.
- `version` is 1 for a graph without structures and 2 for one with them, so
  that older readers refuse what they cannot run. A reader must refuse a
  `version` newer than it knows, and ignore unknown keys.

Structural rules, enforced on load and on every edit: ids are unique, wires
join existing nodes, an input pin has at most one wire, there are no cycles.

### Structures

A structure is a node that holds diagrams of its own, called frames. Its
`spec` is `structure.<kind>` and it carries a `structure` object:

```json
{
  "id": "loop", "spec": "structure.for", "values": {"total": 0},
  "structure": {
    "kind": "for",
    "inputs": [{"name": "x", "mode": "index"}],
    "outputs": [{"name": "squares", "mode": "collect"}],
    "registers": ["total"],
    "frames": [{"nodes": [], "wires": []}]
  }
}
```

| Kind | Frames | Runs |
| --- | --- | --- |
| `subgraph` | 1 | Once |
| `for` | 1 | Once per element of its indexing inputs, or `count` times. The shortest wins when both are given; a count below 1 means no iterations |
| `while` | 1 | Until its frame delivers `true` to `stop`. The test comes after the iteration, so the frame runs at least once. `max_iterations` (optional) turns a loop that never stops into an error |
| `case` | 1 or more | Exactly one frame, chosen by `selector` |
| `try` | 2 | Frame 0, the body. If a node in it fails, the rest of the body is skipped, its outputs are dropped, and frame 1, the handler, runs instead |
| `with` | 1 | Once, between entering and leaving the context manager wired to `resource` |

Every frame is a graph (`nodes`, `wires`) containing two boundary nodes that
cannot be removed: `inputs` (spec `frame.inputs`) hands values to the frame,
`outputs` (spec `frame.outputs`) collects what leaves it. Nodes inside a frame
can only be wired to other nodes of the same frame; everything else crosses
the boundary through a tunnel or a shift register. From the outside the
structure is an ordinary node with these pins:

| Pin | Outside | Inside the frame |
| --- | --- | --- |
| Input tunnel, mode `plain` | Input | `inputs.<name>`: the same value in every iteration and frame |
| Input tunnel, mode `index` (For only) | Input: a list, tuple, iterator or array | `inputs.<name>`: one element per iteration. Arrays are walked along their first axis |
| Output tunnel, mode `last` | Output: the value from the final iteration, or from the frame that ran | `outputs.<name>` |
| Output tunnel, `collect`, `stack`, `concat` (loops only) | Output: one value per iteration as a list, stacked into an array with a new first axis, or joined end to end | `outputs.<name>` |
| Shift register (loops only) | Input: the starting value. Output: the final value | `inputs.<name>`: the value from the previous iteration. `outputs.<name>`: the value for the next one |
| `count` (For) | Optional input when an indexing tunnel exists | |
| `i` (loops) | | `inputs.i`: the iteration number, from 0 |
| `stop` (While) | | `outputs.stop`: a Bool |
| `selector` (Case) | Input: Bool, Int, Str or an Enum | `inputs.selector` |
| `failed`, `error` (Try) | Outputs: whether the body failed (Bool), and the error message (Str, empty if it did not) | |
| `error`, `error_type` (Try) | | In the handler only: `inputs.error`, the message, and `inputs.error_type`, the name of the exception class |
| `resource` (With) | Input: a context manager | `inputs.resource`: what entering it returned |

Tunnel objects have `name` and `mode`, and optionally `type` (a declared
type; values are converted to it) and `fallback` (output tunnels in mode
`last`: the value when no iteration ran, or when the chosen case frame leaves
the output unwired). The names `i`, `count`, `stop` and `selector` are
reserved everywhere; `error`, `error_type` and `failed` in a Try, `resource`
in a With. Input names are unique among inputs and registers, output names
among outputs and registers.

A For loop without shift registers may carry `"parallel": true`: its
iterations are independent, so they may run at the same time. Results keep
their order.

An output of a Try is wired in both frames, or has a fallback, and takes the
join of the two types, like a case structure. A Try catches the failure of a
node (`NodeError`), not Stop. A With structure always leaves its context
manager, passing it the node's exception if the frame failed. If the context
manager swallows the error, the outputs take their fallbacks.

A case structure has `cases`, one entry per frame: a list of values (and, for
integers, inclusive ranges written `{"range": [lo, hi]}`), or `null` for the
default frame. Enum selectors are matched by member name. `true` and `1` are
different values. The first frame that matches runs; the default frame runs
when none does.

Types flow through structures:

- An indexing tunnel delivers the element type: `List[Float]` gives `Float`,
  `Array[uint8, 3]` gives `Array[uint8, 2]`, `Array[float64, 1]` gives `Float`.
- `collect` of `T` is `List[T]`. `stack` of `Float` is `Array[float64, 1]`,
  of `Array[uint8, 2]` is `Array[uint8, 3]`. `concat` keeps the list or array type.
- A shift register starts with the type of its starting value. If the loop
  feeds back something wider (`Int` in, `Float` back), the register takes the
  wider type and the starting value is converted on the way in; unrelated
  types are an error. This is solved before anything runs.
- The output of a case structure is the join of what its frames deliver.

When a loop runs zero times, shift registers pass their starting value
through, collecting outputs are empty (an empty array has the promised dtype
and number of dimensions), and a `last` output takes its fallback. Without a
fallback that is an error at run time.

### Literals

JSON values mean themselves. Other Python values are one-key objects:

| Tag | Example | Value |
| --- | --- | --- |
| `$float` | `{"$float": "nan"}` | `nan`, `inf`, `-inf` |
| `$complex` | `{"$complex": [1.0, -2.0]}` | Complex number |
| `$bytes` | `{"$bytes": "AAE="}` | Bytes, base64 |
| `$path` | `{"$path": "a/b.tif", "flavour": "posix"}` | Path; `flavour` is `posix` or `windows` |
| `$tuple` | `{"$tuple": [1, "a"]}` | Tuple |
| `$dict` | `{"$dict": [[1, "a"]]}` | Dict with non-text keys, or keys starting with `$` |
| `$ndarray` | `{"$ndarray": {"dtype": "uint8", "shape": [2], "data": [1, 2]}}` | Numeric or boolean array |

Anything else cannot be stored as a literal and must come from a node.

## 4. Checking and running

`analyze(graph)` never raises on a broken graph. It returns every problem at once:

| Code | Meaning |
| --- | --- |
| `bad-import` | A module listed in the graph's `imports` cannot be imported |
| `unknown-spec` | The node's spec id is not registered |
| `unknown-pin`, `unknown-config` | A wire, literal or config value names a pin the spec does not have |
| `missing-input`, `missing-config` | A required pin has no wire and no value |
| `variadic-gap` | Grown pins `args_0`, `args_1`, ... are not numbered consecutively |
| `bad-literal`, `bad-config` | A stored value does not fit its pin |
| `type-conflict` | A type variable would need two unrelated types |
| `type-mismatch` | A wire is `INCOMPATIBLE`, or `stop` is not a Bool |
| `infer-failed` (warning) | A type hook crashed; the declared type is used |
| `cycle` | The diagram has a cycle. The model refuses to create one, so this is a safety net |
| `bad-frame` | A structure is malformed: a frame lacks its `inputs` or `outputs` node or lists `imports`, the number of frames is wrong, a tunnel has a mode its structure does not allow or a type that cannot be parsed, or two pins share a name |
| `bad-tunnel` | An indexing input receives something that has no elements, or a `concat` output receives something that is neither a list nor an array |
| `no-count` | A For loop has neither a count nor an indexing input |
| `unwired-tunnel`, `unwired-register`, `unwired-stop` | Nothing is wired to that pin of the `outputs` node, and (tunnels) there is no fallback |
| `register-conflict` | A shift register would have to hold two unrelated types |
| `tunnel-conflict` | The frames of a case structure, or a loop and its fallback, deliver unrelated types to one output |
| `bad-selector` | The selector is not a Bool, Int, Str or Enum |
| `bad-case`, `duplicate-case` | A case value does not fit the selector's type, or is handled by two frames |
| `no-default-case` | Some selector values are handled by no frame |

Problems inside a structure name the node by its path: `loop/add` is node
`add` in the frame of `loop`, `choose[1]/add` is in frame 1 of the case
structure `choose`.

`run(graph)` refuses a graph with errors, then executes its nodes. With one
worker that is in dependency order; among nodes that are ready, the smallest
id goes first, so the order never depends on editing history. With more
workers, independent nodes overlap (see Parallel execution). Guarantees:

- A node runs only after everything it reads from has finished.
- Literals and config values reach a node as private copies.
- A failure raises `NodeError` naming the node (by path inside structures), with the original exception as its cause. `context` lists where it happened, innermost first: `["iteration 2 of 'loop'"]`. Nothing downstream runs.
- A listener receives `graph-started`, `node-started`, `node-finished`, `node-failed`, `node-warning`, `node-cached`, `graph-finished` or `graph-failed`.
- `run(..., cancel=fn)` calls `fn` before every node and every loop iteration and stops with `Cancelled` when it returns true.
- Only the chosen frame of a case structure runs.

### Parallel execution

Nodes that do not depend on each other may run at the same time. Nobody has
to ask for it: `run` starts every node as soon as its inputs are there.

- `run(..., workers=N)` limits how many nodes run at once. The default is the
  number of processors, at least 2 and at most 8. `workers=1` runs one node
  at a time, in the order of the node ids.
- The value semantics above are what make this safe: a value shared by two
  branches is read-only, and a node that writes has its own copy.
- Nodes with side effects (`pure: false`: files, clocks, random numbers) never
  overtake each other. Within a diagram they run one after the other, in the
  order they would have with one worker. A structure that contains such a
  node counts as one. Wire one into the other to force a different order.
- A spec with `thread_safe: false` shares one lock with every other node of
  its library (the first part of the spec id), so they run one at a time.
- An object with reference semantics (an `Object` wire) is used by one node at
  a time.
- When a node fails, nothing new starts in its diagram (so an independent
  node may or may not have run), the nodes already running finish, and
  structures still running inside the diagram stop at their next node. If several nodes fail at once, the one that comes first
  in the one-worker order is reported.
- A listener is called with one event at a time, but the events of nodes
  that run together arrive in no fixed order. `cancel` may be called from
  several threads.

Where a node runs is the `executor` of its spec:

| Executor | Runs | For |
| --- | --- | --- |
| `inline` | In the thread that schedules the diagram | Cheap nodes: the base palette |
| `thread` (the default) | On a thread pool, when something else can run meanwhile | Library calls. NumPy, SciPy and file I/O release the interpreter lock in their heavy operations |
| `process` | In a worker process | Pure-Python work that should use another processor, and libraries that crash or cannot share a process |

The runtime measures every node. One that proves cheaper than 0.2 ms is run
in the scheduling thread from then on, and a diagram in which no two heavy
nodes could ever overlap is run in order, with no scheduling at all. A chain
of nodes therefore costs what it did before.

Worker processes are started on first use (`openramen.runtime.workers.start()`
starts them ahead of time) and kept. Arrays of 256 kB or more travel through
shared memory: an array is copied once on its way in and once on its way
out, and an array that came out of a worker goes into the next one without a
copy. Such arrays are read-only in the main process. A node whose pins have
`Object`, `Callable` or `Iterator` types stays in the main process, and so does
a node whose function or arguments turn out not to be picklable (reported as
a `node-warning`). If a worker dies, the node fails with a `NodeError` and a
new worker takes its place. Workers are started with the `spawn` method, so a
script that uses process nodes needs the usual `if __name__ == "__main__":`
guard around the code that runs the graph.

### Stop

`start(graph, ...)` checks the graph, runs it on a background thread and
returns a `Run`: `cancel()` asks it to stop, `wait(timeout)` and `done` tell
whether it is over, `result(timeout)` returns the values or raises what
stopped the run. Stop takes effect before the next node and before every loop
iteration. A node that is already running finishes first, except in a worker
process, which is killed.

### Caching

`run(..., cache=Cache(max_bytes=...))` keeps the outputs of pure nodes. A node
whose inputs are what they were in an earlier run with the same cache is not
run again; the listener receives `node-cached` for it.

- Only the top level of a graph is looked up. A structure is cached as a
  whole if everything inside it is pure.
- "The same inputs" means equal content for constants, literals, numbers and
  short text, and "produced by the same node from the same inputs" for
  everything else. A node with side effects always runs; what follows it is
  reused only when it returns a number or short text equal to last time.
- A node whose configuration holds something with no stable content (an open
  file, a lock) is not cached.
- When the memory budget is exceeded, the values used least recently are
  dropped, and computed again when next needed. Nothing downstream of them
  is lost by that.
- Arrays that have been through a cache are read-only: they are shared
  between runs.

### Value semantics

A wire carries a value, not a reference: no node can change what another node
sees, however the wire branches. The runtime enforces this so that copies are
made only where a node writes.

| A node input that | Receives |
| --- | --- |
| Is declared `mutate` | A private copy (arrays by `copy()`; lists, dicts, sets, tuples and bytearrays deeply). The node may do with it what it likes |
| Reads an array | A read-only view of the same memory. No copy |
| Reads anything else | The object itself |

An output marked `passthrough` hands the written copy on, which is how
functions that work in place and return nothing (`numpy.fill_diagonal`,
`numpy.put`) become nodes with an output. For arrays its type is the type
that arrived, since writing in place changes neither dtype nor rank.

Two things catch a spec that is wrong about this:

- **Arrays.** A node that writes to an array it only declared to read gets an
  error from NumPy, because the view is read-only. The runtime then runs the
  node again on private copies, reports a `node-warning`, and from then on
  gives every node of that spec copies up front. Other readers never see the
  write. (The node did start once on the views: declare the mutation to avoid that.)
- **Lists, dicts, sets and bytearrays** cannot be made read-only. `run(...,
  detect_mutation=True)`, or `openramen run --detect-mutation`, compares them
  before and after each call and fails with a `NodeError` naming the node and
  the pin if one changed without being declared. It costs a deep copy per
  input, so it is a debugging switch, not the default.

Arrays in a `RunResult` can be such read-only views. Copy one before changing it.

A shift register that is only ever updated in place keeps a single buffer for
the whole loop instead of copying it every iteration. This applies when the
register's value goes to exactly one pin, that pin is declared `mutate`, and
the output that aliases it goes back to the same register and nowhere else.
The starting value is still copied once, so the loop never writes into
something the rest of the diagram can see.

## 5. Importing Python functions

`registry.import_module("numpy")` makes every public function of a module a
node with the id `<module>.<function>`. Specs are built lazily, the first
time each is used. The importer never calls the functions it imports;
importing the module itself runs that module's top-level code, as `import`
does.

For each function the importer needs a signature and types.

**Signature**, first that works: `inspect.signature`; a `name(a, b=1)` line
at the top of the docstring; the structure of a NumPy ufunc; a `signature`
given in an override.

**Type of an input**, first that gives one:

1. An override (see below).
2. The annotation.
3. The docstring, in numpydoc, Google or reST style. A phrase is only used
   when all of it is understood: `int or tuple of ints` gives a type, `int or
   MyClass` gives `Any`. A type that is too narrow would reject correct wiring.
4. A default that is a boolean, number or text.
5. Otherwise `Any`.

Whatever the source, a pin always accepts its own default: `x : int,
optional` with `x=None` is `Optional[Int]`.

**Outputs.** One output named `out` takes the whole result, unless the
Returns section lists several named values that are always returned: then
each is a pin. Documented arrays are typed "array or scalar", because NumPy
returns a scalar for 0-d results. ufuncs get exact types by sampling.

**Flags that add results** (`return_counts`, `retstep`, `full_output`, ...)
become configuration instead of pins, so the checker knows their value: with
all flags off the output has the documented type, otherwise it is `Any`.

**Heuristics**, all correctable by an override:

- Optional keyword-only parameters, and plumbing such as `out`, `where`,
  `order`, start hidden.
- An `out` parameter, and the first input of a function whose summary says
  "in-place", are marked `mutate`.
- Functions in a `random` module, or named like `save...`, `load...`,
  `read...`, are marked impure.

Imported nodes have tier `auto`. Their outputs are checked against the
declared type on every run; a mismatch names the node and the pin.

### Override files

A TOML file per library corrects what introspection gets wrong. Tables are
named by function, relative to the library: `[histogram]`, `["linalg.norm"]`.
OpenRamen bundles such "node packs" in `src/openramen/packs/`; your own file
goes on top with `import_module(..., overrides="mine.toml")`.

| Key | Meaning |
| --- | --- |
| `types` | Table of pin name to type text, inputs and outputs alike |
| `outputs` | List of output pins, `"name"` or `"name: Type"`. Replaces what was found; `[]` means no outputs |
| `config` | Parameters that are configuration, not pins |
| `hidden`, `shown` | Force optional inputs to start hidden or visible |
| `mutates` | Inputs the function writes to |
| `passthrough` | Inputs the function writes to, each of which also becomes an output of the same name carrying the written value. Combine with `outputs = []` when the function returns nothing |
| `aliases` | Table of output to the input it shares memory with |
| `category`, `summary`, `pure`, `thread_safe`, `executor` | The NodeSpec fields of the same name |
| `signature` | The signature to use, as text: `"arange(start, stop=None, step=1)"`. For functions whose own cannot be read |
| `skip` | `true` leaves the function out |
| `function`, `presets` | A variant: this table is a node of its own that calls `function` with the arguments in `presets` fixed |

A node touched by an override has tier `annotated`.

### Observed types

After a run, `result.observed(node)` gives the types a node's outputs really
had, and `pin_outputs(registry, spec_id, types)` makes them the declared
types. From then on they are checked on every run.

## 6. Not specified yet

The front panel and the editor, which get their sections when their milestone
lands.

Still open in running: moving a node between executors by itself when it
proves slow or holds the interpreter lock, caching inside structures,
error wires as an alternative to Try, and an event structure.

Still open in structures: a conditional `collect` (keep only some
iterations), a Script node, and exporting a graph as plain Python.

Still open in the importer: classes and their methods, reading stub files
(`.pyi`), a disk cache of built specs, and flagging nodes whose function
changed between library versions.
