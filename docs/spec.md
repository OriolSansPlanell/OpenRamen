# OpenRamen core specification

Version 1 of the three contracts everything else builds on: the **type
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

PinSpec fields: `name`, `type`, `default` (`MISSING` means required), `doc`,
`access` (`read` or `mutate`), `aliases` (outputs: the input it shares memory
with), `widget`, `kind` (`keyword` or `positional`), `hidden`.

`access`, `aliases`, `thread_safe`, `executor`, `widget` and `hidden` are
recorded now and acted on by later milestones (copy rules, scheduler, panel).

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

- `nodes[].id`: an identifier, unique in the graph. `spec`: a NodeSpec id.
- `values`: literals for unwired inputs. `config`: configuration values.
- `pos` and `label` are optional and only matter to the editor.
- `wires[]`: `[source node, output pin, destination node, input pin]`.
- A reader must refuse a `version` newer than it knows, and ignore unknown keys.

Structural rules, enforced on load and on every edit: ids are unique, wires
join existing nodes, an input pin has at most one wire, there are no cycles.

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
| `unknown-spec` | The node's spec id is not registered |
| `unknown-pin`, `unknown-config` | A wire, literal or config value names a pin the spec does not have |
| `missing-input`, `missing-config` | A required pin has no wire and no value |
| `bad-literal`, `bad-config` | A stored value does not fit its pin |
| `type-conflict` | A type variable would need two unrelated types |
| `type-mismatch` | A wire is `INCOMPATIBLE` |
| `infer-failed` (warning) | A type hook crashed; the declared type is used |

`run(graph)` refuses a graph with errors, then executes nodes in dependency
order; among nodes that are ready, the smallest id goes first, so the order
never depends on editing history. Guarantees:

- A node runs only after everything it reads from has finished.
- Literals and config values reach a node as private copies.
- A failure raises `NodeError` naming the node, with the original exception as its cause. Nothing downstream runs.
- A listener receives `graph-started`, `node-started`, `node-finished`, `node-failed`, `graph-finished` or `graph-failed`.

## 5. Not specified yet

Structures (loops, cases, subgraphs), copy-on-branch rules, parallel
executors, caching, the importer and the front panel. Each gets its section
here when its milestone lands. Until copy rules exist, a node that modifies a
wired input in place can affect other readers of that value.
