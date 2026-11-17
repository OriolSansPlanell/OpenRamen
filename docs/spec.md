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

PinSpec fields: `name`, `type`, `default`, `doc`, `access` (`read` or
`mutate`), `aliases` (outputs: the input it shares memory with), `widget`,
`kind`, `hidden`, and `source` (where the type came from: `annotation`,
`docstring`, `default`, `override` or `observed`).

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

- `imports` (optional): Python modules whose functions the graph uses as
  nodes, e.g. `["numpy", "scipy.ndimage"]`. They are imported when the graph
  is checked or run, never when the file is loaded.
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
| `bad-import` | A module listed in the graph's `imports` cannot be imported |
| `unknown-spec` | The node's spec id is not registered |
| `unknown-pin`, `unknown-config` | A wire, literal or config value names a pin the spec does not have |
| `missing-input`, `missing-config` | A required pin has no wire and no value |
| `variadic-gap` | Grown pins `args_0`, `args_1`, ... are not numbered consecutively |
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

Structures (loops, cases, subgraphs), copy-on-branch rules, parallel
executors, caching and the front panel. Each gets its section here when its
milestone lands. Until copy rules exist, a node that modifies a wired input
in place can affect other readers of that value.

Still open in the importer: classes and their methods, reading stub files
(`.pyi`), a disk cache of built specs, and flagging nodes whose function
changed between library versions.
