"""Headless execution of graphs.

Nodes run one at a time, in dependency order, in the calling thread.  The
same engine runs the frames of structures (loops, cases, subgraphs), so it is
re-entrant.  Parallel executors and caching build on the same :class:`Plan`.

Values on wires have value semantics, enforced here:

* a node that declares it writes to an input (``mutate``) gets a private copy;
* arrays reach every other input as read-only views, so an undeclared write
  cannot change what another node sees;
* a shift register that is only ever updated in place keeps one buffer for
  the whole loop.
"""

from __future__ import annotations

import copy
import enum
import re
import time
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import numpy as np

from .. import types as T
from ..check import Analysis, GraphCheckError, analyze
from ..library import NodeSpec, Registry
from ..model import FRAME_IN, FRAME_OUT, INPUTS, UNSET, Graph, Structure

__all__ = ["Plan", "Step", "StructurePlan", "Event", "RunResult", "NodeError", "Cancelled",
           "GraphCheckError", "compile_graph", "run"]

_PASS, _CONVERT, _ACCEPT = 0, 1, 2           # what a wire does to its value
_READ, _MUTATE, _IN_PLACE = 0, 1, 2          # how the receiving pin uses it
_IMMUTABLE = (type(None), bool, int, float, complex, str, bytes, PurePath)
_READ_ONLY = re.compile(r"read[- ]?only|not writ(?:e)?able", re.I)

#: Specs found to need writable arrays although they do not declare that they
#: write to them.  They are given private copies from then on.
_NEEDS_WRITABLE: set[str] = set()
#: Specs the mutation detector has seen leave their inputs alone.
_VERIFIED: set[tuple[str, str]] = set()


class Cancelled(Exception):
    """The run was stopped through its ``cancel`` callback."""


class NodeError(Exception):
    """A node failed while a graph was running.

    ``original`` is the exception raised inside the node (also available as
    ``__cause__``); ``pin`` is set when a value did not fit a pin.  Inside
    structures ``node`` is a path such as ``"loop/add_1"``, and ``context``
    says which iteration or frame was running.
    """

    def __init__(self, node: str, spec: str, original: BaseException,
                 pin: Optional[str] = None, detail: str = ""):
        self.node = node
        self.spec = spec
        self.pin = pin
        self.original = original
        self.context: list[str] = []
        what = detail or f"{type(original).__name__}: {original}"
        where = f"node {node!r} ({spec})"
        if pin:
            where += f", pin {pin!r}"
        self._message = f"{where}: {what}"
        super().__init__(self._message)

    def __str__(self) -> str:
        if not self.context:
            return self._message
        return f"{self._message} [{', '.join(self.context)}]"


@dataclass(frozen=True)
class Event:
    """Progress notification sent to a listener during a run."""

    kind: str                   # graph-started, node-started, node-finished, node-failed,
    node: Optional[str] = None  # node-warning, graph-finished, graph-failed
    duration: Optional[float] = None
    error: Optional[BaseException] = None
    message: str = ""


Listener = Callable[[Event], None]


@dataclass
class StructurePlan:
    """A structure, ready to run: its frames as plans, and what the checker
    found out about its tunnels and shift registers."""

    node: Structure
    frames: list["Plan"]
    data: dict[str, Any]
    #: Shift registers whose buffer is updated in place, one buffer per loop.
    in_place: set[str] = field(default_factory=set)


@dataclass
class Step:
    """Everything needed to execute one node."""

    node: str
    spec: NodeSpec
    #: pin, source, what the wire does, its payload, how the pin uses the value
    wired: list[tuple[str, tuple[str, str], int, Any, int]]
    literals: dict[str, Any]
    config: dict[str, Any]
    out_types: dict[str, T.Type]
    needs: frozenset[str] = field(default_factory=frozenset)  # upstream node ids
    structure: Optional[StructurePlan] = None
    boundary: str = ""           # "in" or "out" for the boundary nodes of a frame


@dataclass
class Plan:
    """A checked graph, ready to run any number of times."""

    name: str
    steps: list[Step]
    analysis: Analysis


class RunResult:
    """Values produced by a run, addressed by node id and output pin."""

    def __init__(self, values: dict[tuple[str, str], Any],
                 outputs: dict[str, tuple[str, ...]], durations: dict[str, float]):
        self.values = values
        self.durations = durations
        self._outputs = outputs

    def __getitem__(self, key: Union[str, tuple[str, str]]) -> Any:
        """``result["node"]`` (single output) or ``result["node", "pin"]``."""
        if isinstance(key, tuple):
            return self.values[key]
        pins = self._outputs[key]
        if len(pins) == 1:
            return self.values[(key, pins[0])]
        return {pin: self.values[(key, pin)] for pin in pins}

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple):
            return key in self.values
        return key in self._outputs

    def outputs(self, node: str) -> dict[str, Any]:
        """All outputs of one node, by pin name."""
        return {pin: self.values[(node, pin)] for pin in self._outputs[node]}

    def observed(self, node: str) -> dict[str, T.Type]:
        """The types the outputs of a node actually had in this run."""
        return {pin: T.type_of(self.values[(node, pin)]) for pin in self._outputs[node]}

    def __repr__(self) -> str:
        return f"<RunResult: {len(self._outputs)} nodes>"


# --------------------------------------------------------------------------
# Compiling
# --------------------------------------------------------------------------

def compile_graph(graph: Graph, registry: Optional[Registry] = None) -> Plan:
    """Check a graph and turn it into an execution plan.

    Raises :class:`~openramen.check.GraphCheckError` if the graph has errors.
    """
    analysis = analyze(graph, registry)
    analysis.raise_for_errors()
    return _plan(graph, analysis)


def _plan(graph: Graph, analysis: Analysis) -> Plan:
    steps: list[Step] = []
    needs: dict[str, frozenset[str]] = {}
    for nid in analysis.order:
        info = analysis.nodes[nid]
        node = graph.nodes[nid]
        assert info.spec is not None
        wired: list[tuple[str, tuple[str, str], int, Any, int]] = []
        upstream: set[str] = set()
        for pin, (wire, compat) in info.wires.items():
            mode, payload = _PASS, None  # type: tuple[int, Any]
            if compat.verdict is T.Verdict.COERCION:
                mode, payload = _CONVERT, compat.convert
            elif compat.verdict is T.Verdict.DYNAMIC:
                mode, payload = _ACCEPT, info.in_types[pin]
            governing = info.spec.pin_for(pin)
            use = _MUTATE if governing is not None and governing.access == "mutate" else _READ
            wired.append((pin, (wire.src, wire.src_pin), mode, payload, use))
            upstream.add(wire.src)
            upstream |= needs[wire.src]
        needs[nid] = frozenset(upstream)
        structure = None
        if isinstance(node, Structure):
            frames = [_plan(frame, frame_analysis)
                      for frame, frame_analysis in zip(node.frames, info.frames)]
            structure = StructurePlan(node, frames, info.data)
            if node.kind in ("for", "while"):
                structure.in_place = _in_place_registers(frames[0], node.registers)
        boundary = {FRAME_IN: "in", FRAME_OUT: "out"}.get(info.spec.id, "")
        steps.append(Step(nid, info.spec, wired, dict(info.literals), dict(info.config),
                          dict(info.out_types), needs[nid], structure, boundary))
    return Plan(graph.name, steps, analysis)


def _in_place_registers(frame: Plan, registers: list[str]) -> set[str]:
    """Shift registers that follow the accumulate-in-a-buffer pattern.

    That is: the register's value goes to exactly one pin, that pin belongs to
    a node which writes to it in place, and the node's output for that very
    object goes nowhere but back into the register.  Then no other node can
    ever see the buffer, and the copy per iteration can be skipped.
    """
    consumers: dict[tuple[str, str], list[tuple[Step, int]]] = {}
    for step in frame.steps:
        for index, (_pin, source, _mode, _payload, _use) in enumerate(step.wired):
            consumers.setdefault(source, []).append((step, index))
    found: set[str] = set()
    for register in registers:
        readers = consumers.get((INPUTS, register), [])
        if len(readers) != 1:
            continue
        step, index = readers[0]
        pin, source, mode, payload, use = step.wired[index]
        if use != _MUTATE or mode != _PASS or step.structure is not None or step.boundary:
            continue
        handed_on = [p.name for p in step.spec.outputs if p.aliases == pin]
        if len(handed_on) != 1:
            continue
        back = consumers.get((step.node, handed_on[0]), [])
        if len(back) != 1 or back[0][0].boundary != "out":
            continue
        back_pin, _, back_mode, _, _ = back[0][0].wired[back[0][1]]
        if back_pin != register or back_mode != _PASS:
            continue
        step.wired[index] = (pin, source, mode, payload, _IN_PLACE)
        found.add(register)
    return found


# --------------------------------------------------------------------------
# Value semantics
# --------------------------------------------------------------------------

def _fresh(value: Any) -> Any:
    """A private copy of a literal, so no node can alter the graph's own."""
    return value if isinstance(value, _IMMUTABLE) else copy.deepcopy(value)


def _private(value: Any) -> Any:
    """A copy the receiving node may freely modify."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (list, dict, set, bytearray, tuple)):
        return copy.deepcopy(value)
    return value      # immutable, or an object with reference semantics


def _frozen(value: np.ndarray) -> np.ndarray:
    """A read-only view: the node can look, but a write raises."""
    view = value.view()
    view.flags.writeable = False
    return view


def _same(a: Any, b: Any) -> bool:
    """Deep equality for the mutation detector; unknown objects count as equal."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.shape == b.shape and a.dtype == b.dtype
                and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")))
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (set, bytearray, bool, int, str, bytes, complex)):
        return bool(a == b)
    if isinstance(a, float):
        return a == b or (a != a and b != b)
    return True


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

@dataclass
class _Context:
    emit: Listener
    check_outputs: Optional[bool]
    cancel: Optional[Callable[[], bool]]
    detect_mutation: bool


def run(graph: Union[Graph, Plan], registry: Optional[Registry] = None, *,
        targets: Optional[Iterable[str]] = None,
        listener: Optional[Listener] = None,
        check_outputs: Optional[bool] = None,
        cancel: Optional[Callable[[], bool]] = None,
        detect_mutation: bool = False) -> RunResult:
    """Run a graph (or a compiled :class:`Plan`) and return its values.

    * ``targets``: only run these nodes and what they depend on.
    * ``listener``: called with an :class:`Event` as the run progresses.
    * ``check_outputs``: verify each output against its declared type.  By
      default this is done for every node that is not hand-curated.
    * ``cancel``: called before every node and loop iteration; when it
      returns true the run stops with :class:`Cancelled`.
    * ``detect_mutation``: debugging aid.  Compare the lists and dicts given
      to each node before and after the call, and fail if a node changed one
      without declaring it.  (Arrays are always protected.)
    """
    plan = graph if isinstance(graph, Plan) else compile_graph(graph, registry)
    steps = plan.steps
    if targets is not None:
        wanted = set(targets)
        known = {s.node for s in steps}
        missing = wanted - known
        if missing:
            raise KeyError(f"no node {sorted(missing)[0]!r} in graph")
        keep = set(wanted)
        for step in steps:
            if step.node in wanted:
                keep |= step.needs
        steps = [s for s in steps if s.node in keep]

    emit = listener or (lambda event: None)
    context = _Context(emit, check_outputs, cancel, detect_mutation)
    durations: dict[str, float] = {}
    emit(Event("graph-started"))
    started = time.perf_counter()
    try:
        values, _ = _run_steps(steps, context, "", None, durations)
    except BaseException as exc:
        emit(Event("graph-failed", duration=time.perf_counter() - started, error=exc))
        raise
    emit(Event("graph-finished", duration=time.perf_counter() - started))
    outputs = {s.node: tuple(p.name for p in s.spec.outputs) for s in steps}
    return RunResult(values, outputs, durations)


def _run_steps(steps: list[Step], context: _Context, path: str,
               boundary: Optional[dict[str, Any]],
               durations: Optional[dict[str, float]] = None
               ) -> tuple[dict[tuple[str, str], Any], dict[str, Any]]:
    """Run one diagram.  For a frame, ``boundary`` feeds its ``inputs`` node
    and the second result is what arrived at its ``outputs`` node."""
    values: dict[tuple[str, str], Any] = {}
    result: dict[str, Any] = {}
    emit = context.emit
    for step in steps:
        if context.cancel is not None and context.cancel():
            raise Cancelled("the run was cancelled")
        if step.boundary == "in":
            for pin, value in (boundary or {}).items():
                values[(step.node, pin)] = value
            continue
        if step.boundary == "out":
            result = _gather(step, values, path, protect=False)
            continue
        name = path + step.node
        emit(Event("node-started", name))
        t0 = time.perf_counter()
        try:
            _execute(step, values, context, path)
        except NodeError as exc:
            emit(Event("node-failed", name, time.perf_counter() - t0, exc))
            raise
        elapsed = time.perf_counter() - t0
        if durations is not None:
            durations[step.node] = elapsed
        emit(Event("node-finished", name, elapsed))
    return values, result


def _gather(step: Step, values: dict[tuple[str, str], Any], path: str,
            protect: bool = True) -> dict[str, Any]:
    """The arguments of a node: literals, and what its wires deliver."""
    spec = step.spec
    arguments = {pin: _fresh(value) for pin, value in step.literals.items()}
    copies = protect and spec.id in _NEEDS_WRITABLE
    for pin, source, mode, payload, use in step.wired:
        value = values[source]
        if mode == _CONVERT:
            try:
                value = payload(value)
            except Exception as exc:
                raise NodeError(path + step.node, spec.id, exc, pin) from exc
        elif mode == _ACCEPT:
            try:
                value = T.accept(value, payload)
            except T.TypeMismatch as exc:
                raise NodeError(path + step.node, spec.id, exc, pin, str(exc)) from exc
        if protect:
            if use == _MUTATE:
                value = _private(value)
            elif use == _READ and isinstance(value, np.ndarray):
                if copies:
                    value = value.copy()
                elif value.flags.writeable:
                    value = _frozen(value)
        arguments[pin] = value
    return arguments


def _execute(step: Step, values: dict[tuple[str, str], Any], context: _Context,
             path: str) -> None:
    spec = step.spec
    name = path + step.node
    arguments = _gather(step, values, path)
    if step.structure is not None:
        results = _run_structure(step, arguments, context, path)
    else:
        results = _call(step, arguments, context, name)
    verify = spec.tier != "curated" if context.check_outputs is None else context.check_outputs
    for pin, value in results.items():
        value = T.plain_scalar(value)
        if verify and not T.conforms(value, step.out_types[pin]):
            declared = spec.output(pin)
            origin = f" (a type taken from its {declared.source})" if (
                declared is not None and declared.source in ("docstring", "annotation")) else ""
            mismatch = T.TypeMismatch(
                f"output is declared {step.out_types[pin]}{origin} but the node returned "
                f"{T.type_of(value)}")
            raise NodeError(name, spec.id, mismatch, pin, str(mismatch))
        values[(step.node, pin)] = value


def _call(step: Step, arguments: dict[str, Any], context: _Context,
          name: str) -> dict[str, Any]:
    spec = step.spec
    config = {key: _fresh(value) for key, value in step.config.items()}
    watched: dict[str, Any] = {}
    if context.detect_mutation and (spec.id, spec.version) not in _VERIFIED:
        declared = {pin for pin, _, _, _, use in step.wired if use != _READ}
        for pin, value in arguments.items():
            governing = spec.pin_for(pin)
            writes = governing is not None and governing.access == "mutate"
            if pin not in declared and not writes and isinstance(
                    value, (list, dict, set, bytearray)):
                try:
                    watched[pin] = copy.deepcopy(value)
                except Exception:
                    pass          # cannot be snapshotted: cannot be watched
    try:
        results = spec.call(arguments, config)
    except (ValueError, TypeError, BufferError, RuntimeError) as exc:
        frozen = [pin for pin, value in arguments.items()
                  if isinstance(value, np.ndarray) and not value.flags.writeable]
        if not frozen or not _READ_ONLY.search(str(exc)):
            raise NodeError(name, spec.id, exc) from exc
        # The node needs writable arrays: either it writes to an input without
        # declaring it, or it merely refuses read-only buffers.  Private
        # copies are right in both cases.
        retry = dict(arguments)
        for pin in frozen:
            retry[pin] = arguments[pin].copy()
        try:
            results = spec.call(retry, config)
        except Exception as again:
            raise NodeError(name, spec.id, again) from again
        _NEEDS_WRITABLE.add(spec.id)
        context.emit(Event(
            "node-warning", name,
            message=f"{spec.id} needs writable arrays ({exc}); its inputs "
                    f"{', '.join(repr(p) for p in sorted(frozen))} are given private copies"))
    except Exception as exc:
        raise NodeError(name, spec.id, exc) from exc
    for pin, before in watched.items():
        if not _same(before, arguments[pin]):
            problem = RuntimeError(
                f"the node modified its input {pin!r} in place without declaring it; "
                "mark the input as mutated (in an override: mutates or passthrough)")
            raise NodeError(name, spec.id, problem, pin, str(problem))
    if context.detect_mutation:
        _VERIFIED.add((spec.id, spec.version))
    return results


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------

def _fail(name: str, spec: str, message: str, pin: Optional[str] = None) -> NodeError:
    problem = RuntimeError(message)
    return NodeError(name, spec, problem, pin, message)


def _elements(value: Any) -> Iterator[Any]:
    """What an indexing input tunnel hands out, one item per iteration."""
    if isinstance(value, (dict, str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"a {T.type_of(value)} cannot be indexed by a loop")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        raise TypeError("a 0-d array cannot be indexed by a loop")
    return iter(value)


def _run_structure(step: Step, arguments: dict[str, Any], context: _Context,
                   path: str) -> dict[str, Any]:
    plan = step.structure
    assert plan is not None
    node = plan.node
    name = path + step.node
    spec_id = step.spec.id
    converts: dict[str, T.Type] = plan.data.get("convert", {})

    def frame(index: int, boundary: dict[str, Any], what: str) -> dict[str, Any]:
        inner = f"{name}[{index}]/" if node.kind == "case" else name + "/"
        try:
            _, result = _run_steps(plan.frames[index].steps, context, inner, boundary)
        except NodeError as exc:
            exc.context.append(what)
            raise
        return result

    def finish(pin: str, value: Any) -> Any:
        target = converts.get(pin)
        if target is None:
            return value
        try:
            return T.accept(value, target)
        except T.TypeMismatch as exc:
            raise NodeError(name, spec_id, exc, pin, str(exc)) from exc

    if node.kind == "subgraph":
        result = frame(0, {t.name: arguments[t.name] for t in node.inputs}, f"in {name!r}")
        return {t.name: finish(t.name, _tunnel_value(t, result, name, spec_id))
                for t in node.outputs}

    if node.kind == "case":
        selector = arguments["selector"]
        key = selector.name if isinstance(selector, enum.Enum) else selector
        index = _matching_frame(node.cases, key)
        if index is None:
            raise _fail(name, spec_id, f"no frame handles the selector value {key!r}", "selector")
        boundary = {t.name: arguments[t.name] for t in node.inputs}
        boundary["selector"] = selector
        result = frame(index, boundary, f"frame {index} of {name!r}")
        return {t.name: finish(t.name, _tunnel_value(t, result, name, spec_id))
                for t in node.outputs}

    return _run_loop(step, arguments, context, name, frame, finish)


def _matching_frame(cases: list, key: Any) -> Optional[int]:
    default = None
    is_bool = isinstance(key, bool)
    for index, match in enumerate(cases):
        if match is None:
            default = index if default is None else default
            continue
        for entry in match:
            if isinstance(entry, tuple):
                if isinstance(key, int) and not is_bool and entry[0] <= key <= entry[1]:
                    return index
            elif isinstance(entry, bool) == is_bool and type(entry) is type(key) and entry == key:
                return index
    return default


def _tunnel_value(tunnel: Any, result: dict[str, Any], name: str, spec_id: str) -> Any:
    if tunnel.name in result:
        return result[tunnel.name]
    if tunnel.fallback is not UNSET:
        return _fresh(tunnel.fallback)
    raise _fail(name, spec_id, f"output {tunnel.name!r} received no value", tunnel.name)


def _run_loop(step: Step, arguments: dict[str, Any], context: _Context, name: str,
              frame: Callable[[int, dict[str, Any], str], dict[str, Any]],
              finish: Callable[[str, Any], Any]) -> dict[str, Any]:
    plan = step.structure
    assert plan is not None
    node = plan.node
    spec_id = step.spec.id
    register_types: dict[str, T.Type] = plan.data.get("registers", {})
    is_for = node.kind == "for"

    def into_register(register: str, value: Any) -> Any:
        wanted = register_types.get(register, T.ANY)
        if isinstance(wanted, T.AnyT) or T.conforms(value, wanted):
            return value
        try:
            return T.accept(value, wanted)
        except T.TypeMismatch as exc:
            raise NodeError(name, spec_id, exc, register,
                            f"shift register {register!r}: {exc}") from exc

    plain = {t.name: arguments[t.name] for t in node.inputs if t.mode == "plain"}
    indexed: dict[str, Iterator[Any]] = {}
    for tunnel in node.inputs:
        if tunnel.mode == "index":
            try:
                indexed[tunnel.name] = _elements(arguments[tunnel.name])
            except TypeError as exc:
                raise NodeError(name, spec_id, exc, tunnel.name) from exc
    count = arguments.get("count") if is_for else None
    if is_for and count is None and not indexed:
        raise _fail(name, spec_id, "a For loop needs a count or an indexing input", "count")

    registers: dict[str, Any] = {}
    owned: dict[str, Any] = {}        # the buffers of in-place registers
    for register in node.registers:
        value = into_register(register, arguments[register])
        if register in plan.in_place:
            value = owned[register] = _private(value)
        registers[register] = value

    last: dict[str, Any] = {}
    gathered: dict[str, list[Any]] = {t.name: [] for t in node.outputs if t.mode != "last"}
    limit = node.max_iterations
    i = 0
    while count is None or i < count:
        if context.cancel is not None and context.cancel():
            raise Cancelled("the run was cancelled")
        boundary = dict(plain)
        try:
            for pin, iterator in indexed.items():
                boundary[pin] = T.plain_scalar(next(iterator))
        except StopIteration:
            break
        for register in plan.in_place:
            if registers[register] is not owned.get(register):
                # the loop fed back some other object: take a buffer of our own again
                registers[register] = owned[register] = _private(registers[register])
        boundary.update(registers)
        boundary["i"] = i
        result = frame(0, boundary, f"iteration {i} of {name!r}")
        for register in node.registers:
            registers[register] = into_register(register, result[register])
        for tunnel in node.outputs:
            if tunnel.name not in result:
                continue
            if tunnel.mode == "last":
                last[tunnel.name] = result[tunnel.name]
            else:
                gathered[tunnel.name].append(result[tunnel.name])
        i += 1
        if not is_for:
            stop = result.get("stop")
            if not isinstance(stop, (bool, np.bool_)):
                raise _fail(name, spec_id,
                            f"'stop' must be a Bool, got {T.type_of(stop)}", "stop")
            if stop:
                break
            if limit is not None and i >= limit:
                raise _fail(name, spec_id,
                            f"the loop did not stop within {limit} iterations", "stop")

    outputs: dict[str, Any] = {}
    for tunnel in node.outputs:
        mode, element, _out_type = plan.data["outputs"][tunnel.name]
        if mode == "last":
            if tunnel.name in last:
                value = last[tunnel.name]
            elif tunnel.fallback is not UNSET:
                value = _fresh(tunnel.fallback)
            else:
                raise _fail(
                    name, spec_id,
                    f"the loop ran zero times, so output {tunnel.name!r} has no value; "
                    "give it a fallback or use a shift register", tunnel.name)
        else:
            try:
                value = _combine(mode, gathered[tunnel.name], element, _out_type)
            except Exception as exc:
                raise NodeError(name, spec_id, exc, tunnel.name) from exc
        outputs[tunnel.name] = finish(tunnel.name, value)
    outputs.update(registers)
    return outputs


_STACK_DTYPE = {T.BOOL: "bool", T.INT: "int64", T.FLOAT: "float64", T.COMPLEX: "complex128"}


def _empty(element: T.Type, extra_dimensions: int) -> np.ndarray:
    """The array a collecting tunnel gives when the loop ran zero times."""
    dtype, ndim = "float64", 1
    if isinstance(element, T.ArrayT):
        dtype = element.dtype if isinstance(element.dtype, str) else "float64"
        ndim = (element.ndim if isinstance(element.ndim, int) else 0) + extra_dimensions
    return np.empty((0,) * max(ndim, 1), dtype=dtype)


def _combine(mode: str, items: list[Any], element: T.Type, built: T.Type) -> Any:
    """Build the value of a collecting output tunnel from one item per
    iteration.  ``built`` is the type the checker promised."""
    if mode == "collect":
        return list(items)
    if mode == "stack":
        if element in _STACK_DTYPE:
            return np.array(items, dtype=_STACK_DTYPE[element])
        if not items:
            return _empty(element, 1)
        stacked = np.stack([np.asarray(item) for item in items])
        if stacked.dtype == object:
            raise TypeError("these values cannot be stacked into an array")
        return stacked
    # concat: lists or arrays, end to end
    as_list = isinstance(built, T.ListT) or (
        not isinstance(built, T.ArrayT) and all(isinstance(i, list) for i in items))
    if as_list:
        joined: list[Any] = []
        for item in items:
            if not isinstance(item, list):
                raise TypeError(f"cannot concatenate a {T.type_of(item)} to a list")
            joined.extend(item)
        return joined
    if not items:
        return _empty(built, 0)
    return np.concatenate(items)
