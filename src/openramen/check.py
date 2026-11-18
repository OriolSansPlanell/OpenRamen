"""Static analysis of a graph: pins, types, missing inputs.

:func:`analyze` never raises for a broken graph.  It returns an
:class:`Analysis` whose ``problems`` list says what is wrong; a graph is
runnable when it has no problem of severity ``"error"``.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any, Optional

from . import types as T
from .library import DEFAULT, MISSING, NodeSpec, PinSpec, Registry, default_registry
from .library.sampling import scalar_of
from .model import (
    FRAME_IN, FRAME_OUT, INPUTS, OUTPUTS, UNSET, CycleError, Graph, Structure, Wire,
)

__all__ = ["Problem", "NodeInfo", "Analysis", "analyze", "GraphCheckError", "registry_for"]


@dataclass(frozen=True)
class Problem:
    severity: str            # "error" or "warning"
    code: str
    message: str
    node: Optional[str] = None
    pin: Optional[str] = None

    def __str__(self) -> str:
        where = ""
        if self.node:
            where = f" [{self.node}{'.' + self.pin if self.pin else ''}]"
        return f"{self.severity}: {self.message}{where}"


class GraphCheckError(Exception):
    """Raised when a graph with errors is about to be run."""

    def __init__(self, problems: list[Problem]):
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        count = len(self.problems)
        super().__init__(f"graph has {count} error{'s' if count != 1 else ''}:\n{lines}")


@dataclass
class NodeInfo:
    """What the checker found out about one node."""

    spec: Optional[NodeSpec]
    in_types: dict[str, T.Type] = field(default_factory=dict)    # resolved pin types
    out_types: dict[str, T.Type] = field(default_factory=dict)
    wires: dict[str, tuple[Wire, T.Compat]] = field(default_factory=dict)
    literals: dict[str, Any] = field(default_factory=dict)       # after coercion
    config: dict[str, Any] = field(default_factory=dict)         # defaults filled in
    #: Type each input really receives (``None``: nothing arrives).
    received: dict[str, Optional[T.Type]] = field(default_factory=dict)
    #: Structures only: the analysis of each frame, and what the runtime needs
    #: to know about tunnels and shift registers.
    frames: list["Analysis"] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Analysis:
    order: list[str]
    nodes: dict[str, NodeInfo]
    problems: list[Problem]
    #: For the frame of a structure: the type arriving at each pin of its
    #: ``outputs`` node (``None``: unwired).
    frame_out: dict[str, Optional[T.Type]] = field(default_factory=dict)

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def out_type(self, node: str, pin: str) -> T.Type:
        """Resolved type of an output pin (``Any`` when unknown)."""
        info = self.nodes.get(node)
        return T.ANY if info is None else info.out_types.get(pin, T.ANY)

    def wire_verdict(self, node: str, pin: str) -> Optional[T.Verdict]:
        """Verdict of the wire arriving at an input pin, if there is one."""
        info = self.nodes.get(node)
        if info is None or pin not in info.wires:
            return None
        return info.wires[pin][1].verdict

    def raise_for_errors(self) -> None:
        if self.errors:
            raise GraphCheckError(self.errors)


_registries: dict[tuple[str, ...], Registry] = {}


def registry_for(graph: Graph) -> Registry:
    """The registry a graph asks for: the base palette plus its ``imports``.

    This imports the listed Python modules, which runs their top-level code.
    """
    key = tuple(sorted(set(graph.imports)))
    registry = _registries.get(key)
    if registry is None:
        registry = default_registry()
        for module in key:
            registry.import_module(module)
        _registries[key] = registry
    return registry


def _registry(registry: Optional[Registry], graph: Graph) -> Registry:
    return registry if registry is not None else registry_for(graph)


def analyze(graph: Graph, registry: Optional[Registry] = None) -> Analysis:
    """Check a graph and resolve the type of every pin."""
    try:
        registry = _registry(registry, graph)
    except Exception as exc:        # a module named in the graph's imports is unusable
        return Analysis([], {}, [Problem("error", "bad-import", str(exc))])
    problems: list[Problem] = []
    analysis = _Checker(registry).graph(graph, None, "", problems)
    analysis.problems = problems
    return analysis


def _nothing(*args: Any, **kwargs: Any) -> None:
    """Placeholder callable of the specs synthesised for structures."""


def element_type(t: T.Type) -> Optional[T.Type]:
    """Type of one element when a value of type ``t`` is indexed by a loop.

    ``None`` means the type cannot be indexed at all.
    """
    if isinstance(t, (T.AnyT, T.VarT)):
        return T.ANY
    if isinstance(t, (T.ListT, T.IteratorT)):
        return t.elem
    if isinstance(t, T.TupleT):
        return T.union(*t.elems) if t.elems else T.ANY
    if isinstance(t, T.ArrayT):
        if t.ndim == 0:
            return None
        scalar = scalar_of(t.dtype) if isinstance(t.dtype, str) else T.ANY
        if t.ndim == 1:
            return scalar
        dtype = t.dtype if isinstance(t.dtype, str) else None
        if isinstance(t.ndim, int):
            return T.ArrayT(dtype, t.ndim - 1)
        return T.union(T.ArrayT(dtype, None), scalar)
    if isinstance(t, T.UnionT):
        parts = [element_type(o) for o in t.options]
        known = [p for p in parts if p is not None]
        return T.union(*known) if known else None
    return None


def concatenated_type(elem: T.Type) -> Optional[T.Type]:
    """Type a ``concat`` output builds by joining values of type ``elem`` end
    to end, or ``None`` if they are not all lists or all arrays."""
    if isinstance(elem, (T.AnyT, T.VarT)):
        return T.ANY
    options = elem.options if isinstance(elem, T.UnionT) else (elem,)
    lists = [o for o in options if isinstance(o, T.ListT)]
    if len(lists) == len(options):
        return T.ListT(T.union(*(o.elem for o in lists)))
    arrays = [o for o in options if isinstance(o, T.ArrayT) and o.ndim != 0]
    if len(arrays) == len(options):
        dtypes = {o.dtype for o in arrays}
        ranks = {o.ndim for o in arrays}
        return T.ArrayT(dtypes.pop() if len(dtypes) == 1 else None,
                        ranks.pop() if len(ranks) == 1 else None)
    return None


_STACK_DTYPE = {T.BOOL: "bool", T.INT: "int64", T.FLOAT: "float64", T.COMPLEX: "complex128"}


def stacked_type(elem: T.Type) -> T.Type:
    """Type of the array a ``stack`` output builds from elements of type ``elem``."""
    if elem in _STACK_DTYPE:
        return T.ArrayT(_STACK_DTYPE[elem], 1)
    if isinstance(elem, T.ArrayT):
        dtype = elem.dtype if isinstance(elem.dtype, str) else None
        return T.ArrayT(dtype, elem.ndim + 1 if isinstance(elem.ndim, int) else None)
    return T.ArrayT()


def interface(node: Structure) -> NodeSpec:
    """The pins a structure shows to the outside, as a NodeSpec."""
    def declared(tunnel: Any) -> T.Type:
        return T.parse_type(tunnel.type) if tunnel.type else T.ANY

    inputs: list[PinSpec] = []
    if node.kind == "for":
        inputs.append(PinSpec("count", T.INT, default=DEFAULT,
                              doc="Number of iterations."))
    if node.kind == "case":
        inputs.append(PinSpec("selector", T.ANY, doc="Chooses the frame that runs."))
    if node.kind == "with":
        inputs.append(PinSpec("resource", T.ANY,
                              doc="A context manager: entered before the frame runs, "
                                  "left when it is done."))
    inputs += [PinSpec(t.name, declared(t)) for t in node.inputs]
    inputs += [PinSpec(r, T.ANY, doc="Initial value of the shift register.")
               for r in node.registers]
    outputs = [PinSpec(t.name, T.ANY) for t in node.outputs]
    outputs += [PinSpec(r, T.ANY, doc="Final value of the shift register.")
                for r in node.registers]
    if node.kind == "try":
        outputs += [PinSpec("failed", T.BOOL, doc="True if the body failed."),
                    PinSpec("error", T.STR, doc="The error message, or an empty text.")]
    return NodeSpec(f"structure.{node.kind}", inputs=tuple(inputs), outputs=tuple(outputs),
                    fn=_nothing, tier="curated", executor="inline",
                    summary=f"{node.kind.capitalize()} structure.")


def entered_type(resource: T.Type) -> Optional[T.Type]:
    """What a With structure hands to its frame for a resource of this type,
    or ``None`` if that type cannot be a context manager."""
    if isinstance(resource, (T.AnyT, T.VarT, T.UnionT)):
        return T.ANY
    if not isinstance(resource, T.ObjectT):
        return None
    cls = T._resolve_class(resource.cls)
    if cls is None:
        return T.ANY
    enter = getattr(cls, "__enter__", None)
    if enter is None or getattr(cls, "__exit__", None) is None:
        return None
    try:
        returns = typing.get_type_hints(enter).get("return")
    except Exception:
        returns = None
    if returns is None:
        return T.ANY
    if returns is cls or isinstance(returns, typing.TypeVar) or returns is getattr(
            typing, "Self", object()):
        return resource
    return T.from_annotation(returns)


def boundary_specs(in_types: dict[str, T.Type], out_pins: list[str]
                   ) -> tuple[NodeSpec, NodeSpec]:
    """Specs of a frame's ``inputs`` and ``outputs`` nodes."""
    in_spec = NodeSpec(FRAME_IN, outputs=tuple(PinSpec(n, t) for n, t in in_types.items()),
                       fn=_nothing, tier="curated", executor="inline")
    out_spec = NodeSpec(FRAME_OUT,
                        inputs=tuple(PinSpec(n, T.ANY, default=DEFAULT) for n in out_pins),
                        fn=_nothing, tier="curated", executor="inline")
    return in_spec, out_spec


class _Checker:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    # -- one diagram: the top level, or the frame of a structure -------------

    def graph(self, graph: Graph, frame: Optional[tuple[NodeSpec, NodeSpec]],
              path: str, problems: list[Problem]) -> Analysis:
        nodes: dict[str, NodeInfo] = {}

        def error(code: str, message: str, node: Optional[str] = None,
                  pin: Optional[str] = None) -> None:
            problems.append(Problem("error", code, message,
                                    None if node is None else path + node, pin))

        try:
            order = graph.topological_order()
        except CycleError as exc:
            error("cycle", str(exc))
            return Analysis([], nodes, problems)
        if frame is not None:
            for nid, wanted_spec in ((INPUTS, FRAME_IN), (OUTPUTS, FRAME_OUT)):
                if nid not in graph or graph.nodes[nid].spec != wanted_spec:
                    problems.append(Problem(
                        "error", "bad-frame", f"the frame has no {nid!r} node",
                        path.rstrip("/") or None))

        for nid in order:
            node = graph.nodes[nid]
            spec: Optional[NodeSpec]
            if frame is not None and nid == INPUTS and node.spec == FRAME_IN:
                spec = frame[0]
            elif frame is not None and nid == OUTPUTS and node.spec == FRAME_OUT:
                spec = frame[1]
            elif isinstance(node, Structure):
                flaws = node.flaws()
                for flaw in flaws:
                    error("bad-frame", flaw, nid)
                if flaws:
                    nodes[nid] = NodeInfo(None)
                    continue
                spec = interface(node)
            else:
                spec = self.registry.find(node.spec)
            info = NodeInfo(spec)
            nodes[nid] = info
            if spec is None:
                error("unknown-spec", f"unknown node spec {node.spec!r}", nid)
                continue
            incoming = graph.incoming(nid)

            # -- which inputs does this node have?  The declared pins, plus the
            #    ones grown on a variadic function (args_0, args_1, ... and extra
            #    keyword names).
            slots: dict[str, PinSpec] = {p.name: p for p in spec.inputs if not p.variadic}
            for name in sorted(set(incoming) | set(node.values)):
                if name in slots:
                    continue
                grown = spec.pin_for(name)
                if grown is None:
                    error("unknown-pin", f"{node.spec} has no input {name!r}", nid, name)
                else:
                    slots[name] = grown
            numbered = sorted(i for i in map(spec.variadic_index, slots) if i is not None)
            if numbered != list(range(len(numbered))):
                pin_name = spec.var_positional.name if spec.var_positional else ""
                error("variadic-gap",
                      f"inputs {pin_name}_0, {pin_name}_1, ... must be numbered without gaps",
                      nid, pin_name)
            for out_wire in graph.outgoing(nid):
                if spec.output(out_wire.src_pin) is None:
                    error("unknown-pin", f"{node.spec} has no output {out_wire.src_pin!r}",
                          nid, out_wire.src_pin)
            for name in node.config:
                if spec.config_pin(name) is None:
                    error("unknown-config", f"{node.spec} has no configuration {name!r}",
                          nid, name)

            # -- configuration
            for pin in spec.config:
                if pin.name in node.config:
                    try:
                        info.config[pin.name] = T.accept(
                            T.plain_scalar(node.config[pin.name]), pin.type)
                    except T.TypeMismatch as exc:
                        error("bad-config", f"configuration {pin.name!r}: {exc}", nid, pin.name)
                elif pin.required:
                    error("missing-config", f"configuration {pin.name!r} is not set",
                          nid, pin.name)
                elif pin.default is not DEFAULT:
                    info.config[pin.name] = pin.default

            # -- what actually arrives at each input
            actual: dict[str, Optional[T.Type]] = {}
            for name, pin in slots.items():
                wire = incoming.get(name)
                if wire is not None:
                    source = nodes.get(wire.src)
                    actual[name] = T.ANY if source is None else source.out_types.get(
                        wire.src_pin, T.ANY)
                elif name in node.values:
                    actual[name] = T.type_of(T.plain_scalar(node.values[name]))
                elif pin.default is DEFAULT:
                    actual[name] = None          # optional, default value not known
                elif pin.default is not MISSING:
                    actual[name] = T.type_of(pin.default)
                else:
                    actual[name] = None
                    error("missing-input",
                          f"input {name!r} is not connected and has no value", nid, name)

            # -- solve the node's type variables
            bindings: dict[str, Any] = {}
            for name, pin in slots.items():
                try:
                    T.unify(pin.type, actual[name], bindings)
                except T.UnifyError as exc:
                    error("type-conflict", f"input {name!r}: {exc}", nid, name)
            info.in_types = {name: T.substitute(pin.type, bindings)
                             for name, pin in slots.items()}
            info.out_types = {p.name: T.substitute(p.type, bindings) for p in spec.outputs}

            # -- do the wires and literals fit?  What does each pin really receive?
            received = info.received
            for name in slots:
                wanted = info.in_types[name]
                source_type = actual[name]
                received[name] = source_type
                wire = incoming.get(name)
                if wire is not None:
                    assert source_type is not None
                    compat = T.check(source_type, wanted)
                    info.wires[name] = (wire, compat)
                    if compat.ok:
                        received[name] = T.delivered(source_type, wanted, compat)
                    else:
                        error("type-mismatch", f"wire {wire}: {compat.reason}", nid, name)
                elif name in node.values:
                    try:
                        literal = T.accept(T.plain_scalar(node.values[name]), wanted)
                    except T.TypeMismatch as exc:
                        error("bad-literal", f"value of input {name!r}: {exc}", nid, name)
                    else:
                        info.literals[name] = literal
                        received[name] = T.type_of(literal)

            if isinstance(node, Structure):
                self.structure(node, info, path, problems)
            elif spec.infer is not None:
                # polymorphic nodes compute their output types from what they receive
                try:
                    inferred = dict(spec.infer(received, dict(info.config)))
                except Exception as exc:
                    problems.append(Problem(
                        "warning", "infer-failed",
                        f"type inference of {node.spec} failed: {type(exc).__name__}: {exc}",
                        path + nid))
                    inferred = {}
                for name, inferred_type in inferred.items():
                    if name in info.out_types and isinstance(inferred_type, T.Type):
                        info.out_types[name] = inferred_type
            if not isinstance(node, Structure):
                # An array written in place keeps its dtype and rank, so the
                # output that hands it on is as precise as what came in.
                for pin in spec.outputs:
                    arrived = received.get(pin.aliases or "") if pin.passthrough else None
                    if isinstance(arrived, T.ArrayT):
                        info.out_types[pin.name] = arrived

        analysis = Analysis(order, nodes, problems)
        if frame is not None and OUTPUTS in nodes:
            analysis.frame_out = dict(nodes[OUTPUTS].received)
        return analysis

    # -- structures -----------------------------------------------------------

    def structure(self, node: Structure, info: NodeInfo, path: str,
                  problems: list[Problem]) -> None:
        here = path + node.id
        received = info.received

        def error(code: str, message: str, pin: Optional[str] = None) -> None:
            problems.append(Problem("error", code, message, here, pin))

        for frame in node.frames:
            if frame.imports:
                error("bad-frame", "a frame cannot have imports of its own")

        # -- what the frame sees of each input tunnel
        inner: dict[str, T.Type] = {}
        for tunnel in node.inputs:
            outer = received.get(tunnel.name) or T.ANY
            if tunnel.mode == "index":
                elem = element_type(outer)
                if elem is None:
                    error("bad-tunnel",
                          f"input {tunnel.name!r} cannot be indexed: it is {outer}", tunnel.name)
                    elem = T.ANY
                inner[tunnel.name] = elem
            else:
                inner[tunnel.name] = outer
        out_pins = [t.name for t in node.outputs] + list(node.registers)
        info.data = {"registers": {}, "outputs": {}, "convert": {}}

        if node.kind == "case":
            selector = info.received.get("selector") or T.ANY
            inner["selector"] = selector
            self._check_cases(node, selector, error)
            self._branches(node, info, [inner] * len(node.frames), out_pins, path, problems,
                           error)
            return
        if node.kind == "try":
            handler = dict(inner, error=T.STR, error_type=T.STR)
            self._branches(node, info, [inner, handler], out_pins, path, problems, error)
            info.out_types.update(failed=T.BOOL, error=T.STR)
            return
        if node.kind == "with":
            resource = received.get("resource") or T.ANY
            entered = entered_type(resource)
            if entered is None:
                error("type-mismatch",
                      f"'resource' must be a context manager, not {resource}", "resource")
                entered = T.ANY
            inner["resource"] = entered

        if node.kind == "for":
            inner["i"] = T.INT
            if received.get("count") is None and not any(t.mode == "index" for t in node.inputs):
                error("no-count", "a For loop needs a count or an indexing input", "count")
        elif node.kind == "while":
            inner["i"] = T.INT
            out_pins.append("stop")

        # -- shift registers: widen their type until the loop body agrees
        registers = {r: received.get(r) or T.ANY for r in node.registers}
        conflicts: dict[str, T.Type] = {}
        trial: list[Problem] = []
        analysis = Analysis([], {}, trial)
        for _ in range(8):
            trial = []
            conflicts = {}
            analysis = self.graph(node.frame, boundary_specs({**inner, **registers}, out_pins),
                                  here + "/", trial)
            changed = False
            for name in node.registers:
                fed_back = analysis.frame_out.get(name)
                if fed_back is None or T.check(fed_back, registers[name]).ok:
                    continue
                joined = T.join(registers[name], fed_back)
                if joined is None or joined == registers[name]:
                    conflicts[name] = fed_back
                else:
                    registers[name] = joined
                    changed = True
            if not changed:
                break
        else:
            error("register-conflict", "the types of the shift registers do not settle")
        problems.extend(trial)
        info.frames = [analysis]
        frame_out = analysis.frame_out

        for name in node.registers:
            fed_back = frame_out.get(name)
            if fed_back is None:
                error("unwired-register",
                      f"shift register {name!r} is not fed a next value inside the loop", name)
            elif name in conflicts:
                error("register-conflict",
                      f"shift register {name!r} holds {registers[name]} but the loop feeds "
                      f"back {conflicts[name]}", name)
            info.out_types[name] = registers[name]
            info.data["registers"][name] = registers[name]
            start = received.get(name)
            if start is not None and not T.check(start, registers[name]).ok:
                error("register-conflict",
                      f"shift register {name!r} starts as {start}, which does not fit "
                      f"{registers[name]}", name)

        if node.kind == "while":
            stop = frame_out.get("stop")
            if stop is None:
                error("unwired-stop", "the loop's 'stop' pin is not wired inside the loop", "stop")
            elif not T.check(stop, T.BOOL).ok:
                error("type-mismatch", f"'stop' must be Bool, not {stop}", "stop")

        for tunnel in node.outputs:
            got = frame_out.get(tunnel.name)
            if got is None:
                error("unwired-tunnel",
                      f"output {tunnel.name!r} is not wired inside the {node.kind} structure",
                      tunnel.name)
                got = T.ANY
            out_type = self._collected(node, tunnel, got, error)
            self._finish_output(info, tunnel, got, out_type, error)

    def _collected(self, node: Structure, tunnel: Any, got: T.Type, error: Any) -> T.Type:
        if tunnel.mode == "collect":
            return T.ListT(got)
        if tunnel.mode == "stack":
            return stacked_type(got)
        if tunnel.mode == "concat":
            joined_type = concatenated_type(got)
            if joined_type is not None:
                return joined_type
            error("bad-tunnel",
                  f"output {tunnel.name!r} concatenates lists or arrays, not {got}", tunnel.name)
            return T.ANY
        if tunnel.fallback is not UNSET:
            # the fallback stands in when the loop runs zero times
            fallback = T.type_of(T.plain_scalar(tunnel.fallback))
            joined = T.join(got, fallback)
            if joined is None:
                error("tunnel-conflict",
                      f"output {tunnel.name!r} is {got} but its fallback is {fallback}",
                      tunnel.name)
                return got
            return joined
        return got

    def _finish_output(self, info: NodeInfo, tunnel: Any, got: T.Type, out_type: T.Type,
                       error: Any) -> None:
        """Apply a declared tunnel type and record what the runtime must convert."""
        if tunnel.type:
            declared = T.parse_type(tunnel.type)
            if not T.check(out_type, declared).ok:
                error("type-mismatch",
                      f"output {tunnel.name!r} is {out_type}, which does not fit its "
                      f"declared type {declared}", tunnel.name)
            else:
                if T.check(out_type, declared).verdict is not T.Verdict.EXACT:
                    info.data["convert"][tunnel.name] = declared
                out_type = declared
        elif tunnel.mode == "last" and got != out_type:
            info.data["convert"][tunnel.name] = out_type   # e.g. Int widened to Float
        info.out_types[tunnel.name] = out_type
        info.data["outputs"][tunnel.name] = (tunnel.mode, got, out_type)

    def _branches(self, node: Structure, info: NodeInfo, inners: list[dict[str, T.Type]],
                  out_pins: list[str], path: str, problems: list[Problem], error: Any) -> None:
        """Structures of which one frame runs: every output takes the join of
        what the frames deliver."""
        analyses = []
        for index, (frame, inner) in enumerate(zip(node.frames, inners)):
            analyses.append(self.graph(frame, boundary_specs(inner, out_pins),
                                       f"{path}{node.id}[{index}]/", problems))
        info.frames = analyses
        for tunnel in node.outputs:
            fallback = (None if tunnel.fallback is UNSET
                        else T.type_of(T.plain_scalar(tunnel.fallback)))
            joined: Optional[T.Type] = None
            seen: list[T.Type] = []
            for index, analysis in enumerate(analyses):
                got = analysis.frame_out.get(tunnel.name)
                if got is None:
                    if fallback is None:
                        error("unwired-tunnel",
                              f"output {tunnel.name!r} is not wired in frame {index} "
                              "and has no fallback", tunnel.name)
                        continue
                    got = fallback
                seen.append(got)
                joined = got if joined is None else T.join(joined, got)
                if joined is None:
                    break
            if seen and joined is None:
                error("tunnel-conflict",
                      f"output {tunnel.name!r} has unrelated types in different frames: "
                      + ", ".join(sorted({str(t) for t in seen})), tunnel.name)
            out_type = joined if joined is not None else T.ANY
            if any(t != out_type for t in seen):
                info.data["convert"][tunnel.name] = out_type
            self._finish_output(info, tunnel, out_type, out_type, error)

    def _check_cases(self, node: Structure, selector: T.Type, error: Any) -> None:
        dynamic = isinstance(selector, (T.AnyT, T.VarT, T.UnionT))
        if not dynamic and selector not in (T.BOOL, T.INT, T.STR) and not isinstance(
                selector, T.EnumT):
            error("bad-selector",
                  f"a selector is Bool, Int, Str or an Enum, not {selector}", "selector")
            return
        defaults = [i for i, match in enumerate(node.cases) if match is None]
        if len(defaults) > 1:
            error("duplicate-case", "more than one default frame")
        if len(node.cases) != len(node.frames):
            error("bad-frame", "the case structure needs one entry in 'cases' per frame")
        seen: set[Any] = set()
        covered: set[Any] = set()
        for index, match in enumerate(node.cases):
            for entry in match or []:
                key = ("range", entry) if isinstance(entry, tuple) else (type(entry).__name__, entry)
                if key in seen:
                    error("duplicate-case", f"{entry!r} is handled by more than one frame")
                seen.add(key)
                covered.add(entry)
                if dynamic:
                    continue
                fits = (
                    (selector == T.BOOL and isinstance(entry, bool))
                    or (selector == T.INT and not isinstance(entry, (bool, str)))
                    or (selector == T.STR and isinstance(entry, str))
                    or (isinstance(selector, T.EnumT) and isinstance(entry, str)
                        and entry in selector.members))
                if not fits:
                    error("bad-case",
                          f"frame {index} handles {entry!r}, which a {selector} selector "
                          "can never be", "selector")
        if defaults:
            return
        if selector == T.BOOL and {True, False} <= covered:
            return
        if isinstance(selector, T.EnumT) and set(selector.members) <= covered:
            return
        error("no-default-case",
              "not every selector value is handled: add a default frame", "selector")
