"""Static analysis of a graph: pins, types, missing inputs.

:func:`analyze` never raises for a broken graph.  It returns an
:class:`Analysis` whose ``problems`` list says what is wrong; a graph is
runnable when it has no problem of severity ``"error"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import types as T
from .library import DEFAULT, MISSING, NodeSpec, PinSpec, Registry, default_registry
from .model import CycleError, Graph, Wire

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


@dataclass
class Analysis:
    order: list[str]
    nodes: dict[str, NodeInfo]
    problems: list[Problem]

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
    nodes: dict[str, NodeInfo] = {}

    def error(code: str, message: str, node: Optional[str] = None,
              pin: Optional[str] = None) -> None:
        problems.append(Problem("error", code, message, node, pin))

    try:
        order = graph.topological_order()
    except CycleError as exc:
        error("cycle", str(exc))
        return Analysis([], nodes, problems)

    for nid in order:
        node = graph.nodes[nid]
        spec = registry.find(node.spec)
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
                error("missing-input", f"input {name!r} is not connected and has no value",
                      nid, name)

        # -- solve the node's type variables
        bindings: dict[str, Any] = {}
        for name, pin in slots.items():
            try:
                T.unify(pin.type, actual[name], bindings)
            except T.UnifyError as exc:
                error("type-conflict", f"input {name!r}: {exc}", nid, name)
        info.in_types = {name: T.substitute(pin.type, bindings) for name, pin in slots.items()}
        info.out_types = {p.name: T.substitute(p.type, bindings) for p in spec.outputs}

        # -- do the wires and literals fit?  What does each pin really receive?
        received: dict[str, Optional[T.Type]] = {}
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

        # -- polymorphic nodes compute their output types from what they receive
        if spec.infer is not None:
            try:
                inferred = dict(spec.infer(received, dict(info.config)))
            except Exception as exc:
                problems.append(Problem(
                    "warning", "infer-failed",
                    f"type inference of {node.spec} failed: {type(exc).__name__}: {exc}", nid))
                inferred = {}
            for name, inferred_type in inferred.items():
                if name in info.out_types and isinstance(inferred_type, T.Type):
                    info.out_types[name] = inferred_type

    return Analysis(order, nodes, problems)
