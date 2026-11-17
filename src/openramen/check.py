"""Static analysis of a graph: pins, types, missing inputs.

:func:`analyze` never raises for a broken graph.  It returns an
:class:`Analysis` whose ``problems`` list says what is wrong; a graph is
runnable when it has no problem of severity ``"error"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import types as T
from .library import MISSING, NodeSpec, Registry, default_registry
from .model import CycleError, Graph, Wire

__all__ = ["Problem", "NodeInfo", "Analysis", "analyze", "GraphCheckError"]


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


_base: Optional[Registry] = None


def _registry(registry: Optional[Registry]) -> Registry:
    global _base
    if registry is not None:
        return registry
    if _base is None:
        _base = default_registry()
    return _base


def analyze(graph: Graph, registry: Optional[Registry] = None) -> Analysis:
    """Check a graph and resolve the type of every pin."""
    registry = _registry(registry)
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

        # -- do the pins that are used exist?
        for pin_name in incoming:
            if spec.input(pin_name) is None:
                error("unknown-pin", f"{node.spec} has no input {pin_name!r}", nid, pin_name)
        for out_wire in graph.outgoing(nid):
            if spec.output(out_wire.src_pin) is None:
                error("unknown-pin", f"{node.spec} has no output {out_wire.src_pin!r}",
                      nid, out_wire.src_pin)
        for name in node.values:
            if spec.input(name) is None:
                error("unknown-pin", f"{node.spec} has no input {name!r}", nid, name)
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
            else:
                info.config[pin.name] = pin.default

        # -- what actually arrives at each input
        actual: dict[str, Optional[T.Type]] = {}
        for pin in spec.inputs:
            wire = incoming.get(pin.name)
            if wire is not None:
                source = nodes.get(wire.src)
                actual[pin.name] = T.ANY if source is None else source.out_types.get(
                    wire.src_pin, T.ANY)
            elif pin.name in node.values:
                actual[pin.name] = T.type_of(T.plain_scalar(node.values[pin.name]))
            elif pin.default is not MISSING:
                actual[pin.name] = T.type_of(pin.default)
            else:
                actual[pin.name] = None
                error("missing-input", f"input {pin.name!r} is not connected and has no value",
                      nid, pin.name)

        # -- solve the node's type variables
        bindings: dict[str, Any] = {}
        for pin in spec.inputs:
            try:
                T.unify(pin.type, actual[pin.name], bindings)
            except T.UnifyError as exc:
                error("type-conflict", f"input {pin.name!r}: {exc}", nid, pin.name)
        info.in_types = {p.name: T.substitute(p.type, bindings) for p in spec.inputs}
        info.out_types = {p.name: T.substitute(p.type, bindings) for p in spec.outputs}

        # -- do the wires and literals fit?  What does each pin really receive?
        received: dict[str, Optional[T.Type]] = {}
        for pin in spec.inputs:
            wanted = info.in_types[pin.name]
            source_type = actual[pin.name]
            received[pin.name] = source_type
            wire = incoming.get(pin.name)
            if wire is not None:
                assert source_type is not None
                compat = T.check(source_type, wanted)
                info.wires[pin.name] = (wire, compat)
                if compat.ok:
                    received[pin.name] = T.delivered(source_type, wanted, compat)
                else:
                    error("type-mismatch", f"wire {wire}: {compat.reason}", nid, pin.name)
            elif pin.name in node.values:
                try:
                    literal = T.accept(T.plain_scalar(node.values[pin.name]), wanted)
                except T.TypeMismatch as exc:
                    error("bad-literal", f"value of input {pin.name!r}: {exc}", nid, pin.name)
                else:
                    info.literals[pin.name] = literal
                    received[pin.name] = T.type_of(literal)

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
