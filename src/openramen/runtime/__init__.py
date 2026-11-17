"""Headless execution of graphs.

This first runtime runs one node at a time, in dependency order, in the
calling thread.  Parallel executors, caching and structures build on the same
:class:`Plan`.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Callable, Iterable, Optional, Union

from .. import types as T
from ..check import Analysis, GraphCheckError, analyze
from ..library import NodeSpec, Registry
from ..model import Graph

__all__ = ["Plan", "Step", "Event", "RunResult", "NodeError", "compile_graph", "run"]

_PASS, _CONVERT, _ACCEPT = 0, 1, 2
_IMMUTABLE = (type(None), bool, int, float, complex, str, bytes, PurePath)


class NodeError(Exception):
    """A node failed while a graph was running.

    ``original`` is the exception raised inside the node (also available as
    ``__cause__``); ``pin`` is set when a value did not fit a pin.
    """

    def __init__(self, node: str, spec: str, original: BaseException,
                 pin: Optional[str] = None, detail: str = ""):
        self.node = node
        self.spec = spec
        self.pin = pin
        self.original = original
        what = detail or f"{type(original).__name__}: {original}"
        where = f"node {node!r} ({spec})"
        if pin:
            where += f", pin {pin!r}"
        super().__init__(f"{where}: {what}")


@dataclass(frozen=True)
class Event:
    """Progress notification sent to a listener during a run."""

    kind: str                 # graph-started, node-started, node-finished,
    node: Optional[str] = None  # node-failed, graph-finished, graph-failed
    duration: Optional[float] = None
    error: Optional[BaseException] = None


Listener = Callable[[Event], None]


@dataclass
class Step:
    """Everything needed to execute one node."""

    node: str
    spec: NodeSpec
    wired: list[tuple[str, tuple[str, str], int, Any]]  # pin, source, mode, payload
    literals: dict[str, Any]
    config: dict[str, Any]
    out_types: dict[str, T.Type]
    needs: frozenset[str] = field(default_factory=frozenset)  # upstream node ids


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

    def __repr__(self) -> str:
        return f"<RunResult: {len(self._outputs)} nodes>"


def compile_graph(graph: Graph, registry: Optional[Registry] = None) -> Plan:
    """Check a graph and turn it into an execution plan.

    Raises :class:`~openramen.check.GraphCheckError` if the graph has errors.
    """
    analysis = analyze(graph, registry)
    analysis.raise_for_errors()
    steps: list[Step] = []
    needs: dict[str, frozenset[str]] = {}
    for nid in analysis.order:
        info = analysis.nodes[nid]
        assert info.spec is not None
        wired: list[tuple[str, tuple[str, str], int, Any]] = []
        upstream: set[str] = set()
        for pin, (wire, compat) in info.wires.items():
            mode, payload = _PASS, None  # type: tuple[int, Any]
            if compat.verdict is T.Verdict.COERCION:
                mode, payload = _CONVERT, compat.convert
            elif compat.verdict is T.Verdict.DYNAMIC:
                mode, payload = _ACCEPT, info.in_types[pin]
            wired.append((pin, (wire.src, wire.src_pin), mode, payload))
            upstream.add(wire.src)
            upstream |= needs[wire.src]
        needs[nid] = frozenset(upstream)
        steps.append(Step(nid, info.spec, wired, dict(info.literals), dict(info.config),
                          dict(info.out_types), needs[nid]))
    return Plan(graph.name, steps, analysis)


def _fresh(value: Any) -> Any:
    """A private copy of a literal, so no node can alter the graph's own."""
    return value if isinstance(value, _IMMUTABLE) else copy.deepcopy(value)


def run(graph: Union[Graph, Plan], registry: Optional[Registry] = None, *,
        targets: Optional[Iterable[str]] = None,
        listener: Optional[Listener] = None,
        check_outputs: Optional[bool] = None) -> RunResult:
    """Run a graph (or a compiled :class:`Plan`) and return its values.

    * ``targets``: only run these nodes and what they depend on.
    * ``listener``: called with an :class:`Event` as the run progresses.
    * ``check_outputs``: verify each output against its declared type.  By
      default this is done for every node that is not hand-curated.
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
    values: dict[tuple[str, str], Any] = {}
    durations: dict[str, float] = {}
    emit(Event("graph-started"))
    started = time.perf_counter()
    try:
        for step in steps:
            emit(Event("node-started", step.node))
            t0 = time.perf_counter()
            try:
                _execute(step, values, check_outputs)
            except NodeError as exc:
                emit(Event("node-failed", step.node, time.perf_counter() - t0, exc))
                raise
            durations[step.node] = time.perf_counter() - t0
            emit(Event("node-finished", step.node, durations[step.node]))
    except BaseException as exc:
        emit(Event("graph-failed", duration=time.perf_counter() - started, error=exc))
        raise
    emit(Event("graph-finished", duration=time.perf_counter() - started))
    outputs = {s.node: tuple(p.name for p in s.spec.outputs) for s in steps}
    return RunResult(values, outputs, durations)


def _execute(step: Step, values: dict[tuple[str, str], Any],
             check_outputs: Optional[bool]) -> None:
    spec = step.spec
    arguments = {pin: _fresh(value) for pin, value in step.literals.items()}
    for pin, source, mode, payload in step.wired:
        value = values[source]
        if mode == _CONVERT:
            try:
                value = payload(value)
            except Exception as exc:
                raise NodeError(step.node, spec.id, exc, pin) from exc
        elif mode == _ACCEPT:
            try:
                value = T.accept(value, payload)
            except T.TypeMismatch as exc:
                raise NodeError(step.node, spec.id, exc, pin, str(exc)) from exc
        arguments[pin] = value
    config = {name: _fresh(value) for name, value in step.config.items()}
    try:
        results = spec.call(arguments, config)
    except Exception as exc:
        raise NodeError(step.node, spec.id, exc) from exc
    verify = spec.tier != "curated" if check_outputs is None else check_outputs
    for pin, value in results.items():
        value = T.plain_scalar(value)
        if verify and not T.conforms(value, step.out_types[pin]):
            mismatch = T.TypeMismatch(
                f"output is declared {step.out_types[pin]} but the node returned "
                f"{T.type_of(value)}")
            raise NodeError(step.node, spec.id, mismatch, pin, str(mismatch))
        values[(step.node, pin)] = value
