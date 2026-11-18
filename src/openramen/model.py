"""The graph document: nodes, wires, and their JSON form.

This module knows nothing about node libraries or execution.  A
:class:`Graph` only guarantees its own structure:

* node ids are unique identifiers,
* every wire joins two existing nodes,
* an input pin has at most one incoming wire,
* there are no cycles.

Whether pins exist and types fit is decided by :mod:`openramen.check`.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional, Union

from . import types as T
from .serial import SerializationError, decode_value, encode_value

__all__ = ["Graph", "Node", "Wire", "Structure", "Tunnel", "GraphError", "CycleError",
           "FORMAT", "VERSION", "UNSET", "FRAME_IN", "FRAME_OUT", "INPUTS", "OUTPUTS"]

FORMAT = "openramen.graph"
#: Newest file version this code reads.  Files are written with the lowest
#: version that can express them: 2 only when a graph contains structures.
VERSION = 2

#: A frame (the diagram inside a structure) talks to the outside through two
#: boundary nodes with these ids and specs.
INPUTS, OUTPUTS = "inputs", "outputs"
FRAME_IN, FRAME_OUT = "frame.inputs", "frame.outputs"

STRUCTURE_KINDS = ("for", "while", "case", "subgraph", "try", "with")
INPUT_MODES = ("plain", "index")
OUTPUT_MODES = ("last", "collect", "stack", "concat")
#: Pin names the structures use themselves.
RESERVED_PINS = ("i", "count", "stop", "selector")
#: Further pin names that only one kind of structure uses itself.
KIND_PINS = {"try": ("error", "error_type", "failed"), "with": ("resource",)}

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


class GraphError(Exception):
    """An operation would break the structure of a graph."""


class CycleError(GraphError):
    """A wire would create a cycle."""


@dataclass
class Node:
    """One node placed in a graph.

    ``values`` holds literals for unwired input pins, ``config`` holds the
    node's configuration parameters.
    """

    id: str
    spec: str
    values: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    pos: tuple[float, float] = (0.0, 0.0)
    label: str = ""


@dataclass(frozen=True, order=True)
class Wire:
    """A connection from an output pin to an input pin."""

    src: str
    src_pin: str
    dst: str
    dst_pin: str

    def __str__(self) -> str:
        return f"{self.src}.{self.src_pin} -> {self.dst}.{self.dst_pin}"


class _Unset:
    """Marker for "no fallback value given"."""

    _instance: "Optional[_Unset]" = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __copy__(self) -> "_Unset":
        return self

    def __deepcopy__(self, memo: dict) -> "_Unset":
        return self

    def __reduce__(self) -> str:
        return "UNSET"


UNSET = _Unset()


@dataclass
class Tunnel:
    """A pin on the border of a structure.

    Input tunnels are ``plain`` (the same value on every iteration) or
    ``index`` (one element per iteration of a For loop).  Output tunnels give
    the ``last`` value, ``collect`` one value per iteration into a list,
    ``stack`` them into an array, or ``concat`` lists or arrays end to end.
    ``fallback`` is the value of an output that would otherwise have none: a
    case frame that leaves it unwired, or a loop that ran zero times.
    """

    name: str
    mode: str
    type: str = ""          # optional declared type, as text
    fallback: Any = UNSET


@dataclass
class Structure(Node):
    """A node that owns diagrams: a loop, a case structure, a subgraph, a Try
    or a With structure.

    ``frames`` are the inner diagrams: one, one per case, or for a Try the
    body and its error handler.  Each frame has an ``inputs`` node, whose
    outputs are what the frame receives, and an ``outputs`` node, whose
    inputs are what it gives back.
    """

    kind: str = "subgraph"
    frames: list["Graph"] = field(default_factory=list)
    inputs: list[Tunnel] = field(default_factory=list)
    outputs: list[Tunnel] = field(default_factory=list)
    #: Shift registers (loops only): state carried from one iteration to the
    #: next.  Each is an input (initial value) and an output (final value)
    #: outside, and a current/next pair on the boundary nodes inside.
    registers: list[str] = field(default_factory=list)
    #: Case structures: what each frame matches.  A list of values (and
    #: ``(low, high)`` integer ranges, both ends included), or ``None`` for
    #: the default frame.
    cases: list[Optional[list]] = field(default_factory=list)
    #: While loops: fail instead of looping forever.  ``None`` means no limit.
    max_iterations: Optional[int] = None
    #: For loops without shift registers: iterations may run at the same time.
    parallel: bool = False

    @property
    def reserved(self) -> tuple[str, ...]:
        """Pin names this structure uses itself."""
        return RESERVED_PINS + KIND_PINS.get(self.kind, ())

    @property
    def frame(self) -> "Graph":
        """The first (for loops and subgraphs: the only) frame."""
        return self.frames[0]

    def input(self, name: str) -> Optional[Tunnel]:
        return next((t for t in self.inputs if t.name == name), None)

    def output(self, name: str) -> Optional[Tunnel]:
        return next((t for t in self.outputs if t.name == name), None)

    def flaws(self) -> list[str]:
        """What is wrong with the shape of this structure (not its wiring).

        The methods of this class never produce a flawed structure, but its
        attributes are plain lists and can be edited directly.
        """
        found: list[str] = []
        loop = self.kind in ("for", "while")
        if self.kind not in STRUCTURE_KINDS:
            return [f"unknown structure kind {self.kind!r}"]
        if not self.frames or not all(isinstance(f, Graph) for f in self.frames):
            return ["the structure has no frame"]
        if self.kind == "case":
            if len(self.cases) != len(self.frames):
                found.append(f"{len(self.frames)} frames but {len(self.cases)} case entries")
        else:
            wanted = 2 if self.kind == "try" else 1
            if len(self.frames) != wanted:
                found.append(
                    f"a {self.kind} structure has exactly "
                    f"{'two frames' if wanted == 2 else 'one frame'}, not {len(self.frames)}")
            if self.cases:
                found.append("only a case structure has cases")
        if self.registers and not loop:
            found.append("only loops have shift registers")
        if self.parallel and (self.kind != "for" or self.registers):
            found.append("only a For loop without shift registers can run its "
                         "iterations in parallel")
        for side, tunnels, modes in (("input", self.inputs, INPUT_MODES),
                                     ("output", self.outputs, OUTPUT_MODES)):
            names = [t.name for t in tunnels] + list(self.registers)
            for name in sorted(set(names)):
                if names.count(name) > 1:
                    found.append(f"more than one {side} pin is called {name!r}")
                if name in self.reserved:
                    found.append(f"{name!r} is a pin the structure uses itself")
            for tunnel in tunnels:
                if tunnel.mode not in modes:
                    found.append(f"{side} {tunnel.name!r} has the unknown mode {tunnel.mode!r}")
                elif tunnel.mode == "index" and self.kind != "for":
                    found.append(f"input {tunnel.name!r}: only a For loop can index its inputs")
                elif side == "output" and tunnel.mode != "last" and not loop:
                    found.append(f"output {tunnel.name!r}: only loops can {tunnel.mode} "
                                 "their outputs")
                if tunnel.type:
                    try:
                        T.parse_type(tunnel.type)
                    except T.TypeSyntaxError as exc:
                        found.append(f"{side} {tunnel.name!r}: {exc}")
        return found

    def _check_new(self, name: str, taken: list[str]) -> None:
        _check_name("pin name", name)
        if name in self.reserved:
            raise GraphError(f"{name!r} is a pin the structure uses itself")
        if name in taken:
            raise GraphError(f"structure {self.id!r} already has a pin {name!r}")

    def add_input(self, name: str, mode: str = "plain", type: str = "") -> Tunnel:
        """Add an input tunnel."""
        self._check_new(name, [t.name for t in self.inputs] + self.registers)
        if mode not in INPUT_MODES:
            raise GraphError(f"input mode must be one of {INPUT_MODES}")
        if mode == "index" and self.kind != "for":
            raise GraphError("only a For loop can index its inputs")
        tunnel = Tunnel(name, mode, _type_text(type))
        self.inputs.append(tunnel)
        return tunnel

    def add_output(self, name: str, mode: str = "last", type: str = "",
                   fallback: Any = UNSET) -> Tunnel:
        """Add an output tunnel."""
        self._check_new(name, [t.name for t in self.outputs] + self.registers)
        if mode not in OUTPUT_MODES:
            raise GraphError(f"output mode must be one of {OUTPUT_MODES}")
        if mode != "last" and self.kind not in ("for", "while"):
            raise GraphError(f"only loops can {mode} their outputs")
        tunnel = Tunnel(name, mode, _type_text(type), fallback)
        self.outputs.append(tunnel)
        return tunnel

    def add_register(self, name: str) -> str:
        """Add a shift register (loops only)."""
        if self.kind not in ("for", "while"):
            raise GraphError("only loops have shift registers")
        if self.parallel:
            raise GraphError("a loop that runs its iterations in parallel cannot have "
                             "shift registers")
        self._check_new(name, [t.name for t in self.inputs + self.outputs] + self.registers)
        self.registers.append(name)
        return name

    def add_frame(self, match: Optional[list] = None) -> "Graph":
        """Add a frame to a case structure.  ``match=None`` is the default frame."""
        if self.kind != "case":
            raise GraphError("only a case structure has several frames")
        self.cases.append(_check_match(match))
        frame = _new_frame(f"{self.id}[{len(self.frames)}]")
        self.frames.append(frame)
        return frame


def _type_text(text: Any) -> str:
    if not text:
        return ""
    try:
        return T.format_type(T.parse_type(text))
    except (T.TypeSyntaxError, TypeError) as exc:
        raise GraphError(f"invalid tunnel type: {exc}") from exc


def _check_match(match: Any) -> Optional[list]:
    if match is None:
        return None
    if not isinstance(match, (list, tuple)) or not match:
        raise GraphError("a case matches a non-empty list of values, or None for the default")
    out: list = []
    for entry in match:
        if isinstance(entry, (bool, str)) or (isinstance(entry, int)):
            out.append(entry)
        elif (isinstance(entry, (tuple, list)) and len(entry) == 2
              and all(isinstance(e, int) and not isinstance(e, bool) for e in entry)
              and entry[0] <= entry[1]):
            out.append((entry[0], entry[1]))
        else:
            raise GraphError(
                f"invalid case value {entry!r}: use a boolean, an integer, text, "
                "or a (low, high) integer range")
    return out


def _new_frame(name: str = "") -> "Graph":
    frame = Graph(name)
    frame.add_node(FRAME_IN, INPUTS)
    frame.add_node(FRAME_OUT, OUTPUTS)
    return frame


NodeRef = Union[str, Node]


def _node_id(ref: NodeRef) -> str:
    return ref.id if isinstance(ref, Node) else ref


def _check_name(kind: str, name: Any) -> str:
    if not isinstance(name, str) or not _ID.match(name):
        raise GraphError(f"invalid {kind} {name!r}: must be an identifier")
    return name


class Graph:
    """A dataflow diagram."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        #: Python modules whose functions this graph uses as nodes.  They are
        #: imported when the graph is checked or run, never when it is loaded.
        self.imports: list[str] = []
        self._nodes: dict[str, Node] = {}
        self._in: dict[str, dict[str, Wire]] = {}    # dst -> dst_pin -> wire
        self._out: dict[str, set[Wire]] = {}         # src -> wires

    # -- reading ------------------------------------------------------------

    @property
    def nodes(self) -> dict[str, Node]:
        """Nodes by id, in insertion order.  Do not add or remove entries."""
        return self._nodes

    @property
    def wires(self) -> list[Wire]:
        """All wires, sorted for stable output."""
        return sorted(self._all_wires())

    def _all_wires(self) -> Iterator[Wire]:
        for pins in self._in.values():
            yield from pins.values()

    def __contains__(self, ref: object) -> bool:
        return isinstance(ref, (str, Node)) and _node_id(ref) in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    def node(self, ref: NodeRef) -> Node:
        try:
            return self._nodes[_node_id(ref)]
        except KeyError:
            raise GraphError(f"no node {_node_id(ref)!r} in graph") from None

    def incoming(self, ref: NodeRef) -> dict[str, Wire]:
        """Wires arriving at a node, by input pin."""
        return dict(self._in.get(self.node(ref).id, {}))

    def outgoing(self, ref: NodeRef, pin: Optional[str] = None) -> list[Wire]:
        """Wires leaving a node, optionally from one output pin only."""
        nid = self.node(ref).id
        wires = self._out.get(nid, ())
        return sorted(w for w in wires if pin is None or w.src_pin == pin)

    # -- editing ------------------------------------------------------------

    def _fresh_id(self, spec: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", spec.rsplit(".", 1)[-1]).lower() or "node"
        if not re.match(r"[a-z_]", base):
            base = "n" + base
        n = 1
        while f"{base}_{n}" in self._nodes:
            n += 1
        return f"{base}_{n}"

    def add_node(self, spec: Any, id: Optional[str] = None, *,
                 values: Optional[dict[str, Any]] = None,
                 config: Optional[dict[str, Any]] = None,
                 pos: Iterable[float] = (0.0, 0.0), label: str = "") -> Node:
        """Place a node.  ``spec`` is a spec id or an object with an ``id``."""
        spec_id = spec if isinstance(spec, str) else getattr(spec, "id", None)
        if not isinstance(spec_id, str) or not spec_id:
            raise GraphError(f"invalid node spec {spec!r}")
        if id is None:
            id = self._fresh_id(spec_id)
        _check_name("node id", id)
        if id in self._nodes:
            raise GraphError(f"a node with id {id!r} already exists")
        for pin in list(values or ()) + list(config or ()):
            _check_name("pin name", pin)
        xy = tuple(float(c) for c in pos)
        if len(xy) != 2:
            raise GraphError(f"invalid position {pos!r}")
        node = Node(id, spec_id, dict(values or {}), dict(config or {}),
                    (xy[0], xy[1]), str(label))
        self._nodes[id] = node
        return node

    def add_structure(self, kind: str, id: Optional[str] = None, *,
                      cases: Optional[list] = None,
                      pos: Iterable[float] = (0.0, 0.0), label: str = "") -> Structure:
        """Place a structure: ``"for"``, ``"while"``, ``"case"``, ``"subgraph"``,
        ``"try"`` or ``"with"``.

        A case structure takes ``cases``, one entry per frame: a list of the
        values that frame handles, or ``None`` for the default frame.
        """
        if kind not in STRUCTURE_KINDS:
            raise GraphError(f"structure kind must be one of {STRUCTURE_KINDS}")
        if (kind == "case") != (cases is not None):
            raise GraphError("'cases' is needed for a case structure, and only there")
        plain = self.add_node(f"structure.{kind}", id, pos=pos, label=label)
        structure = Structure(plain.id, plain.spec, pos=plain.pos, label=plain.label, kind=kind)
        self._nodes[plain.id] = structure
        try:
            if kind == "case":
                for match in cases or []:
                    structure.add_frame(match)
                if not structure.frames:
                    raise GraphError("a case structure needs at least one frame")
            elif kind == "try":
                structure.frames += [_new_frame(f"{plain.id}[0]"), _new_frame(f"{plain.id}[1]")]
            else:
                structure.frames.append(_new_frame(plain.id))
        except GraphError:
            del self._nodes[plain.id]
            raise
        return structure

    def add_for(self, id: Optional[str] = None, *, parallel: bool = False,
                **options: Any) -> Structure:
        """A For loop: runs its frame ``count`` times, or once per indexed element.

        With ``parallel`` the iterations may run at the same time; such a loop
        cannot have shift registers.
        """
        loop = self.add_structure("for", id, **options)
        loop.parallel = bool(parallel)
        return loop

    def add_while(self, id: Optional[str] = None, *, max_iterations: Optional[int] = None,
                  **options: Any) -> Structure:
        """A While loop: runs its frame until the inner ``stop`` pin is true."""
        loop = self.add_structure("while", id, **options)
        loop.max_iterations = max_iterations
        return loop

    def add_case(self, id: Optional[str] = None, *, cases: list, **options: Any) -> Structure:
        """A case structure: runs the frame that matches its ``selector``."""
        return self.add_structure("case", id, cases=cases, **options)

    def add_subgraph(self, id: Optional[str] = None, **options: Any) -> Structure:
        """A diagram used as one node."""
        return self.add_structure("subgraph", id, **options)

    def add_try(self, id: Optional[str] = None, **options: Any) -> Structure:
        """A Try structure: frame 0 is the body, frame 1 runs if a node in the
        body fails and receives the ``error`` message and its ``error_type``."""
        return self.add_structure("try", id, **options)

    def add_with(self, id: Optional[str] = None, **options: Any) -> Structure:
        """A With structure: enters the context manager wired to ``resource``,
        runs its frame, and always leaves the context manager again."""
        return self.add_structure("with", id, **options)

    def remove_node(self, ref: NodeRef) -> None:
        """Remove a node and every wire attached to it."""
        nid = self.node(ref).id
        for wire in [*self._in.get(nid, {}).values(), *self._out.get(nid, ())]:
            self._drop(wire)
        self._in.pop(nid, None)
        self._out.pop(nid, None)
        del self._nodes[nid]

    def connect(self, src: NodeRef, src_pin: str, dst: NodeRef, dst_pin: str, *,
                replace: bool = False) -> Wire:
        """Wire an output pin to an input pin."""
        wire = Wire(self.node(src).id, _check_name("pin name", src_pin),
                    self.node(dst).id, _check_name("pin name", dst_pin))
        old = self._in.get(wire.dst, {}).get(wire.dst_pin)
        if old == wire:
            return wire
        if old is not None and not replace:
            raise GraphError(
                f"input {wire.dst}.{wire.dst_pin} is already wired from "
                f"{old.src}.{old.src_pin}")
        if wire.src == wire.dst or self._reaches(wire.dst, wire.src):
            raise CycleError(f"wire {wire} would create a cycle")
        if old is not None:
            self._drop(old)
        self._in.setdefault(wire.dst, {})[wire.dst_pin] = wire
        self._out.setdefault(wire.src, set()).add(wire)
        return wire

    def disconnect(self, dst: NodeRef, dst_pin: str) -> bool:
        """Remove the wire arriving at an input pin.  True if there was one."""
        wire = self._in.get(self.node(dst).id, {}).get(dst_pin)
        if wire is None:
            return False
        self._drop(wire)
        return True

    def _drop(self, wire: Wire) -> None:
        del self._in[wire.dst][wire.dst_pin]
        self._out[wire.src].discard(wire)

    def _reaches(self, start: str, goal: str) -> bool:
        """Is there a directed path from ``start`` to ``goal``?"""
        stack, seen = [start], {start}
        while stack:
            current = stack.pop()
            if current == goal:
                return True
            for wire in self._out.get(current, ()):
                if wire.dst not in seen:
                    seen.add(wire.dst)
                    stack.append(wire.dst)
        return False

    # -- analysis -----------------------------------------------------------

    def topological_order(self) -> list[str]:
        """Node ids such that every node comes after the nodes it reads from.

        The order is deterministic and does not depend on editing history:
        among ready nodes, the smallest id goes first.
        """
        pending = {nid: 0 for nid in self._nodes}
        for wire in self._all_wires():
            pending[wire.dst] += 1
        rank = {nid: i for i, nid in enumerate(sorted(self._nodes))}
        ready = sorted((nid for nid, n in pending.items() if n == 0),
                       key=rank.__getitem__, reverse=True)
        order: list[str] = []
        while ready:
            nid = ready.pop()
            order.append(nid)
            freed = []
            for wire in self._out.get(nid, ()):
                pending[wire.dst] -= 1
                if pending[wire.dst] == 0:
                    freed.append(wire.dst)
            if freed:
                ready.extend(freed)
                ready.sort(key=rank.__getitem__, reverse=True)
        if len(order) != len(self._nodes):  # only possible with corrupted state
            raise CycleError("graph contains a cycle")
        return order

    # -- copying and comparing ----------------------------------------------

    def copy(self) -> "Graph":
        clone = Graph(self.name)
        clone.imports = list(self.imports)
        for node in self._nodes.values():
            base = (node.id, node.spec, copy.deepcopy(node.values),
                    copy.deepcopy(node.config), node.pos, node.label)
            if isinstance(node, Structure):
                clone._nodes[node.id] = Structure(
                    *base, kind=node.kind, frames=[f.copy() for f in node.frames],
                    inputs=[copy.deepcopy(t) for t in node.inputs],
                    outputs=[copy.deepcopy(t) for t in node.outputs],
                    registers=list(node.registers), cases=copy.deepcopy(node.cases),
                    max_iterations=node.max_iterations, parallel=node.parallel)
            else:
                clone._nodes[node.id] = Node(*base)
        for wire in self._all_wires():
            clone._in.setdefault(wire.dst, {})[wire.dst_pin] = wire
            clone._out.setdefault(wire.src, set()).add(wire)
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Graph):
            return NotImplemented
        try:
            return self.to_dict() == other.to_dict()
        except SerializationError:
            return self is other

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"<Graph {self.name!r}: {len(self._nodes)} nodes, {sum(map(len, self._in.values()))} wires>"

    # -- JSON -----------------------------------------------------------------

    def has_structures(self) -> bool:
        return any(isinstance(node, Structure) for node in self._nodes.values())

    def _body(self) -> dict[str, Any]:
        """Nodes and wires as JSON data, frames of structures included."""
        nodes = []
        for node in sorted(self._nodes.values(), key=lambda n: n.id):
            item: dict[str, Any] = {"id": node.id, "spec": node.spec}
            if node.values:
                item["values"] = {k: encode_value(v) for k, v in sorted(node.values.items())}
            if node.config:
                item["config"] = {k: encode_value(v) for k, v in sorted(node.config.items())}
            if node.pos != (0.0, 0.0):
                item["pos"] = [node.pos[0], node.pos[1]]
            if node.label:
                item["label"] = node.label
            if isinstance(node, Structure):
                item["structure"] = _structure_to_dict(node)
            nodes.append(item)
        return {"nodes": nodes,
                "wires": [[w.src, w.src_pin, w.dst, w.dst_pin] for w in self.wires]}

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON data.  Nodes and wires are sorted so files diff well."""
        for module in self.imports:
            if not isinstance(module, str) or not _MODULE.match(module):
                raise GraphError(f"invalid module name {module!r} in imports")
        return {
            "format": FORMAT,
            "version": 2 if self.has_structures() else 1,
            "name": self.name,
            "imports": sorted(set(self.imports)),
            **self._body(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Graph":
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise GraphError("not an OpenRamen graph file")
        version = data.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise GraphError(f"invalid graph file version {version!r}")
        if version > VERSION:
            raise GraphError(
                f"graph file version {version} is newer than this OpenRamen "
                f"understands (version {VERSION})")
        name = data.get("name", "")
        if not isinstance(name, str):
            raise GraphError("graph name must be a string")
        graph = cls(name)
        imports = data.get("imports", [])
        if not isinstance(imports, list) or not all(
                isinstance(m, str) and _MODULE.match(m) for m in imports):
            raise GraphError("'imports' must be a list of module names")
        graph.imports = sorted(set(imports))
        graph._load_body(data)
        return graph

    def _load_body(self, data: dict[str, Any]) -> None:
        nodes = data.get("nodes", [])
        wires = data.get("wires", [])
        if not isinstance(nodes, list) or not isinstance(wires, list):
            raise GraphError("'nodes' and 'wires' must be lists")
        for item in nodes:
            if not isinstance(item, dict) or "id" not in item or "spec" not in item:
                raise GraphError(f"malformed node entry {item!r}")
            values, config = item.get("values", {}), item.get("config", {})
            pos = item.get("pos", (0.0, 0.0))
            label = item.get("label", "")
            if not isinstance(values, dict) or not isinstance(config, dict):
                raise GraphError(f"node {item['id']!r}: 'values' and 'config' must be objects")
            if (not isinstance(pos, (list, tuple)) or len(pos) != 2
                    or not all(isinstance(c, (int, float)) and not isinstance(c, bool)
                               for c in pos)):
                raise GraphError(f"node {item['id']!r}: invalid position {pos!r}")
            if not isinstance(label, str):
                raise GraphError(f"node {item['id']!r}: label must be a string")
            try:
                node = self.add_node(
                    item["spec"], item["id"],
                    values={k: decode_value(v) for k, v in values.items()},
                    config={k: decode_value(v) for k, v in config.items()},
                    pos=pos, label=label)
                if "structure" in item:
                    self._nodes[node.id] = _structure_from_dict(node, item["structure"])
            except SerializationError as exc:
                raise GraphError(f"node {item['id']!r}: {exc}") from exc
        for item in wires:
            if (not isinstance(item, (list, tuple)) or len(item) != 4
                    or not all(isinstance(part, str) for part in item)):
                raise GraphError(f"malformed wire entry {item!r}")
            self.connect(*item)

    def dumps(self) -> str:
        """The file form: one node and one wire per line, so files diff well."""
        data = self.to_dict()
        if not data["imports"]:
            del data["imports"]
        return _render(data, 0) + "\n"

    @classmethod
    def loads(cls, text: str) -> "Graph":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GraphError(f"not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def save(self, path: "str | os.PathLike[str]") -> None:
        text = self.dumps()  # serialise fully before touching the file
        tmp = f"{os.fspath(path)}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: "str | os.PathLike[str]") -> "Graph":
        with open(path, encoding="utf-8") as fh:
            return cls.loads(fh.read())


# --------------------------------------------------------------------------
# Structures as JSON
# --------------------------------------------------------------------------

def _tunnel_to_dict(tunnel: Tunnel) -> dict[str, Any]:
    item: dict[str, Any] = {"name": tunnel.name, "mode": tunnel.mode}
    if tunnel.type:
        item["type"] = tunnel.type
    if tunnel.fallback is not UNSET:
        item["fallback"] = encode_value(tunnel.fallback)
    return item


def _structure_to_dict(node: Structure) -> dict[str, Any]:
    item: dict[str, Any] = {
        "kind": node.kind,
        "inputs": [_tunnel_to_dict(t) for t in node.inputs],
        "outputs": [_tunnel_to_dict(t) for t in node.outputs],
    }
    if node.registers:
        item["registers"] = list(node.registers)
    if node.kind == "case":
        item["cases"] = [
            None if match is None else
            [{"range": list(e)} if isinstance(e, tuple) else e for e in match]
            for match in node.cases]
    if node.max_iterations is not None:
        item["max_iterations"] = node.max_iterations
    if node.parallel:
        item["parallel"] = True
    item["frames"] = [frame._body() for frame in node.frames]
    return item


def _structure_from_dict(node: Node, data: Any) -> Structure:
    where = f"structure {node.id!r}"
    if not isinstance(data, dict) or data.get("kind") not in STRUCTURE_KINDS:
        raise GraphError(f"{where}: malformed structure")
    kind = data["kind"]
    if node.spec != f"structure.{kind}":
        raise GraphError(f"{where}: spec {node.spec!r} does not match kind {kind!r}")
    structure = Structure(node.id, node.spec, node.values, node.config, node.pos, node.label,
                          kind=kind)
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames or not all(isinstance(f, dict) for f in frames):
        raise GraphError(f"{where}: 'frames' must be a non-empty list")
    if kind != "case" and len(frames) != (2 if kind == "try" else 1):
        raise GraphError(f"{where}: a {kind} structure has exactly "
                         + ("two frames" if kind == "try" else "one frame"))
    for index, body in enumerate(frames):
        frame = Graph(f"{node.id}[{index}]" if kind in ("case", "try") else node.id)
        frame._load_body(body)
        structure.frames.append(frame)
    try:
        for direction, add in (("inputs", structure.add_input), ("outputs", structure.add_output)):
            tunnels = data.get(direction, [])
            if not isinstance(tunnels, list):
                raise GraphError(f"'{direction}' must be a list")
            for tunnel in tunnels:
                if not isinstance(tunnel, dict) or not isinstance(tunnel.get("name"), str):
                    raise GraphError(f"malformed tunnel {tunnel!r}")
                options: dict[str, Any] = {"mode": tunnel.get("mode"), "type": tunnel.get("type", "")}
                if "fallback" in tunnel:
                    if direction == "inputs":
                        raise GraphError("an input tunnel has no fallback")
                    options["fallback"] = decode_value(tunnel["fallback"])
                add(tunnel["name"], **options)
        registers = data.get("registers", [])
        if not isinstance(registers, list):
            raise GraphError("'registers' must be a list")
        for register in registers:
            structure.add_register(register)
        if kind == "case":
            cases = data.get("cases")
            if not isinstance(cases, list) or len(cases) != len(frames):
                raise GraphError("'cases' must have one entry per frame")
            structure.cases = [
                _check_match(None if match is None else [
                    tuple(e["range"]) if isinstance(e, dict) and isinstance(e.get("range"), list)
                    else e for e in match] if isinstance(match, list) else match)
                for match in cases]
        elif "cases" in data:
            raise GraphError("only a case structure has 'cases'")
        limit = data.get("max_iterations")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise GraphError("'max_iterations' must be a positive integer")
        structure.max_iterations = limit
        if data.get("parallel"):
            if kind != "for" or structure.registers:
                raise GraphError("only a For loop without shift registers can be parallel")
            structure.parallel = True
    except GraphError as exc:
        raise GraphError(f"{where}: {exc}") from exc
    return structure


_EXPANDED = ("nodes", "wires", "frames")


def _render(value: Any, indent: int, key: str = "") -> str:
    """JSON text with one node, wire and frame per line and the rest compact."""
    pad = "  " * indent
    if isinstance(value, dict) and any(k in value for k in (*_EXPANDED, "structure")):
        lines = [f"{pad}  {json.dumps(k, ensure_ascii=False)}: {_render(v, indent + 1, k)}"
                 for k, v in value.items()]
        return "{\n" + ",\n".join(lines) + "\n" + pad + "}"
    if isinstance(value, list) and value and key in _EXPANDED:
        lines = [f"{pad}  {_render(v, indent + 1)}" for v in value]
        return "[\n" + ",\n".join(lines) + "\n" + pad + "]"
    return json.dumps(value, allow_nan=False, ensure_ascii=False)
