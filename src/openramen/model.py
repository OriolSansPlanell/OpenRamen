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

from .serial import SerializationError, decode_value, encode_value

__all__ = ["Graph", "Node", "Wire", "GraphError", "CycleError", "FORMAT", "VERSION"]

FORMAT = "openramen.graph"
VERSION = 1

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
            clone._nodes[node.id] = Node(node.id, node.spec, copy.deepcopy(node.values),
                                         copy.deepcopy(node.config), node.pos, node.label)
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

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON data.  Nodes and wires are sorted so files diff well."""
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
            nodes.append(item)
        for module in self.imports:
            if not isinstance(module, str) or not _MODULE.match(module):
                raise GraphError(f"invalid module name {module!r} in imports")
        return {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "imports": sorted(set(self.imports)),
            "nodes": nodes,
            "wires": [[w.src, w.src_pin, w.dst, w.dst_pin] for w in self.wires],
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
                graph.add_node(
                    item["spec"], item["id"],
                    values={k: decode_value(v) for k, v in values.items()},
                    config={k: decode_value(v) for k, v in config.items()},
                    pos=pos, label=label)
            except SerializationError as exc:
                raise GraphError(f"node {item['id']!r}: {exc}") from exc
        for item in wires:
            if (not isinstance(item, (list, tuple)) or len(item) != 4
                    or not all(isinstance(part, str) for part in item)):
                raise GraphError(f"malformed wire entry {item!r}")
            graph.connect(*item)
        return graph

    def dumps(self) -> str:
        """The file form: one node and one wire per line, so files diff well."""
        data = self.to_dict()

        def line(item: Any) -> str:
            return json.dumps(item, allow_nan=False, ensure_ascii=False)

        def block(items: list) -> str:
            if not items:
                return "[]"
            return "[\n" + ",\n".join("    " + line(i) for i in items) + "\n  ]"

        return (
            "{\n"
            f'  "format": {line(data["format"])},\n'
            f'  "version": {line(data["version"])},\n'
            f'  "name": {line(data["name"])},\n'
            + (f'  "imports": {line(data["imports"])},\n' if data["imports"] else "") +
            f'  "nodes": {block(data["nodes"])},\n'
            f'  "wires": {block(data["wires"])}\n'
            "}\n"
        )

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
