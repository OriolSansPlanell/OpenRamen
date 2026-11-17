"""Graph model and JSON form."""

import json
import math
import random
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest

from openramen.model import FORMAT, VERSION, CycleError, Graph, GraphError, Node, Wire
from openramen.serial import SerializationError, decode_value, encode_value


def roundtrip(value):
    data = encode_value(value)
    text = json.dumps(data, allow_nan=False)      # must be strict JSON
    return decode_value(json.loads(text))


class TestLiterals:
    @pytest.mark.parametrize("value", [
        None, True, False, 0, -1, 2**70, 1.5, -0.0, "", "text", "\u30e9\u30fc\u30e1\u30f3",
        [], [1, 2.5, "a", None], [[1], [2, [3]]], {}, {"a": 1, "b": [1, 2]},
        (), (1, "a"), (1, (2, 3)), [(), [()]],
        1j, complex(1.5, -2), b"", b"\x00\xff", {1: "a", 2.5: "b"}, {(1, 2): [3]},
        {"$weird": 1}, {"a": {"$b": (1,)}}, {None: 1, True: 2},
    ])
    def test_roundtrip_is_exact(self, value):
        out = roundtrip(value)
        assert out == value and type(out) is type(value)
        assert json.dumps(encode_value(out), sort_keys=True) == \
            json.dumps(encode_value(value), sort_keys=True)

    def test_element_types_survive(self):
        out = roundtrip([1, 1.0, True, (1,), [1]])
        assert [type(x) for x in out] == [int, float, bool, tuple, list]

    def test_non_finite_floats(self):
        assert roundtrip(math.inf) == math.inf and roundtrip(-math.inf) == -math.inf
        assert math.isnan(roundtrip(math.nan))
        z = roundtrip(complex(math.nan, math.inf))
        assert math.isnan(z.real) and z.imag == math.inf
        assert math.isnan(roundtrip([math.nan])[0])

    def test_paths(self):
        assert roundtrip(Path("a/b.txt")) == Path("a/b.txt")
        assert isinstance(roundtrip(Path("a")), Path)          # usable, not just pure
        assert roundtrip(PurePosixPath("/x/y")) == PurePosixPath("/x/y")
        win = roundtrip(PureWindowsPath(r"C:\data\x.tif"))
        assert isinstance(win, PureWindowsPath) and win == PureWindowsPath(r"C:\data\x.tif")

    @pytest.mark.parametrize("array", [
        np.zeros((2, 3)), np.arange(6, dtype="uint8").reshape(3, 2), np.array(2.5),
        np.array([], dtype="int32"), np.zeros((0, 4)), np.array([True, False]),
        np.array([1 + 2j, 3j], dtype="complex64"), np.array([np.nan, np.inf, -1.0]),
        np.arange(24, dtype="float32").reshape(2, 3, 4),
    ])
    def test_arrays(self, array):
        out = roundtrip(array)
        assert isinstance(out, np.ndarray)
        assert out.dtype == array.dtype and out.shape == array.shape
        assert np.array_equal(out, array, equal_nan=True)

    def test_numpy_scalars_become_python_scalars(self):
        assert roundtrip(np.float32(1.5)) == 1.5 and type(roundtrip(np.float32(1.5))) is float
        assert type(roundtrip(np.int16(3))) is int
        assert roundtrip(np.bool_(True)) is True
        assert roundtrip(np.str_("a")) == "a"
        assert roundtrip(np.complex64(1j)) == 1j

    @pytest.mark.parametrize("value", [
        object(), {1, 2}, len, np.array(["a"]), np.array([object()]),
        np.datetime64("2020-01-01"), [object()], {"a": {1, 2}}, bytearray(b"a"),
    ])
    def test_unsupported_values_are_refused(self, value):
        with pytest.raises(SerializationError):
            encode_value(value)

    @pytest.mark.parametrize("data", [
        {"$unknown": 1}, {"$complex": [1]}, {"$complex": "x"}, {"$float": "big"},
        {"$bytes": "not base64!"}, {"$tuple": 3}, {"$dict": [[1]]}, {"$dict": [[[1], 2]]},
        {"$ndarray": {"dtype": "float64"}}, {"$ndarray": {"dtype": "object", "shape": [1], "data": [1]}},
        {"$ndarray": {"dtype": "float64", "shape": [5], "data": [1.0]}},
        {"$a": 1, "$b": 2}, {"$path": 3}, {"$bytes": 5},
    ])
    def test_malformed_literals_are_refused(self, data):
        with pytest.raises(SerializationError):
            decode_value(data)

    @pytest.mark.parametrize("seed", range(200))
    def test_random_roundtrip(self, seed):
        rng = random.Random(seed)

        def make(depth=0):
            leaf = [None, True, 3, -2.5, "s", b"b", 1j, math.inf, PurePosixPath("/p")]
            if depth > 2 or rng.random() < 0.4:
                return rng.choice(leaf)
            kind = rng.choice(["list", "tuple", "dict", "idict", "array"])
            if kind == "list":
                return [make(depth + 1) for _ in range(rng.randint(0, 3))]
            if kind == "tuple":
                return tuple(make(depth + 1) for _ in range(rng.randint(0, 3)))
            if kind == "dict":
                return {rng.choice(["a", "b", "$c"]): make(depth + 1) for _ in range(2)}
            if kind == "idict":
                return {rng.randint(0, 5): make(depth + 1) for _ in range(2)}
            return np.arange(rng.randint(0, 5), dtype=rng.choice(["int8", "float64"]))

        def same(a, b):
            if isinstance(a, np.ndarray):
                return (isinstance(b, np.ndarray) and a.dtype == b.dtype
                        and np.array_equal(a, b))
            if type(a) is not type(b) and not (isinstance(a, PurePosixPath)
                                                and isinstance(b, PurePosixPath)):
                return False
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
            return a == b

        value = make()
        assert same(roundtrip(value), value)


def diamond():
    g = Graph("diamond")
    g.add_node("base.constant", "src", config={"value": 1})
    g.add_node("base.add", "left", values={"b": 1})
    g.add_node("base.add", "right", values={"b": 2})
    g.add_node("base.multiply", "join")
    g.connect("src", "out", "left", "a")
    g.connect("src", "out", "right", "a")
    g.connect("left", "out", "join", "a")
    g.connect("right", "out", "join", "b")
    return g


class TestEditing:
    def test_add_node_generates_readable_unique_ids(self):
        g = Graph()
        assert g.add_node("base.add").id == "add_1"
        assert g.add_node("base.add").id == "add_2"
        assert g.add_node("numpy.linalg.Norm").id == "norm_1"
        assert g.add_node("pkg.2d-thing").id == "n2d_thing_1"
        g.remove_node("add_1")
        assert g.add_node("base.add").id == "add_1"
        assert len(g) == 4 and "add_2" in g and "nope" not in g and 3 not in g

    def test_add_node_accepts_spec_objects(self):
        from openramen.library import default_registry
        spec = default_registry().get("base.add")
        g = Graph()
        assert g.add_node(spec).spec == "base.add"

    @pytest.mark.parametrize("bad_id", ["", "1a", "a b", "a.b", "a-b", "\u00e9", 3, None])
    def test_invalid_ids_are_refused(self, bad_id):
        g = Graph()
        if bad_id is None:
            g.add_node("base.add", None)  # None means "generate one"
            return
        with pytest.raises(GraphError):
            g.add_node("base.add", bad_id)

    def test_invalid_nodes_are_refused(self):
        g = Graph()
        g.add_node("base.add", "a")
        with pytest.raises(GraphError, match="already exists"):
            g.add_node("base.add", "a")
        for bad_spec in ["", None, 3, object()]:
            with pytest.raises(GraphError):
                g.add_node(bad_spec)
        with pytest.raises(GraphError):
            g.add_node("base.add", values={"not a pin": 1})
        with pytest.raises(GraphError):
            g.add_node("base.add", pos=(1, 2, 3))
        assert len(g) == 1

    def test_node_values_are_copied_from_the_caller(self):
        values = {"a": 1}
        g = Graph()
        node = g.add_node("base.add", values=values)
        values["a"] = 99
        assert node.values == {"a": 1}

    def test_connect_and_query(self):
        g = diamond()
        assert g.incoming("join") == {"a": Wire("left", "out", "join", "a"),
                                      "b": Wire("right", "out", "join", "b")}
        assert g.incoming("src") == {}
        assert [w.dst for w in g.outgoing("src")] == ["left", "right"]
        assert g.outgoing("src", "nope") == [] and g.outgoing("join") == []
        assert len(g.wires) == 4 and g.wires == sorted(g.wires)
        assert str(g.wires[0]) == "left.out -> join.a"

    def test_connect_accepts_node_objects(self):
        g = Graph()
        a, b = g.add_node("base.constant"), g.add_node("base.negate")
        wire = g.connect(a, "out", b, "x")
        assert wire == Wire(a.id, "out", b.id, "x") and g.node(a) is a

    def test_an_input_takes_one_wire(self):
        g = diamond()
        with pytest.raises(GraphError, match="already wired"):
            g.connect("src", "out", "join", "a")
        assert g.incoming("join")["a"].src == "left"       # unchanged
        g.connect("src", "out", "join", "a", replace=True)
        assert g.incoming("join")["a"].src == "src"
        assert g.outgoing("left") == []
        # connecting the same wire twice is harmless
        assert g.connect("src", "out", "join", "a") == Wire("src", "out", "join", "a")
        assert len(g.wires) == 4

    def test_cycles_are_refused(self):
        g = diamond()
        with pytest.raises(CycleError):
            g.connect("join", "out", "src", "x")
        with pytest.raises(CycleError):
            g.connect("left", "out", "left", "b")
        with pytest.raises(CycleError):
            g.connect("join", "out", "left", "b")
        assert len(g.wires) == 4
        g.topological_order()

    def test_failed_replace_keeps_the_old_wire(self):
        g = diamond()
        with pytest.raises(CycleError):
            g.connect("join", "out", "left", "a", replace=True)
        assert g.incoming("left")["a"].src == "src"

    def test_unknown_nodes_and_bad_pin_names(self):
        g = diamond()
        with pytest.raises(GraphError, match="no node"):
            g.connect("ghost", "out", "join", "a")
        with pytest.raises(GraphError):
            g.connect("src", "out", "ghost", "a")
        with pytest.raises(GraphError):
            g.connect("src", "bad pin", "left", "b")
        with pytest.raises(GraphError):
            g.node("ghost")
        with pytest.raises(GraphError):
            g.incoming("ghost")
        with pytest.raises(GraphError):
            g.remove_node("ghost")

    def test_disconnect(self):
        g = diamond()
        assert g.disconnect("join", "a") is True
        assert g.disconnect("join", "a") is False
        assert "a" not in g.incoming("join") and g.outgoing("left") == []
        g.connect("right", "out", "join", "a")     # the pin is free again

    def test_remove_node_removes_its_wires(self):
        g = diamond()
        g.remove_node("left")
        assert "left" not in g and len(g.wires) == 2
        assert g.incoming("join") == {"b": Wire("right", "out", "join", "b")}
        assert [w.dst for w in g.outgoing("src")] == ["right"]
        g.remove_node("src")
        assert g.wires == [Wire("right", "out", "join", "b")]
        # a node that comes back under the same id carries no stale edges
        g.add_node("base.add", "left")
        g.connect("join", "out", "left", "a")
        assert g.topological_order() == ["right", "join", "left"]


class TestTopologicalOrder:
    def test_dependencies_come_first(self):
        order = diamond().topological_order()
        assert order.index("src") < order.index("left") < order.index("join")
        assert order.index("right") < order.index("join")

    def test_order_does_not_depend_on_editing_history(self):
        a = diamond()
        b = Graph("diamond")
        for nid in ["join", "right", "left", "src"]:
            b.add_node(a.node(nid).spec, nid, values=a.node(nid).values,
                       config=a.node(nid).config)
        for wire in reversed(a.wires):
            b.connect(wire.src, wire.src_pin, wire.dst, wire.dst_pin)
        assert a.topological_order() == b.topological_order()
        assert a == b

    def test_empty_and_disconnected(self):
        assert Graph().topological_order() == []
        g = Graph()
        for name in ["c", "a", "b"]:
            g.add_node("base.constant", name)
        assert g.topological_order() == ["a", "b", "c"]

    @pytest.mark.parametrize("seed", range(100))
    def test_random_dags(self, seed):
        rng = random.Random(seed)
        g = Graph()
        ids = [g.add_node("base.identity").id for _ in range(rng.randint(1, 30))]
        attempts = accepted = 0
        for _ in range(60):
            a, b = rng.choice(ids), rng.choice(ids)
            attempts += 1
            try:
                g.connect(a, "out", b, f"p{rng.randint(0, 3)}")
                accepted += 1
            except GraphError:
                pass
        order = g.topological_order()
        assert sorted(order) == sorted(ids)
        position = {nid: i for i, nid in enumerate(order)}
        assert all(position[w.src] < position[w.dst] for w in g.wires)
        assert order == Graph.loads(g.dumps()).topological_order()
        # remove a few nodes: the graph must stay consistent
        for nid in rng.sample(ids, k=len(ids) // 3):
            g.remove_node(nid)
        assert all(w.src in g and w.dst in g for w in g.wires)
        assert sorted(g.topological_order()) == sorted(n.id for n in g)
        for node in g:
            assert all(w.dst == node.id for w in g.incoming(node).values())
            assert all(w.src == node.id for w in g.outgoing(node))


class TestJson:
    def test_roundtrip(self, tmp_path):
        g = diamond()
        g.node("left").pos = (10.0, -5.5)
        g.node("left").label = "plus one"
        g.node("right").values["b"] = (1, 2j)
        path = tmp_path / "g.json"
        g.save(path)
        loaded = Graph.load(path)
        assert loaded == g and loaded is not g
        assert loaded.name == "diamond"
        assert loaded.node("left").pos == (10.0, -5.5)
        assert loaded.node("left").label == "plus one"
        assert loaded.node("right").values["b"] == (1, 2j)
        assert loaded.dumps() == g.dumps()
        assert not list(tmp_path.glob("*.tmp"))

    def test_file_is_strict_json_with_one_item_per_line(self):
        text = diamond().dumps()
        data = json.loads(text)
        assert data["format"] == FORMAT and data["version"] == VERSION
        assert [n["id"] for n in data["nodes"]] == ["join", "left", "right", "src"]
        assert text.endswith("}\n")
        lines = text.splitlines()
        assert sum(line.lstrip().startswith('{"id"') for line in lines) == 4
        assert sum(line.lstrip().startswith('["') for line in lines) == 4

    def test_output_is_stable(self):
        g = diamond()
        assert g.dumps() == g.copy().dumps() == Graph.loads(g.dumps()).dumps()

    def test_empty_graph(self):
        g = Graph.loads(Graph("nothing").dumps())
        assert len(g) == 0 and g.wires == [] and g.name == "nothing"

    def test_unicode_survives(self):
        g = Graph("r\u00e2men \u30e9\u30fc\u30e1\u30f3 \"quoted\"\n")
        g.add_node("base.constant", config={"value": "\u00e5\u00e4\u00f6"}, label="\u00b5m")
        assert Graph.loads(g.dumps()) == g

    def test_a_failed_save_leaves_the_old_file(self, tmp_path):
        path = tmp_path / "g.json"
        diamond().save(path)
        before = path.read_text()
        bad = diamond()
        bad.node("src").config["value"] = object()
        with pytest.raises(SerializationError):
            bad.save(path)
        assert path.read_text() == before

    def test_copy_is_independent(self):
        g = diamond()
        g.node("src").config["value"] = [1, 2]
        clone = g.copy()
        assert clone == g
        clone.node("src").config["value"].append(3)
        clone.remove_node("left")
        clone.add_node("base.add", "extra")
        assert g.node("src").config["value"] == [1, 2]
        assert "left" in g and "extra" not in g and len(g.wires) == 4
        assert clone != g

    def test_equality(self):
        assert diamond() == diamond()
        other = diamond()
        other.node("left").values["b"] = 5
        assert diamond() != other
        assert diamond() != "diamond"
        with pytest.raises(TypeError):
            hash(diamond())

    @pytest.mark.parametrize("text, message", [
        ("not json", "not valid JSON"),
        ("[]", "not an OpenRamen graph"),
        ("{}", "not an OpenRamen graph"),
        ('{"format": "something.else", "version": 1}', "not an OpenRamen graph"),
        ('{"format": "openramen.graph"}', "version"),
        ('{"format": "openramen.graph", "version": "1"}', "version"),
        ('{"format": "openramen.graph", "version": 0}', "version"),
        ('{"format": "openramen.graph", "version": true}', "version"),
        ('{"format": "openramen.graph", "version": 999}', "newer"),
        ('{"format": "openramen.graph", "version": 1, "name": 3}', "name"),
        ('{"format": "openramen.graph", "version": 1, "nodes": {}}', "lists"),
        ('{"format": "openramen.graph", "version": 1, "nodes": [3]}', "malformed node"),
        ('{"format": "openramen.graph", "version": 1, "nodes": [{"id": "a"}]}', "malformed node"),
        ('{"format": "openramen.graph", "version": 1, "nodes": [{"id": "a b", "spec": "x"}]}',
         "invalid node id"),
        ('{"format": "openramen.graph", "version": 1, "nodes": [{"id": "a", "spec": ""}]}',
         "invalid node spec"),
        ('{"format": "openramen.graph", "version": 1, "nodes": '
         '[{"id": "a", "spec": "x"}, {"id": "a", "spec": "x"}]}', "already exists"),
        ('{"format": "openramen.graph", "version": 1, "nodes": '
         '[{"id": "a", "spec": "x", "values": []}]}', "must be objects"),
        ('{"format": "openramen.graph", "version": 1, "nodes": '
         '[{"id": "a", "spec": "x", "pos": [1]}]}', "position"),
        ('{"format": "openramen.graph", "version": 1, "nodes": '
         '[{"id": "a", "spec": "x", "pos": ["1", "2"]}]}', "position"),
        ('{"format": "openramen.graph", "version": 1, "nodes": '
         '[{"id": "a", "spec": "x", "label": 3}]}', "label"),
        ('{"format": "openramen.graph", "version": 1, "nodes": '
         '[{"id": "a", "spec": "x", "values": {"p": {"$nope": 1}}}]}', "unknown literal tag"),
        ('{"format": "openramen.graph", "version": 1, "wires": [["a", "o", "b"]]}',
         "malformed wire"),
        ('{"format": "openramen.graph", "version": 1, "wires": [["a", "o", "b", 1]]}',
         "malformed wire"),
        ('{"format": "openramen.graph", "version": 1, "wires": [["a", "o", "b", "i"]]}',
         "no node"),
        ('{"format": "openramen.graph", "version": 1, "nodes": [{"id": "a", "spec": "x"}], '
         '"wires": [["a", "o", "a", "i"]]}', "cycle"),
        ('{"format": "openramen.graph", "version": 1, "nodes": '
         '[{"id": "a", "spec": "x"}, {"id": "b", "spec": "x"}], '
         '"wires": [["a", "o", "b", "i"], ["b", "o", "a", "i"]]}', "cycle"),
        ('{"format": "openramen.graph", "version": 1, "nodes": '
         '[{"id": "a", "spec": "x"}, {"id": "b", "spec": "x"}, {"id": "c", "spec": "x"}], '
         '"wires": [["a", "o", "c", "i"], ["b", "o", "c", "i"]]}', "already wired"),
    ])
    def test_bad_files_are_refused_with_a_clear_message(self, text, message):
        with pytest.raises(GraphError, match=message):
            Graph.loads(text)

    def test_unknown_keys_are_ignored_for_forward_compatibility(self):
        text = ('{"format": "openramen.graph", "version": 1, "future": 1, "nodes": '
                '[{"id": "a", "spec": "x", "colour": "red"}]}')
        assert Graph.loads(text).node("a").spec == "x"

    def test_missing_file(self, tmp_path):
        with pytest.raises(OSError):
            Graph.load(tmp_path / "missing.json")

    def test_node_dataclass_defaults(self):
        node = Node("a", "base.add")
        assert node.values == {} and node.config == {} and node.pos == (0.0, 0.0)
        assert "diamond" in repr(diamond()) and "4 nodes" in repr(diamond())
