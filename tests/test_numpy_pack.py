"""The importer and the NumPy node pack, measured on real NumPy calls."""

import numpy as np
import pytest

import openramen as ramen
from openramen import Graph, analyze, run
from openramen import types as T
from openramen.library import report

from numpy_cases import CASES, MODULES, build, expected, outcome, registry, same

PACKED = registry(pack=True)
RAW = registry(pack=False)
IDS = [f"{name}{'-' + '-'.join(kwargs) if kwargs else ''}-{i}"
       for i, (name, args, kwargs) in enumerate(CASES)]


@pytest.mark.parametrize("name, args, kwargs", CASES, ids=IDS)
def test_node_returns_what_numpy_returns(name, args, kwargs):
    """With the pack applied, every case runs and gives NumPy's own answer."""
    kind, detail = outcome(PACKED, name, args, kwargs)
    assert kind == "ok", f"{name}: {kind}: {detail}"


def test_raw_import_needs_almost_no_manual_fixes():
    """The published coverage figure: cases that run right from introspection alone."""
    outcomes = [outcome(RAW, *case)[0] for case in CASES]
    share = outcomes.count("ok") / len(CASES)
    assert len(CASES) >= 160 and len({c[0] for c in CASES}) >= 150
    assert share >= 0.95, sorted(
        {c[0] for c, o in zip(CASES, outcomes) if o != "ok"})


def test_every_public_function_is_importable():
    for provider in PACKED.providers:
        summary = report(provider)
        assert summary.failures == {}, summary.failures
        assert summary.imported + summary.skipped == summary.candidates
        assert summary.documented == summary.imported
        assert summary.typed_inputs / summary.inputs > 0.6
    numpy_summary = report([p for p in RAW.providers if p.prefix == "numpy"][0])
    assert numpy_summary.candidates > 300 and numpy_summary.skipped == 0


def test_every_spec_is_well_formed():
    count = 0
    for spec in PACKED:
        if not spec.id.startswith("numpy."):
            continue
        count += 1
        assert spec.summary, spec.id
        assert spec.tier in ("auto", "annotated") and callable(spec.resolve())
        for pin in spec.inputs + spec.outputs + spec.config:
            assert T.parse_type(str(pin.type)) == pin.type, (spec.id, pin.name)
            assert pin.source in ("", "annotation", "docstring", "default", "override")
        for pin in spec.inputs:
            if pin.default is not ramen.MISSING and pin.default is not ramen.DEFAULT \
                    and not pin.variadic:
                T.accept(T.plain_scalar(pin.default), pin.type)   # defaults always fit
    assert count > 350


class TestPackContents:
    def test_named_tuple_results_get_one_pin_per_field(self):
        spec = PACKED.get("numpy.unique_counts")
        assert [p.name for p in spec.outputs] == ["values", "counts"]
        g = build(PACKED, "unique_counts", (np.array([3, 1, 3]),), {})
        result = run(g, PACKED)["n"]
        assert result["values"].tolist() == [1, 3] and result["counts"].tolist() == [1, 2]

    def test_variants(self):
        spec = PACKED.get("numpy.linspace_step")
        assert "retstep" not in [p.name for p in spec.inputs + spec.config]
        assert [p.name for p in spec.outputs] == ["samples", "step"]
        g = build(PACKED, "linspace_step", (0.0, 1.0), {"num": 5})
        result = run(g, PACKED)["n"]
        assert result["step"] == 0.25 and result["samples"].tolist() == [0, 0.25, 0.5, 0.75, 1.0]
        g = build(PACKED, "intersect1d_indices", ([1, 2, 3], [3, 1, 9]), {})
        result = run(g, PACKED)["n"]
        assert [result[k].tolist() for k in ("values", "index1", "index2")] == [[1, 3], [0, 2], [1, 0]]
        g = build(PACKED, "average_weights", ([1.0, 3.0],), {"weights": [1.0, 3.0]})
        assert run(g, PACKED)["n"] == {"average": 2.5, "sum_of_weights": 4.0}

    def test_flags_are_configuration(self):
        spec = PACKED.get("numpy.unique")
        assert {"return_index", "return_inverse", "return_counts"} <= {p.name for p in spec.config}
        plain = build(PACKED, "unique", ([3, 1, 3],), {})
        flagged = build(PACKED, "unique", ([3, 1, 3],), {"return_counts": True})
        assert str(analyze(plain, PACKED).out_type("n", "out")) == \
            "Union[Array[_, _], Bool, Complex, Float, Int]"
        assert str(analyze(flagged, PACKED).out_type("n", "out")) == "Any"
        values, counts = run(flagged, PACKED)["n"]
        assert values.tolist() == [1, 3] and counts.tolist() == [1, 2]

    def test_views_and_in_place_functions_are_marked(self):
        for name, source in [("reshape", "a"), ("ravel", "a"), ("transpose", "a"), ("flip", "m")]:
            assert PACKED.get(f"numpy.{name}").outputs[0].aliases == source, name
        for name, pin in [("fill_diagonal", "a"), ("put", "a"), ("copyto", "dst")]:
            spec = PACKED.get(f"numpy.{name}")
            assert spec.input(pin).access == "mutate" and not spec.pure, name
            assert spec.outputs == ()
        assert PACKED.get("numpy.add").input("out").access == "mutate"   # found by the importer

    def test_utilities_are_left_out(self):
        for name in ("show_config", "info", "seterr", "set_printoptions"):
            assert PACKED.find(f"numpy.{name}") is None
            assert RAW.find(f"numpy.{name}") is not None

    def test_impure_functions_are_recognised(self):
        registry = ramen.default_registry()
        registry.import_module("numpy.random")
        assert not registry.get("numpy.random.rand").pure
        assert not PACKED.get("numpy.save").pure and not PACKED.get("numpy.loadtxt").pure
        assert PACKED.get("numpy.sum").pure and PACKED.get("numpy.linalg.norm").pure


def test_a_pipeline_of_imported_nodes():
    g = Graph("pipeline")
    g.imports = list(MODULES)
    g.add_node("base.constant", "image", config={"value": np.arange(12.0).reshape(3, 4)})
    g.add_node("numpy.fft.fft2", "spectrum")
    g.add_node("numpy.abs", "magnitude")
    g.add_node("numpy.max", "peak")
    g.add_node("numpy.linalg.norm", "norm")
    g.add_node("base.divide", "ratio")
    g.connect("image", "out", "spectrum", "a")
    g.connect("spectrum", "out", "magnitude", "x")
    g.connect("magnitude", "out", "peak", "a")
    g.connect("image", "out", "norm", "x")
    g.connect("peak", "out", "ratio", "a")
    g.connect("norm", "out", "ratio", "b")
    analysis = analyze(g)
    assert analysis.ok, analysis.problems
    image = np.arange(12.0).reshape(3, 4)
    want = np.abs(np.fft.fft2(image)).max() / np.linalg.norm(image)
    assert run(g)["ratio"] == pytest.approx(want)
    assert run(Graph.loads(g.dumps()))["ratio"] == pytest.approx(want)


@pytest.mark.parametrize("module, minimum", [("scipy.ndimage", 60), ("skimage.filters", 30)])
def test_other_scientific_libraries_import_too(module, minimum):
    pytest.importorskip(module.split(".")[0])
    registry = ramen.default_registry()
    library = registry.import_module(module)
    summary = report(library)
    assert summary.imported >= minimum and summary.failures == {}
    assert summary.typed_inputs / summary.inputs > 0.6


def test_scipy_gaussian_filter_end_to_end():
    ndimage = pytest.importorskip("scipy.ndimage")
    registry = ramen.default_registry()
    registry.import_module("scipy.ndimage")
    spec = registry.get("scipy.ndimage.gaussian_filter")
    assert str(spec.input("input").type) == str(ramen.library.docparse.ARRAY_LIKE)
    image = np.zeros((5, 5))
    image[2, 2] = 1.0
    g = Graph()
    g.add_node("base.constant", "image", config={"value": image})
    g.add_node(spec, "blur", values={"sigma": 1.0})
    g.connect("image", "out", "blur", "input")
    assert analyze(g, registry).ok
    assert same(run(g, registry)["blur"], ndimage.gaussian_filter(image, 1.0))
