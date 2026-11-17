"""The import engine: any Python function becomes a node."""

import inspect
import math
import types as pytypes
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest

import openramen as ramen
from openramen import Graph, analyze, run
from openramen import types as T
from openramen.library import (
    DEFAULT, ImportFailure, ModuleLibrary, Override, Registry, SpecError, default_registry,
    import_callable, import_module, load_overrides, pin_outputs, report,
)
from openramen.library.docparse import ARRAY_LIKE, ARRAY_OR_SCALAR
from openramen.library.importer import get_signature, pack_overrides

from helpers import const, single


def smooth(image, sigma=1.0, mode="reflect", *, truncate=4.0, out=None):
    """Blur an image with a Gaussian.

    The longer story.

    Parameters
    ----------
    image : ndarray
        The picture.
    sigma : float or sequence of floats, optional
        Width of the blur.
    mode : {'reflect', 'wrap'}, optional
        Edge handling.
    truncate : float
        Cut-off.
    out : ndarray, optional
        Where to write.

    Returns
    -------
    blurred : ndarray
        The result.
    """
    return image * sigma


def pins(spec):
    return {p.name: str(p.type) for p in spec.inputs}


class TestImportCallable:
    def test_everything_comes_from_the_docstring(self):
        s = import_callable(smooth, "t.smooth")
        assert s.summary == "Blur an image with a Gaussian." and s.description == "The longer story."
        assert pins(s) == {
            "image": "Array[_, _]",
            "sigma": "Union[Float, List[Float], Tuple[Float, ...]]",
            "mode": "Str", "truncate": "Float", "out": "Optional[Array[_, _]]"}
        assert [p.source for p in s.inputs] == ["docstring"] * 5
        assert s.input("image").doc == "The picture." and s.input("image").required
        assert s.input("sigma").default == 1.0 and not s.input("sigma").required
        assert [(p.name, str(p.type), p.source) for p in s.outputs] == [
            ("out", "Union[Array[_, _], Bool, Complex, Float, Int]", "docstring")]
        assert s.outputs[0].doc == "The result."
        assert (s.tier, s.pure, s.executor, s.fn) == ("auto", True, "thread", smooth)

    def test_kinds_hidden_and_mutated_pins(self):
        s = import_callable(smooth, "t.smooth")
        assert [p.kind for p in s.inputs] == [
            "keyword", "keyword", "keyword", "keyword_only", "keyword_only"]
        # keyword-only options and plumbing such as ``out`` start hidden
        assert [p.name for p in s.inputs if p.hidden] == ["truncate", "out"]
        assert [p.name for p in s.inputs if p.access == "mutate"] == ["out"]

    def test_annotations_beat_the_docstring(self):
        def f(a: int, b: "Array[uint8, 2]", c: npt.ArrayLike, d: Optional[float] = None,  # noqa: F821
              e=None) -> Tuple[int, float]:
            """Do.

            Parameters
            ----------
            a : str
            e : bool, optional

            Returns
            -------
            count : int
            mean : float
            """
        s = import_callable(f, "t.f")
        assert pins(s) == {"a": "Int", "b": "Array[uint8, 2]", "c": str(ARRAY_LIKE),
                           "d": "Optional[Float]", "e": "Optional[Bool]"}
        assert [p.source for p in s.inputs] == ["annotation"] * 4 + ["docstring"]
        assert [(p.name, str(p.type)) for p in s.outputs] == [("count", "Int"), ("mean", "Float")]

    def test_defaults_type_a_pin_only_when_nothing_else_does(self):
        def f(a, flag=True, n=3, x=1.5, name="a", z=1j, items=(), obj=None, cb=len):
            return a
        s = import_callable(f, "t.f")
        assert pins(s) == {"a": "Any", "flag": "Bool", "n": "Int", "x": "Float", "name": "Str",
                           "z": "Complex", "items": "Any", "obj": "Any", "cb": "Any"}
        assert [p.source for p in s.inputs] == ["", "default", "default", "default", "default",
                                                "default", "", "", ""]
        assert str(s.outputs[0].type) == "Any" and s.summary == ""

    def test_a_default_is_always_acceptable_to_its_own_pin(self):
        def f(a=None, b=0, c="auto", d=None):
            """X.

            Parameters
            ----------
            a : int
            b : float
            c : int
            d : ndarray, optional
            """
        s = import_callable(f, "t.f")
        assert pins(s) == {"a": "Optional[Int]", "b": "Float", "c": "Union[Int, Str]",
                           "d": "Optional[Array[_, _]]"}
        for pin in s.inputs:
            T.accept(pin.default, pin.type)

    def test_unclear_documentation_gives_any_never_a_wrong_type(self):
        def f(a, b, c):
            """X.

            Parameters
            ----------
            a : MyThing or int
            b : see below
            c : int or dtype

            Returns
            -------
            something : a special object
            """
        s = import_callable(f, "t.f")
        assert pins(s) == {"a": "Any", "b": "Any", "c": "Any"}
        assert str(s.outputs[0].type) == "Any"

    def test_variadic_pins_are_typed_by_annotations_only(self):
        def f(*operands: float, **options):
            """X.

            Parameters
            ----------
            operands : list of array_like
            options : dict
            """
        s = import_callable(f, "t.f")
        assert pins(s) == {"operands": "Float", "options": "Any"}
        assert s.var_positional.name == "operands" and s.var_keyword.name == "options"

    @pytest.mark.parametrize("returns, expected", [
        ("out : ndarray", [("out", str(ARRAY_OR_SCALAR))]),
        ("ndarray", [("out", str(ARRAY_OR_SCALAR))]),
        ("y : array_like", [("out", str(ARRAY_OR_SCALAR))]),
        ("n : int", [("out", "Int")]),
        ("flag : bool", [("out", "Bool")]),
        ("r : float or ndarray", [("out", str(ARRAY_OR_SCALAR))]),
        ("r : scalar", [("out", "Union[Bool, Complex, Float, Int]")]),
        ("t : tuple of ndarrays", [("out", f"Tuple[{ARRAY_OR_SCALAR}, ...]")]),
        ("names : list of str", [("out", "List[Str]")]),
        ("None", []),
        ("thing : MyClass", [("out", "Any")]),
        ("a : int\nb : str", [("a", "Int"), ("b", "Str")]),
        ("a : int\nb : MyClass", [("a", "Int"), ("b", "Any")]),
        ("a : int\nb : ndarray, optional", [("out", "Any")]),
        ("a : int\nb : ndarray\n    Only provided if `return_b` is True.", [("out", "Any")]),
        ("a : int\nint", [("out", "Any")]),
        ("a : int\na : str", [("out", "Any")]),
    ])
    def test_outputs_from_the_returns_section(self, returns, expected):
        def f(x):
            pass
        body = "\n".join("        " + line for line in returns.splitlines())
        f.__doc__ = f"X.\n\n        Returns\n        -------\n{body}\n        "
        s = import_callable(f, "t.f")
        assert [(p.name, str(p.type)) for p in s.outputs] == expected

    def test_return_annotation(self):
        def nothing(x) -> None:
            pass

        def number(x) -> float:
            """X.

            Returns
            -------
            a : int
            b : int
            """

        assert import_callable(nothing, "t.n").outputs == ()
        assert [(p.name, str(p.type)) for p in import_callable(number, "t.m").outputs] == [
            ("out", "Float")]

    def test_flags_that_add_results_become_configuration(self):
        def f(a, return_counts=False, full_output=None, retstep=False, return_type="x"):
            """X.

            Parameters
            ----------
            a : int

            Returns
            -------
            values : ndarray
            counts : ndarray, optional
            """
        s = import_callable(f, "t.f")
        assert [p.name for p in s.config] == ["return_counts", "full_output", "retstep"]
        assert [p.name for p in s.inputs] == ["a", "return_type"]     # not a boolean flag
        assert str(s.outputs[0].type) == "Any"
        assert s.infer({}, {"return_counts": False}) == {"out": ARRAY_OR_SCALAR}
        assert s.infer({}, {"return_counts": True}) == {}
        assert s.infer({}, {"retstep": True, "return_counts": False}) == {}

    def test_in_place_and_impure_heuristics(self):
        def shuffle(x, other):
            """Shuffle a list in-place."""

        def save_table(path, rows):
            """Write rows."""

        def clean(x):
            """Return a copy, never in place of the original data.

            This is not done in place.
            """

        s = import_callable(shuffle, "t.shuffle")
        assert s.input("x").access == "mutate" and s.input("other").access == "read"
        assert not s.pure
        assert not import_callable(save_table, "t.save").pure
        c = import_callable(clean, "t.clean")
        assert c.pure is False and c.input("x").access == "mutate"   # the summary says so
        assert import_callable(smooth, "t.s").pure

    def test_things_that_cannot_be_imported(self):
        for obj, reason in [(int, "classes"), (42, "not callable"), (math.log, "no signature")]:
            if obj is math.log:
                try:
                    inspect.signature(math.log)
                    continue          # newer Pythons know its signature
                except ValueError:
                    pass
            with pytest.raises(ImportFailure, match=reason):
                import_callable(obj, "t.x")

    def test_default_id(self):
        assert import_callable(smooth).id == f"{__name__}.smooth"
        assert import_callable(np.sum).id == "numpy.sum"


class NoSignature:
    """A callable like an old C function: only its docstring knows the signature."""

    __doc__ = ("nosig(a, b=2, *, scale=np.float64, mode=some.unknown.thing)\n\n"
               "Add things.\n\nParameters\n----------\na : int\n    First.\nb : int\n")
    __name__ = "nosig"
    __module__ = __name__

    @property
    def __signature__(self):
        raise ValueError("no signature found")

    def __call__(self, a, b=2, *, scale=float, mode="m"):
        return (a + b, scale, mode)


class TestSignatures:
    def test_inspect_first(self):
        sig, source = get_signature(smooth)
        assert source == "inspect" and list(sig.parameters)[:2] == ["image", "sigma"]

    def test_docstring_signature_line(self):
        obj = NoSignature()
        with pytest.raises(ValueError):
            inspect.signature(obj)
        sig, source = get_signature(obj)
        assert source == "docstring" and list(sig.parameters) == ["a", "b", "scale", "mode"]
        s = import_callable(obj, "t.nosig")
        assert pins(s) == {"a": "Int", "b": "Int", "scale": "Any", "mode": "Any"}
        assert s.input("scale").default is np.float64      # resolved, not executed
        assert s.input("mode").default is DEFAULT          # unknown: never passed
        assert not s.input("mode").required and s.summary == "Add things."
        assert s.call({"a": 1}) == {"out": (3, float, "m")}
        g = Graph()
        g.add_node(s, "n", values={"a": 5, "b": 1})
        registry = default_registry()
        registry.register(s)
        assert analyze(g, registry).ok and run(g, registry)["n"] == (6, float, "m")

    def test_ufunc_structure_is_the_fallback(self, monkeypatch):
        def no_signature(obj, **kwargs):
            raise ValueError("no signature")
        monkeypatch.setattr(inspect, "signature", no_signature)
        for ufunc, names in [(np.sqrt, ["x"]), (np.add, ["x1", "x2"])]:
            sig, source = get_signature(ufunc, ramen.library.docparse.Doc())
            assert source == "ufunc" and list(sig.parameters) == names
        s = import_callable(np.frompyfunc(lambda a, b, c: a, 3, 2), "t.custom")
        assert [(p.name, p.kind) for p in s.inputs] == [
            ("x1", "positional"), ("x2", "positional"), ("x3", "positional")]
        assert [p.name for p in s.outputs] == ["out1", "out2"]

    def test_nothing_to_go_on(self):
        class Opaque:
            __signature__ = property(lambda self: (_ for _ in ()).throw(ValueError()))

            def __call__(self):
                pass
        with pytest.raises(ImportFailure, match="no signature"):
            import_callable(Opaque(), "t.o")


class TestUfuncs:
    def types(self, name, **inputs):
        registry = default_registry()
        registry.import_module("numpy", pack=False)
        graph = single(f"numpy.{name}", registry, **inputs)
        analysis = analyze(graph, registry)
        assert analysis.ok, analysis.problems
        result = run(graph, registry)           # auto tier: outputs are checked
        spec = registry.get(f"numpy.{name}")
        return [str(analysis.out_type("n", p.name)) for p in spec.outputs], result["n"]

    def test_exact_types_by_sampling(self):
        u8 = np.array([1, 2], dtype="uint8")
        assert self.types("add", x1=u8, x2=1)[0] == ["Array[uint8, 1]"]
        assert self.types("add", x1=u8, x2=1.5)[0] == ["Array[float64, 1]"]
        assert self.types("add", x1=2, x2=3) == (["Int"], 5)
        assert self.types("sqrt", x=4.0) == (["Float"], 2.0)
        assert self.types("less", x1=u8, x2=2)[0] == ["Array[bool, 1]"]
        assert self.types("divmod", x1=7, x2=2) == (["Int", "Int"], {"out1": 3, "out2": 1})
        assert self.types("divmod", x1=u8, x2=2)[0] == ["Array[uint8, 1]", "Array[uint8, 1]"]

    def test_zero_dimensional_arrays_give_scalars_as_numpy_does(self):
        types, value = self.types("add", x1=np.array(2.0), x2=1)
        assert types == ["Float"] and value == 3.0 and type(value) is float

    def test_unknown_inputs_fall_back_to_the_declared_type(self):
        types, value = self.types("add", x1=[1, 2], x2=[3, 4])
        assert types == [str(ARRAY_OR_SCALAR)] and value.tolist() == [4, 6]

    def test_array_of_unknown_rank_may_come_back_as_a_scalar(self):
        registry = default_registry()
        registry.import_module("numpy")
        g = Graph()
        const(g, [], "shape")
        g.add_node("base.array_zeros", "zeros")       # rank unknown statically
        g.add_node("numpy.add", "n", values={"x2": 1})
        g.connect("shape", "out", "zeros", "shape")
        g.connect("zeros", "out", "n", "x1")
        analysis = analyze(g, registry)
        assert str(analysis.out_type("n", "out")) == "Union[Array[float64, _], Float]"
        assert run(g, registry)["n"] == 1.0           # a 0-d array: NumPy returns a scalar

    def test_optional_positional_buffers_do_not_count_as_operands(self):
        """Older NumPy documents ``divmod(x1, x2[, out1, out2], /, ...)``."""
        for text in ("divmod(x1, x2, out1=None, out2=None, /, *, where=True, dtype=None)",
                     "divmod(x1, x2[, out1, out2], / [, out=(None, None)], *, where=True)"):
            spec = import_callable(np.divmod, "t.divmod",
                                   override=Override.from_mapping({"signature": text}))
            operands = [p.name for p in spec.inputs if p.kind == "positional" and p.required]
            assert operands == ["x1", "x2"], text
            assert [p.name for p in spec.outputs] == ["out1", "out2"]
            g = Graph()
            g.add_node(spec, "n", values={"x1": 7, "x2": 2})
            registry = default_registry()
            registry.register(spec)
            analysis = analyze(g, registry)
            assert [str(t) for t in analysis.nodes["n"].out_types.values()] == ["Int", "Int"], text
            assert run(g, registry)["n"] == {"out1": 3, "out2": 1}
            # a wired output buffer makes sampling meaningless: fall back
            assert spec.infer({"x1": T.INT, "x2": T.INT, "out1": T.ArrayT()}, {}) == {}

    def test_options_that_change_the_result_type_disable_sampling(self):
        registry = default_registry()
        registry.import_module("numpy")
        g = single("numpy.add", registry, x1=np.array([1, 2], dtype="uint8"), x2=1,
                   literals={"dtype": "float32"})
        assert str(analyze(g, registry).out_type("n", "out")) == str(ARRAY_OR_SCALAR)
        assert run(g, registry)["n"].dtype == np.float32


def make_module():
    module = pytypes.ModuleType("fakelib")

    def double(x: float) -> float:
        """Twice x."""
        return 2 * x

    def _private(x):
        return x

    def broken(x):
        return x

    def pair(x):
        """Give two things.

        Returns
        -------
        low : int
        high : int
        """
        return x, x + 1

    class Thing:
        pass

    for fn in (double, _private, broken, pair):
        fn.__module__ = "fakelib"
    Thing.__module__ = "fakelib"
    module.double, module._private, module.broken, module.pair = double, _private, broken, pair
    module.Thing = Thing
    module.sqrt = math.sqrt            # merely imported from elsewhere
    module.CONSTANT = 3
    module.sub = pytypes.ModuleType("fakelib.sub")
    return module


class TestModuleLibrary:
    def test_names_are_the_modules_own_public_functions(self):
        library = ModuleLibrary(make_module())
        assert library.names() == ["broken", "double", "pair"]
        assert library.ids() == ["fakelib.broken", "fakelib.double", "fakelib.pair"]
        assert library.prefix == "fakelib"

    def test_all_is_trusted_when_present(self):
        module = make_module()
        module.__all__ = ["double", "sqrt", "Thing", "CONSTANT", "missing", "not valid", 3]
        assert ModuleLibrary(module).names() == ["double", "sqrt"]

    def test_load(self):
        library = ModuleLibrary(make_module())
        spec = library.load("fakelib.double")
        assert (spec.id, spec.target, spec.category) == ("fakelib.double", "fakelib:double", "fakelib")
        assert spec.call({"x": 2.0}) == {"out": 4.0}
        assert library.load("fakelib.nope") is None and library.load("fakelib._private") is None
        assert library.load("fakelib.Thing") is None and library.failures == {}

    def test_failures_are_recorded_not_raised(self):
        overrides = {"broken": Override(config=("nope",)), "double": Override(skip=True)}
        library = ModuleLibrary(make_module(), overrides=overrides)
        assert library.load("fakelib.broken") is None and library.load("fakelib.double") is None
        assert "no parameter named 'nope'" in library.failures["broken"]
        assert library.failures["double"] == "skipped by an override"

    def test_bad_modules(self):
        with pytest.raises(ImportFailure, match="cannot import module"):
            ModuleLibrary("no_such_module_anywhere")
        with pytest.raises(ImportFailure, match="invalid prefix"):
            ModuleLibrary(make_module(), prefix="not valid")

    def test_custom_prefix(self):
        library = ModuleLibrary(make_module(), prefix="mine.lib")
        assert library.load("mine.lib.double").id == "mine.lib.double"


class TestRegistryIntegration:
    def test_specs_are_built_on_demand(self):
        registry = Registry()
        library = import_module(registry, make_module())
        assert registry._specs == {}                          # nothing built yet
        assert registry.ids() == ["fakelib.broken", "fakelib.double", "fakelib.pair"]
        assert "fakelib.double" in registry and "fakelib.nope" not in registry
        assert list(registry._specs) == ["fakelib.double"]
        assert registry.get("fakelib.double") is registry.get("fakelib.double")
        assert len(registry) == 3 and len(list(registry)) == 3
        assert list(registry.categories()) == ["fakelib"]
        assert registry.providers == [library]
        with pytest.raises(ramen.UnknownSpecError):
            registry.get("fakelib.nope")

    def test_copies_share_providers_but_not_registrations(self):
        registry = default_registry()
        import_module(registry, make_module())
        clone = registry.copy()
        assert "fakelib.pair" in clone
        clone.register(import_callable(smooth, "extra.smooth"))
        assert "extra.smooth" not in registry
        other = Registry()
        other.update(registry)
        assert "fakelib.double" in other and "base.add" in other

    def test_the_longest_prefix_answers_first(self):
        registry = Registry()
        registry.import_module("numpy")
        registry.import_module("numpy.linalg")
        assert registry.get("numpy.linalg.norm").target == "numpy.linalg:norm"
        assert registry.get("numpy.sum").target == "numpy:sum"
        assert registry.find("numpy.linalg.nope") is None and registry.find("numpyx.sum") is None
        again = registry.import_module("numpy")                # replaces, never duplicates
        assert [p.prefix for p in registry.providers] == ["numpy.linalg", "numpy"]
        assert registry.providers[1] is again

    def test_imported_nodes_run_in_graphs(self):
        registry = default_registry()
        registry.import_module(make_module())
        g = Graph()
        const(g, 4, "c")
        g.add_node("fakelib.double", "d")
        g.add_node("fakelib.pair", "p")
        g.connect("c", "out", "d", "x")            # Int into Float: coerced
        g.connect("d", "out", "p", "x")
        analysis = analyze(g, registry)
        assert analysis.ok and analysis.wire_verdict("d", "x") is T.Verdict.COERCION
        with pytest.raises(ramen.NodeError, match="declared Int .*docstring.* returned Float"):
            run(g, registry)                       # the docstring promised ints
        assert run(g, registry, check_outputs=False)["p"] == {"low": 8.0, "high": 9.0}


class TestOverrides:
    def test_every_key(self):
        override = Override.from_mapping({
            "types": {"image": "Array[float64, 2]", "out": "Array[float64, 2]"},
            "config": ["mode"], "hidden": ["sigma"], "shown": ["truncate"],
            "mutates": ["image"], "aliases": {"out": "image"},
            "category": "Filters", "pure": False, "thread_safe": False,
            "executor": "process", "summary": "Blur."})
        s = import_callable(smooth, "t.smooth", override=override)
        assert pins(s)["image"] == "Array[float64, 2]" and s.input("image").source == "override"
        assert [p.name for p in s.config] == ["mode"] and "mode" not in pins(s)
        assert s.input("sigma").hidden and not s.input("truncate").hidden
        assert s.input("image").access == "mutate"
        assert (s.outputs[0].aliases, str(s.outputs[0].type)) == ("image", "Array[float64, 2]")
        assert (s.category, s.pure, s.thread_safe, s.executor, s.summary, s.tier) == (
            "Filters", False, False, "process", "Blur.", "annotated")

    def test_outputs(self):
        def f(x):
            return x, x
        s = import_callable(f, "t.f", override=Override.from_mapping(
            {"outputs": ["first: Int", "second"], "aliases": {"second": "x"}}))
        assert [(p.name, str(p.type), p.aliases, p.source) for p in s.outputs] == [
            ("first", "Int", None, "override"), ("second", "Any", "x", "override")]
        none = import_callable(f, "t.g", override=Override.from_mapping({"outputs": []}))
        assert none.outputs == ()

    def test_variants(self):
        def stats(values, with_count=False):
            """Mean of values, and how many there were if asked."""
            mean = sum(values) / len(values)
            return (mean, len(values)) if with_count else mean
        module = pytypes.ModuleType("statlib")
        stats.__module__ = "statlib"
        module.stats = stats
        overrides = load_overrides({"stats_count": {
            "function": "stats", "presets": {"with_count": True},
            "outputs": ["mean: Float", "count: Int"]}})
        registry = default_registry()
        library = import_module(registry, module, overrides=overrides)
        assert library.names() == ["stats", "stats_count"]
        variant = registry.get("statlib.stats_count")
        assert [p.name for p in variant.inputs] == ["values"] and variant.config == ()
        assert variant.call({"values": [1, 2, 3]}) == {"mean": 2.0, "count": 3}
        plain = registry.get("statlib.stats")
        assert [p.name for p in plain.config] == ["with_count"]      # the flag is configuration
        assert plain.call({"values": [1, 3]}) == {"out": 2.0}
        assert plain.call({"values": [1, 3]}, {"with_count": True}) == {"out": (2.0, 2)}

    @pytest.mark.parametrize("data, message", [
        ({"colour": "red"}, "unknown key 'colour'"),
        ({"types": {"a": "Nonsense"}}, "unknown type"),
        ({"types": ["a"]}, "must map names to text"),
        ({"outputs": ["a: Wrong[", "b"]}, "unknown type"),
        ({"outputs": "a"}, "must be a list of names"),
        ({"config": [1]}, "must be a list of names"),
        ({"skip": "yes"}, "must be true or false"),
        ({"pure": 1}, "must be true or false"),
        ({"category": 3}, "must be text"),
        ({"function": 3}, "must be a name"),
        ({"presets": [1]}, "must be a table"),
    ])
    def test_malformed_overrides(self, data, message):
        with pytest.raises(SpecError, match=message):
            Override.from_mapping(data, "pack")
        with pytest.raises(SpecError):
            Override.from_mapping("not a table")  # type: ignore[arg-type]

    @pytest.mark.parametrize("data, message", [
        ({"config": ["nope"]}, "override config: no parameter named 'nope'"),
        ({"hidden": ["nope"]}, "override hidden"),
        ({"mutates": ["nope"]}, "override mutates"),
        ({"presets": {"nope": 1}}, "override presets"),
        ({"aliases": {"out": "nope"}}, "no input named 'nope'"),
        ({"executor": "gpu"}, "executor"),
    ])
    def test_overrides_that_do_not_match_the_function(self, data, message):
        with pytest.raises(ImportFailure, match=message):
            import_callable(smooth, "t.s", override=Override.from_mapping(data))

    def test_signature_override(self):
        obj = NoSignature()
        override = Override.from_mapping({"signature": "nosig(first, second=5, *, scale=float)"})
        s = import_callable(obj, "t.n", override=override)
        assert [(p.name, p.kind) for p in s.inputs] == [
            ("first", "keyword"), ("second", "keyword"), ("scale", "keyword_only")]
        assert s.input("second").default == 5 and s.tier == "annotated"
        assert s.call({"first": 1}) == {"out": (3, float, "m")}      # passed by position
        for bad in ("not a signature", "f(a,, b)", 3):
            with pytest.raises(SpecError, match="signature"):
                Override.from_mapping({"signature": bad})

    def test_pack_gives_stable_pins_where_numpy_versions_differ(self):
        registry = Registry()
        registry.import_module("numpy")
        registry.import_module("numpy.random")
        assert [p.name for p in registry.get("numpy.arange").inputs][:3] == [
            "start_or_stop", "stop", "step"]
        assert registry.get("numpy.arange").call({"start_or_stop": 3})["out"].tolist() == [0, 1, 2]
        assert registry.get("numpy.arange").call(
            {"start_or_stop": 1, "stop": 7, "step": 3})["out"].tolist() == [1, 4]
        assert [p.name for p in registry.get("numpy.where").inputs] == ["condition", "x", "y"]
        assert registry.get("numpy.random.rand").var_positional.name == "args"
        assert registry.get("numpy.random.rand").call({"args_0": 2, "args_1": 3})["out"].shape == (2, 3)

    def test_toml_files(self, tmp_path):
        path = tmp_path / "mylib.toml"
        path.write_text('[smooth]\ntypes = { image = "Array[uint8, 2]" }\nhidden = ["mode"]\n\n'
                        '["filters.blur"]\nskip = true\n')
        overrides = load_overrides(path)
        assert set(overrides) == {"smooth", "filters.blur"} and overrides["filters.blur"].skip
        assert overrides["smooth"].types == {"image": "Array[uint8, 2]"}
        path.write_text("[smooth\n")
        with pytest.raises(SpecError, match="not valid TOML"):
            load_overrides(path)
        path.write_text("[smooth]\ncolour = 1\n")
        with pytest.raises(SpecError, match=r"mylib.toml: \[smooth\]: unknown key"):
            load_overrides(path)

    def test_user_overrides_go_on_top_of_the_pack(self, tmp_path):
        assert "histogramdd" in pack_overrides("numpy") and pack_overrides("no_such_lib") == {}
        path = tmp_path / "mine.toml"
        path.write_text('[sum]\ncategory = "Mine"\n[histogramdd]\ncategory = "Mine too"\n')
        registry = Registry()
        registry.import_module("numpy", overrides=path)
        assert registry.get("numpy.sum").category == "Mine"
        assert registry.get("numpy.histogramdd").category == "Mine too"
        assert registry.get("numpy.linspace_step").tier == "annotated"       # from the pack
        bare = Registry()
        bare.import_module("numpy", pack=False)
        assert bare.find("numpy.linspace_step") is None and bare.get("numpy.sum").tier == "auto"

    def test_a_variant_never_replaces_a_real_function(self):
        module = make_module()
        overrides = load_overrides({"double": {"function": "pair", "presets": {}}})
        library = ModuleLibrary(module, overrides=overrides)
        assert library.load("fakelib.double").target == "fakelib:double"
        assert "already exists" in library.failures["double (variant)"]
        missing = ModuleLibrary(module, overrides=load_overrides(
            {"ghost": {"function": "nope"}}))
        assert missing.load("fakelib.ghost") is None
        assert "variant of unknown function 'nope'" in missing.failures["ghost"]


class TestObservation:
    def test_pin_what_a_run_showed(self):
        def measure(image):
            return image.mean(), image.shape
        module = pytypes.ModuleType("lab")
        measure.__module__ = "lab"
        module.measure = measure
        registry = default_registry()
        registry.import_module(module)
        g = single("lab.measure", registry, image=np.ones((2, 3)))
        assert str(analyze(g, registry).out_type("n", "out")) == "Any"
        result = run(g, registry)
        observed = result.observed("n")
        assert {k: str(v) for k, v in observed.items()} == {"out": "Tuple[Float, Tuple[Int, Int]]"}

        pinned = pin_outputs(registry, "lab.measure", observed)
        assert (pinned.tier, pinned.outputs[0].source) == ("annotated", "observed")
        assert str(analyze(g, registry).out_type("n", "out")) == "Tuple[Float, Tuple[Int, Int]]"
        assert run(g, registry)["n"] == (1.0, (2, 3))
        # the pinned type is now checked on every run
        other = single("lab.measure", registry, image=np.ones((2, 3, 4)))
        with pytest.raises(ramen.NodeError, match="output is declared"):
            run(other, registry)

    def test_pinning_accepts_text_and_rejects_unknown_pins(self):
        registry = default_registry()
        registry.import_module("numpy", pack=False)
        pinned = pin_outputs(registry, "numpy.add", {"out": "Array[float64, 2]"})
        assert str(pinned.outputs[0].type) == "Array[float64, 2]" and pinned.infer is None
        with pytest.raises(SpecError, match="no output 'nope'"):
            pin_outputs(registry, "numpy.add", {"nope": "Int"})
        with pytest.raises(ramen.UnknownSpecError):
            pin_outputs(registry, "numpy.nope", {})
        curated = pin_outputs(registry, "base.identity", {"out": "Int"})
        assert curated.tier == "curated"


class TestReport:
    def test_counts(self):
        overrides = {"broken": Override(config=("nope",)), "pair": Override(skip=True)}
        summary = report(ModuleLibrary(make_module(), overrides=overrides))
        assert (summary.candidates, summary.imported, summary.skipped) == (3, 1, 1)
        assert list(summary.failures) == ["broken"]
        assert (summary.inputs, summary.typed_inputs, summary.outputs, summary.typed_outputs) == (
            1, 1, 1, 1)
        assert summary.fully_typed == 1 and summary.documented == 1
        assert summary.sources == {"annotation": 2}
        text = "\n".join(summary.lines())
        assert "imported as nodes    1 (50%, 1 skipped on purpose)" in text
        assert "inputs with a type   1 of 1 (100%)" in text

    def test_empty_module(self):
        summary = report(ModuleLibrary(pytypes.ModuleType("empty")))
        assert summary.candidates == 0 and "n/a" in "\n".join(summary.lines())

    def test_command_line(self, capsys):
        from openramen.__main__ import main
        assert main(["import", "numpy.fft"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("numpy.fft") and "imported as nodes" in out
        assert main(["import", "numpy.fft", "--list"]) == 0
        assert "numpy.fft.fft " in capsys.readouterr().out
        assert main(["import", "numpy", "--no-pack"]) == 0
        assert "skipped" not in capsys.readouterr().out
        assert main(["import", "no_such_module_anywhere"]) == 2
        assert "cannot import module" in capsys.readouterr().err


class TestGraphImports:
    def test_saved_with_the_graph(self):
        g = Graph("with imports")
        g.imports = ["numpy.linalg", "numpy", "numpy"]
        g.add_node("numpy.linalg.det", "det", values={"a": [[1.0, 2.0], [3.0, 4.0]]})
        text = g.dumps()
        assert '"imports": ["numpy", "numpy.linalg"]' in text
        loaded = Graph.loads(text)
        assert loaded.imports == ["numpy", "numpy.linalg"] and loaded == g
        assert g.copy().imports == g.imports
        assert run(loaded)["det"] == pytest.approx(-2.0)
        assert '"imports"' not in Graph("plain").dumps()

    def test_imports_matter_for_equality(self):
        a, b = Graph(), Graph()
        b.imports = ["numpy"]
        assert a != b

    @pytest.mark.parametrize("imports", [["not valid"], ["a..b"], [3], "numpy", [""]])
    def test_invalid_module_names(self, imports):
        g = Graph()
        g.imports = imports if isinstance(imports, list) else [imports + " x"]
        with pytest.raises(ramen.GraphError):
            g.dumps()
        with pytest.raises(ramen.GraphError, match="imports"):
            Graph.from_dict({"format": "openramen.graph", "version": 1, "imports": imports})

    def test_loading_does_not_import_anything(self):
        import sys
        text = ('{"format": "openramen.graph", "version": 1, '
                '"imports": ["no_such_module_anywhere"]}')
        g = Graph.loads(text)                                  # fine: nothing is imported
        assert "no_such_module_anywhere" not in sys.modules
        analysis = analyze(g)
        assert [p.code for p in analysis.problems] == ["bad-import"]
        assert "no_such_module_anywhere" in analysis.problems[0].message
        with pytest.raises(ramen.GraphCheckError):
            run(g)

    def test_without_the_import_the_nodes_are_unknown(self):
        g = Graph()
        g.add_node("numpy.sum", "n", values={"a": [1, 2]})
        assert [p.code for p in analyze(g).problems] == ["unknown-spec"]
        g.imports = ["numpy"]
        assert run(g)["n"] == 3
        assert ramen.registry_for(g) is ramen.registry_for(g)   # built once per import set


class TestPassingArguments:
    def test_by_position_while_there_is_no_gap(self):
        """Some functions report an ordinary parameter that is really positional-only."""
        def really_positional(x, /, scale=1):
            return x * scale
        import inspect as _inspect
        lying = _inspect.Signature([
            _inspect.Parameter("x", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter("scale", _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1)])
        really_positional.__signature__ = lying
        s = import_callable(really_positional, "t.rp")
        assert [p.kind for p in s.inputs] == ["keyword", "keyword"]
        assert s.call({"x": 3}) == {"out": 3} and s.call({"x": 3, "scale": 2}) == {"out": 6}

    def test_pins_after_a_removed_parameter_go_by_keyword(self):
        def f(a, flag=False, b=2, c=3):
            return (a, flag, b, c)
        for override in (Override(config=("flag",)), Override(presets={"flag": True})):
            s = import_callable(f, "t.f", override=override)
            assert [(p.name, p.kind) for p in s.inputs] == [
                ("a", "keyword"), ("b", "keyword_only"), ("c", "keyword_only")]
            want_flag = bool(override.presets)
            assert s.call({"a": 1, "b": 20, "c": 30}) == {"out": (1, want_flag, 20, 30)}
            assert not s.input("b").hidden          # still an ordinary pin on the node
        configured = import_callable(f, "t.g", override=Override(config=("flag",)))
        assert configured.call({"a": 1, "c": 9}, {"flag": True}) == {"out": (1, True, 2, 9)}

    def test_the_same_holds_for_hand_written_specs(self):
        def f(a, mode="x", b=2):
            return (a, mode, b)
        s = ramen.spec_from_function(f, "t.f", config=("mode",))
        assert [(p.name, p.kind) for p in s.inputs] == [("a", "keyword"), ("b", "keyword_only")]
        assert s.call({"a": 1, "b": 5}, {"mode": "y"}) == {"out": (1, "y", 5)}

    def test_presets_must_be_passable_by_keyword(self):
        def f(a, /, b=1, *rest):
            return a
        for name in ("a", "b"):
            with pytest.raises(ImportFailure, match="passed by keyword"):
                import_callable(f, "t.f", override=Override(presets={name: 1}))
