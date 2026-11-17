"""The import engine: ordinary Python functions become nodes.

``import_callable`` builds a NodeSpec for one function from, in order of
trust: an override, its annotations, its docstring, and its default values.
``ModuleLibrary`` does that lazily for a whole module, and ``import_module``
attaches one to a registry.

Nothing here calls the functions it imports.  Importing a *module* does run
that module's own top-level code, exactly as ``import`` does.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import re
import sys
import types as pytypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from .. import types as T
from .docparse import (
    ARRAY_LIKE, ARRAY_OR_SCALAR, SCALAR, Doc, parse_docstring, signature_from_text,
    type_from_phrase,
)
from .registry import Registry
from .sampling import result_type
from .spec import (
    DEFAULT, MISSING, PARAMETER_KINDS, NodeSpec, PinSpec, SpecError, _annotations,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib

__all__ = [
    "ImportFailure", "Override", "ModuleLibrary", "ImportReport",
    "import_callable", "import_module", "load_overrides", "pack_overrides",
    "pin_outputs", "report",
]


class ImportFailure(Exception):
    """A callable cannot be turned into a node."""


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------

_OVERRIDE_KEYS = {
    "types", "outputs", "config", "hidden", "shown", "mutates", "aliases", "skip",
    "category", "pure", "thread_safe", "executor", "summary", "function", "presets",
    "signature",
}
_FIELD_KEYS = ("category", "pure", "thread_safe", "executor", "summary")


@dataclass
class Override:
    """Corrections to what introspection found for one function."""

    types: dict[str, str] = field(default_factory=dict)       # pin name -> type text
    outputs: Optional[list[tuple[str, Optional[str]]]] = None  # (name, type text)
    config: tuple[str, ...] = ()
    hidden: tuple[str, ...] = ()
    shown: tuple[str, ...] = ()
    mutates: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)     # output -> input
    skip: bool = False
    fields: dict[str, Any] = field(default_factory=dict)      # category, pure, ...
    #: A variant: this entry is a node of its own that calls ``function`` with
    #: ``presets`` fixed.  ``unique_counts`` can be ``unique`` with
    #: ``return_counts = true`` and therefore two outputs.
    function: str = ""
    presets: dict[str, Any] = field(default_factory=dict)
    #: The signature to use, as text: ``"arange(start, stop=None, step=1)"``.
    #: For functions whose own signature cannot be read or cannot be wired.
    signature: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str = "override") -> "Override":
        if not isinstance(data, Mapping):
            raise SpecError(f"{where}: expected a table")
        unknown = set(data) - _OVERRIDE_KEYS
        if unknown:
            raise SpecError(f"{where}: unknown key {sorted(unknown)[0]!r}")

        def names(key: str) -> tuple[str, ...]:
            value = data.get(key, ())
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise SpecError(f"{where}: {key!r} must be a list of names")
            return tuple(value)

        def table(key: str) -> dict[str, str]:
            value = data.get(key, {})
            if not isinstance(value, Mapping) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                raise SpecError(f"{where}: {key!r} must map names to text")
            return dict(value)

        outputs: Optional[list[tuple[str, Optional[str]]]] = None
        if "outputs" in data:
            outputs = []
            for entry in names("outputs"):
                name, sep, type_text = entry.partition(":")
                outputs.append((name.strip(), type_text.strip() if sep else None))
        types = table("types")
        for text in list(types.values()) + [t for _, t in outputs or [] if t]:
            try:
                T.parse_type(text)
            except T.TypeSyntaxError as exc:
                raise SpecError(f"{where}: {exc}") from exc
        if not isinstance(data.get("skip", False), bool):
            raise SpecError(f"{where}: 'skip' must be true or false")
        fields = {k: data[k] for k in _FIELD_KEYS if k in data}
        for key in ("pure", "thread_safe"):
            if key in fields and not isinstance(fields[key], bool):
                raise SpecError(f"{where}: {key!r} must be true or false")
        for key in ("category", "executor", "summary"):
            if key in fields and not isinstance(fields[key], str):
                raise SpecError(f"{where}: {key!r} must be text")
        function = data.get("function", "")
        presets = data.get("presets", {})
        if not isinstance(function, str):
            raise SpecError(f"{where}: 'function' must be a name")
        if not isinstance(presets, Mapping) or not all(isinstance(k, str) for k in presets):
            raise SpecError(f"{where}: 'presets' must be a table of argument values")
        signature = data.get("signature", "")
        if not isinstance(signature, str) or (
                signature and signature_from_text(signature) is None):
            raise SpecError(f"{where}: 'signature' must look like \"name(a, b=1)\"")
        return cls(types, outputs, names("config"), names("hidden"), names("shown"),
                   names("mutates"), table("aliases"), bool(data.get("skip", False)), fields,
                   function, dict(presets), signature)


def load_overrides(source: "str | Path | Mapping[str, Any]") -> dict[str, Override]:
    """Read an override file (TOML): one table per function.

    Keys are function names relative to the library, e.g. ``[histogram]`` or
    ``["linalg.norm"]`` in a file for NumPy.
    """
    if isinstance(source, Mapping):
        data: Mapping[str, Any] = source
        where = "overrides"
    else:
        path = Path(source)
        where = path.name
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise SpecError(f"{where}: not valid TOML: {exc}") from exc
    return {str(name): Override.from_mapping(table, f"{where}: [{name}]")
            for name, table in data.items()}


def pack_overrides(library: str) -> dict[str, Override]:
    """The overrides bundled with OpenRamen for a library, if any."""
    path = Path(__file__).resolve().parent.parent / "packs" / f"{library}.toml"
    return load_overrides(path) if path.is_file() else {}


# --------------------------------------------------------------------------
# One callable
# --------------------------------------------------------------------------

#: Optional parameters that are plumbing more often than data: hidden by default.
_PLUMBING = {"out", "where", "casting", "order", "subok", "like", "device", "signature",
             "extobj", "copy", "overwrite_input", "overwrite_a", "overwrite_b",
             "check_finite", "workers"}
_IN_PLACE = re.compile(r"\bin[- ]place\b", re.I)
_IMPURE_NAME = re.compile(
    r"^(?:save|load|read|write|print|open|input|show|seed|shuffle|set_|get_|fromfile|"
    r"tofile|genfromtxt|memmap|imread|imsave|imshow|plot|rand|random|permutation|choice)",
    re.I)
_TYPED_DEFAULTS = (bool, int, float, complex, str)
#: Parameters that switch extra return values on: the number of outputs then
#: depends on a value, so the result is kept as one untyped output.
_OUTPUT_FLAG = re.compile(r"^(?:return_\w+|ret[a-z]+|full_output|full|with_\w+)$")
_CONDITIONAL = re.compile(
    r"\bonly\s+(?:provided|returned|present|given|available|if)\b|"
    r"\bif\s+`{0,2}(?:return_\w+|ret[a-z]+|full_output|full)`{0,2}\s+is\b|\boptional\b", re.I)


def _namespace(obj: Any) -> dict[str, Any]:
    module = sys.modules.get(getattr(obj, "__module__", None) or "")
    namespace = dict(vars(module)) if module is not None else {}
    namespace.setdefault("np", np)
    namespace.setdefault("numpy", np)
    return namespace


def get_signature(obj: Any, doc: Optional[Doc] = None) -> tuple[inspect.Signature, str]:
    """The signature of a callable and where it was found.

    Tries ``inspect``, then a signature line at the top of the docstring, then
    the structure of a NumPy ufunc.
    """
    try:
        return inspect.signature(obj), "inspect"
    except (TypeError, ValueError):
        pass
    if doc is None:
        doc = parse_docstring(getattr(obj, "__doc__", None))
    if doc.signature:
        signature = signature_from_text(doc.signature, _namespace(obj))
        if signature is not None:
            return signature, "docstring"
    if isinstance(obj, np.ufunc):
        names = ["x"] if obj.nin == 1 else [f"x{i + 1}" for i in range(obj.nin)]
        params = [inspect.Parameter(n, inspect.Parameter.POSITIONAL_ONLY) for n in names]
        return inspect.Signature(params), "ufunc"
    raise ImportFailure("no signature could be found")


def _annotation_type(annotation: Any) -> T.Type:
    try:
        import numpy.typing as npt
        if annotation is npt.ArrayLike:
            return ARRAY_LIKE
        if annotation is npt.DTypeLike:
            return T.ANY
    except Exception:  # pragma: no cover
        pass
    return T.from_annotation(annotation)


def _widen_output(t: T.Type) -> T.Type:
    """NumPy returns a scalar where the documentation says "ndarray" for 0-d
    results, so a documented array may always be a scalar too."""
    if isinstance(t, T.ArrayT) or t == ARRAY_LIKE:
        return ARRAY_OR_SCALAR
    if isinstance(t, T.ListT):
        return T.ListT(_widen_output(t.elem))
    if isinstance(t, T.TupleT):
        return T.TupleT(tuple(_widen_output(e) for e in t.elems), t.variadic)
    if isinstance(t, T.UnionT) and any(isinstance(o, T.ArrayT) for o in t.options):
        return T.union(*(o for o in t.options if not isinstance(o, T.ArrayT)),
                       ARRAY_OR_SCALAR)
    if isinstance(t, T.UnionT) and {T.INT, T.FLOAT, T.COMPLEX} <= set(t.options):
        return T.union(t, T.BOOL)   # "scalar" results include booleans
    return t


def _is_ufunc_like(obj: Any) -> bool:
    return isinstance(obj, np.ufunc)


def _ufunc_infer(obj: Any, value_pins: tuple[str, ...], out_pins: tuple[str, ...],
                 untouched: Mapping[str, T.Type]) -> Callable[..., Mapping[str, T.Type]]:
    """Exact output types of a ufunc, by sampling it.

    ``untouched`` gives, for the options that change the result type (``out``,
    ``dtype``, ``signature``), the type they have when left alone.
    """
    def infer(inputs: Mapping[str, Optional[T.Type]], config: Mapping[str, Any]
              ) -> Mapping[str, T.Type]:
        for extra, default_type in untouched.items():
            if inputs.get(extra) not in (None, default_type):
                return {}                     # set by the user: sampling would be wrong
        found = result_type(obj, *(inputs.get(p) for p in value_pins), raw=True)
        if found is None:
            return {}
        if len(out_pins) == 1:
            return {out_pins[0]: found}
        if isinstance(found, T.TupleT) and not found.variadic and len(found.elems) == len(out_pins):
            return dict(zip(out_pins, found.elems))
        return {}
    return infer


def import_callable(obj: Any, id: Optional[str] = None, *,
                    override: Optional[Override] = None,
                    target: str = "", category: str = "") -> NodeSpec:
    """Build a NodeSpec for any Python callable.  Raises :class:`ImportFailure`."""
    if isinstance(obj, type):
        raise ImportFailure("classes are not imported yet")
    if not callable(obj):
        raise ImportFailure("not callable")
    override = override or Override()
    if override.skip:
        raise ImportFailure("skipped by an override")
    doc = parse_docstring(getattr(obj, "__doc__", None))
    if override.signature:
        given = signature_from_text(override.signature, _namespace(obj))
        if given is None:
            raise ImportFailure(f"override signature {override.signature!r} cannot be read")
        signature = given
    else:
        signature, _ = get_signature(obj, doc)
    try:
        hints = _annotations(obj)
    except Exception:
        hints = {}
    parameters = signature.parameters

    for key, names in (("config", override.config), ("hidden", override.hidden),
                       ("shown", override.shown), ("mutates", override.mutates),
                       ("presets", tuple(override.presets))):
        for name in names:
            if name not in parameters:
                raise ImportFailure(f"override {key}: no parameter named {name!r}")

    has_var_positional = any(p.kind is p.VAR_POSITIONAL for p in parameters.values())
    in_place = bool(_IN_PLACE.search(doc.summary))

    def by_keyword(param: inspect.Parameter) -> bool:
        return param.kind is param.KEYWORD_ONLY or (
            param.kind is param.POSITIONAL_OR_KEYWORD and not has_var_positional)

    # Flags that add return values: configuration, not wires, so that the
    # checker knows their value.  With every flag off the documented first
    # result is the output; otherwise the output is whatever comes back.
    for name in override.presets:
        if not by_keyword(parameters[name]):
            raise ImportFailure(
                f"override presets: {name!r} cannot always be passed by keyword")
    flags = [n for n, p in parameters.items()
             if _OUTPUT_FLAG.match(n) and p.default in (False, None)
             and type(p.default) in (bool, type(None))
             and n not in override.presets and by_keyword(p)]
    inputs: list[PinSpec] = []
    config: list[PinSpec] = []
    first_data_input = True
    removed = False       # a parameter before this one is not a pin any more
    try:
        for name, param in parameters.items():
            if name in override.presets:
                removed = True
                continue                      # fixed by the variant: not a pin
            kind = PARAMETER_KINDS[param.kind]
            if removed and kind == "keyword":
                kind = "keyword_only"         # its position is no longer reachable
            default = MISSING if param.default is param.empty else param.default
            variadic = kind in ("var_positional", "var_keyword")
            # Documentation describes *args as a whole, a pin holds one element:
            # only an annotation or an override can type a variadic pin.
            pin_type, source = _input_type(name, default, hints, Doc() if variadic else doc,
                                           override)
            text = doc.params[name].doc if name in doc.params else ""
            if name in override.config or name in flags:
                if not by_keyword(param):
                    raise ImportFailure(
                        f"override config: {name!r} cannot always be passed by keyword")
                config.append(PinSpec(name, pin_type, default, text, source=source))
                removed = True
                continue
            optional = default is not MISSING and not variadic
            hidden = optional and (param.kind is param.KEYWORD_ONLY or name in _PLUMBING)
            if name in override.hidden:
                hidden = True
            if name in override.shown:
                hidden = False
            mutate = name in override.mutates or name == "out" or (
                in_place and first_data_input and not variadic)
            first_data_input = False
            inputs.append(PinSpec(name, pin_type, default, text,
                                  access="mutate" if mutate else "read",
                                  kind=kind, hidden=hidden, source=source))
        outputs = _outputs(hints, doc, override, {p.name for p in inputs}, bool(flags))
        infer = None
        if flags and override.outputs is None and len(outputs) == 1:
            plain = _first_result_type(hints, doc)
            if not isinstance(plain, T.AnyT):
                infer = _flag_infer(tuple(flags), outputs[0].name, plain)
        if _is_ufunc_like(obj) and override.outputs is None:
            value_pins = tuple(p.name for p in inputs if p.kind == "positional")
            if len(value_pins) == obj.nin and len(outputs) != obj.nout:
                outputs = _numbered_outputs(obj.nout)
            if len(value_pins) == obj.nin:
                untouched = {p.name: T.type_of(p.default) for p in inputs
                             if p.name in ("out", "dtype", "signature")
                             and p.default is not MISSING and p.default is not DEFAULT}
                infer = _ufunc_infer(obj, value_pins, tuple(p.name for p in outputs),
                                     untouched)
        if id is None:
            module = getattr(obj, "__module__", None) or "user"
            id = f"{module}.{getattr(obj, '__name__', 'node')}"
        module_name = getattr(obj, "__module__", None) or ""
        name = getattr(obj, "__name__", "") or id.rsplit(".", 1)[-1]
        fields: dict[str, Any] = dict(
            summary=doc.summary, description=doc.description,
            category=category or module_name,
            pure=not (".random" in f".{module_name}" or bool(_IMPURE_NAME.match(name))
                      or in_place),
            tier="annotated" if _touched(override) else "auto",
            version=_package_version(module_name),
        )
        fields.update(override.fields)
        fn = functools.partial(obj, **override.presets) if override.presets else obj
        return NodeSpec(id=id, inputs=tuple(inputs), outputs=tuple(outputs),
                        config=tuple(config), fn=fn, target=target, infer=infer, **fields)
    except (SpecError, T.TypeSyntaxError) as exc:
        raise ImportFailure(str(exc)) from exc


def _first_result_type(hints: Mapping[str, Any], doc: Doc) -> T.Type:
    """Type of the result a flagged function returns with all flags off."""
    if "return" in hints:
        return T.ANY                 # an annotation already describes every case
    if not doc.returns:
        return T.ANY
    described = type_from_phrase(doc.returns[0].type)
    return T.ANY if described is None else _widen_output(described)


def _flag_infer(flags: tuple[str, ...], pin: str, plain: T.Type
                ) -> Callable[..., Mapping[str, T.Type]]:
    def infer(inputs: Mapping[str, Optional[T.Type]], config: Mapping[str, Any]
              ) -> Mapping[str, T.Type]:
        return {} if any(config.get(flag) for flag in flags) else {pin: plain}
    return infer


def _touched(override: Override) -> bool:
    return override != Override()


def _package_version(module_name: str) -> str:
    top = module_name.partition(".")[0]
    module = sys.modules.get(top)
    version = getattr(module, "__version__", "")
    return version if isinstance(version, str) else ""


def _input_type(name: str, default: Any, hints: Mapping[str, Any], doc: Doc,
                override: Override) -> tuple[T.Type, str]:
    """The type of an input pin and where it came from."""
    if name in override.types:
        return T.parse_type(override.types[name]), "override"
    found: Optional[T.Type] = None
    source = ""
    if name in hints:
        annotated = _annotation_type(hints[name])
        if not isinstance(annotated, T.AnyT):
            found, source = annotated, "annotation"
    if found is None and name in doc.params:
        described = type_from_phrase(doc.params[name].type)
        if described is not None and not isinstance(described, T.AnyT):
            found, source = described, "docstring"
    has_value = default is not MISSING and default is not DEFAULT
    if found is None:
        if has_value and type(default) in _TYPED_DEFAULTS:
            return T.type_of(default), "default"
        return T.ANY, ""
    if has_value and not T.conforms(T.plain_scalar(default), found):
        # A default must always be acceptable: "x : int, optional" with x=None.
        try:
            T.accept(T.plain_scalar(default), found)
        except T.TypeMismatch:
            found = T.union(found, T.type_of(T.plain_scalar(default)))
    return found, source


def _numbered_outputs(count: int) -> list[PinSpec]:
    if count == 1:
        return [PinSpec("out", ARRAY_OR_SCALAR)]
    return [PinSpec(f"out{i + 1}", ARRAY_OR_SCALAR) for i in range(count)]


def _outputs(hints: Mapping[str, Any], doc: Doc, override: Override,
             input_names: set[str], flagged: bool = False) -> list[PinSpec]:
    def aliases(name: str) -> Optional[str]:
        target = override.aliases.get(name)
        if target is not None and target not in input_names:
            raise ImportFailure(f"override aliases: no input named {target!r}")
        return target

    if override.outputs is not None:
        return [PinSpec(name, T.parse_type(text) if text else T.ANY, aliases=aliases(name),
                        source="override") for name, text in override.outputs]

    def single(pin_type: T.Type, source: str, text: str = "") -> list[PinSpec]:
        if "out" in override.types:
            pin_type, source = T.parse_type(override.types["out"]), "override"
        return [PinSpec("out", pin_type, doc=text, aliases=aliases("out"), source=source)]

    documented = [r for r in doc.returns]
    named = [r for r in documented if r.name.isidentifier()]
    if "return" in hints:
        annotation = hints["return"]
        if annotation is None or annotation is type(None):
            return []
        annotated = _annotation_type(annotation)
        if (isinstance(annotated, T.TupleT) and not annotated.variadic
                and len(annotated.elems) >= 2 and len(named) == len(annotated.elems)
                and len({r.name for r in named}) == len(named)):
            return [PinSpec(r.name, t, doc=r.doc, aliases=aliases(r.name), source="annotation")
                    for r, t in zip(named, annotated.elems)]
        if not isinstance(annotated, T.AnyT):
            return single(annotated, "annotation")
    if not doc.has_returns_section or not documented:
        return single(T.ANY, "")
    if len(documented) == 1:
        item = documented[0]
        if item.type.strip().lower() == "none" and not item.name:
            return []
        described = type_from_phrase(item.type)
        if described is None or item.optional:
            return single(T.ANY, "", item.doc)
        return single(_widen_output(described), "docstring", item.doc)
    unpackable = (len(named) == len(documented)
                  and len({r.name for r in named}) == len(named)
                  and not flagged
                  and not any(r.optional or _CONDITIONAL.search(r.doc) for r in documented))
    if not unpackable:
        # e.g. np.unique: some results only exist when a flag asks for them
        return single(T.ANY, "")
    pins = []
    for item in documented:
        described = type_from_phrase(item.type)
        source = "override" if item.name in override.types else (
            "docstring" if described is not None else "")
        pin_type = (T.parse_type(override.types[item.name]) if item.name in override.types
                    else _widen_output(described) if described is not None else T.ANY)
        pins.append(PinSpec(item.name, pin_type, doc=item.doc, aliases=aliases(item.name),
                            source=source))
    return pins


# --------------------------------------------------------------------------
# A whole module
# --------------------------------------------------------------------------

class ModuleLibrary:
    """The functions of one Python module, turned into nodes on demand."""

    def __init__(self, module: Union[str, pytypes.ModuleType], *,
                 overrides: Optional[Mapping[str, Override]] = None,
                 prefix: Optional[str] = None) -> None:
        if isinstance(module, str):
            try:
                module = importlib.import_module(module)
            except Exception as exc:
                raise ImportFailure(f"cannot import module {module!r}: {exc}") from exc
        self.module = module
        self.prefix = prefix or module.__name__
        if not all(part.isidentifier() for part in self.prefix.split(".")):
            raise ImportFailure(f"invalid prefix {self.prefix!r}")
        self.overrides = dict(overrides or {})
        #: Why a function could not be imported, by name.  Filled as specs are built.
        self.failures: dict[str, str] = {}
        self._names: Optional[list[str]] = None
        self._functions: set[str] = set()

    def _override_key(self, name: str) -> str:
        top = self.module.__name__.partition(".")[0]
        return f"{self.module.__name__}.{name}"[len(top) + 1:]

    def names(self) -> list[str]:
        """Names of the public functions of the module."""
        if self._names is None:
            module = self.module
            public = getattr(module, "__all__", None)
            listed = isinstance(public, (list, tuple))
            candidates = (list(public or ()) if listed
                          else [n for n in dir(module) if not n.startswith("_")])
            top = module.__name__.partition(".")[0]
            names = []
            for name in candidates:
                if not isinstance(name, str) or not name.isidentifier():
                    continue
                try:
                    obj = getattr(module, name)
                except Exception:
                    continue
                if not callable(obj) or isinstance(obj, (type, pytypes.ModuleType)):
                    continue
                home = (getattr(obj, "__module__", None) or top).partition(".")[0]
                if not listed and home != top:
                    continue          # something the module merely imported
                names.append(name)
            self._functions = set(names)
            for key, override in self.overrides.items():
                # variants declared for this module: "unique_counts", "linalg.eig_values"
                owner, _, variant = key.rpartition(".")
                if (override.function and variant.isidentifier()
                        and self._override_key(variant) == key):
                    names.append(variant)
            self._names = sorted(set(names))
        return list(self._names)

    def ids(self) -> list[str]:
        return [f"{self.prefix}.{name}" for name in self.names()]

    def load(self, id: str) -> Optional[NodeSpec]:
        name = id[len(self.prefix) + 1:]
        if name not in self.names():
            return None
        override = self.overrides.get(self._override_key(name))
        if (override is not None and override.function and override.function != name
                and name in self._functions):
            # A variant must not replace a real function of the same name.
            self.failures[f"{name} (variant)"] = (
                f"a function named {name!r} already exists in {self.module.__name__}")
            override = None
        source = override.function if override is not None and override.function else name
        try:
            if source not in self._functions:
                raise ImportFailure(f"variant of unknown function {source!r}")
            return import_callable(
                getattr(self.module, source), id, override=override,
                target=f"{self.module.__name__}:{source}", category=self.module.__name__)
        except ImportFailure as exc:
            self.failures[name] = str(exc)
        except Exception as exc:    # a broken function must not break the palette
            self.failures[name] = f"{type(exc).__name__}: {exc}"
        return None


def import_module(registry: Registry, module: Union[str, pytypes.ModuleType], *,
                  overrides: "None | str | Path | Mapping[str, Any]" = None,
                  pack: bool = True, prefix: Optional[str] = None) -> ModuleLibrary:
    """Make the public functions of ``module`` available in ``registry``.

    Node ids are ``<module>.<function>``, e.g. ``numpy.histogram``.  Specs are
    built lazily, the first time each one is used.  ``pack=True`` applies the
    overrides bundled with OpenRamen for that library; ``overrides`` adds your
    own on top (a TOML file or a mapping).
    """
    name = module if isinstance(module, str) else module.__name__
    merged: dict[str, Override] = {}
    if pack:
        merged.update(pack_overrides(name.partition(".")[0]))
    if overrides is not None:
        extra = overrides if _is_override_map(overrides) else load_overrides(overrides)  # type: ignore[arg-type]
        merged.update(extra)  # type: ignore[arg-type]
    library = ModuleLibrary(module, overrides=merged, prefix=prefix)
    registry.add_provider(library)
    return library


def _is_override_map(value: Any) -> bool:
    return isinstance(value, Mapping) and all(isinstance(v, Override) for v in value.values())


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------

def pin_outputs(registry: Registry, spec_id: str,
                types: Mapping[str, "T.Type | str"]) -> NodeSpec:
    """Fix the output types of a node, typically to what a run showed.

    The node becomes "annotated": from then on every run checks that its
    outputs still have these types.
    """
    spec = registry.get(spec_id)
    unknown = set(types) - {p.name for p in spec.outputs}
    if unknown:
        raise SpecError(f"{spec_id} has no output {sorted(unknown)[0]!r}")
    outputs = tuple(
        dataclasses.replace(pin, type=T.parse_type(types[pin.name]), source="observed")
        if pin.name in types else pin for pin in spec.outputs)
    tier = spec.tier if spec.tier == "curated" else "annotated"
    # A sampling hook would overrule the pinned type, so it is dropped.
    pinned = dataclasses.replace(spec, outputs=outputs, tier=tier, infer=None)
    return registry.register(pinned, replace=True)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

@dataclass
class ImportReport:
    """How much of a module the importer understood."""

    module: str
    version: str
    candidates: int = 0
    imported: int = 0
    skipped: int = 0              # left out on purpose by an override
    failures: dict[str, str] = field(default_factory=dict)
    inputs: int = 0
    typed_inputs: int = 0
    outputs: int = 0
    typed_outputs: int = 0
    fully_typed: int = 0          # nodes whose every pin has a known type
    documented: int = 0           # nodes with a summary
    sources: dict[str, int] = field(default_factory=dict)

    def share(self, part: int, whole: int) -> str:
        return "n/a" if not whole else f"{100 * part / whole:.0f}%"

    def lines(self) -> list[str]:
        return [
            f"{self.module} {self.version}".rstrip(),
            f"  functions found      {self.candidates}",
            f"  imported as nodes    {self.imported} "
            f"({self.share(self.imported, self.candidates - self.skipped)}"
            + (f", {self.skipped} skipped on purpose)" if self.skipped else ")"),
            f"  with a summary       {self.documented} ({self.share(self.documented, self.imported)})",
            f"  inputs with a type   {self.typed_inputs} of {self.inputs} "
            f"({self.share(self.typed_inputs, self.inputs)})",
            f"  outputs with a type  {self.typed_outputs} of {self.outputs} "
            f"({self.share(self.typed_outputs, self.outputs)})",
            f"  fully typed nodes    {self.fully_typed} ({self.share(self.fully_typed, self.imported)})",
            "  type sources         " + (", ".join(
                f"{k} {v}" for k, v in sorted(self.sources.items(), key=lambda kv: -kv[1])) or "none"),
        ]


def report(library: ModuleLibrary) -> ImportReport:
    """Build every spec of a module and count what was understood."""
    out = ImportReport(library.module.__name__, _package_version(library.module.__name__))
    names = library.names()
    out.candidates = len(names)
    for name in names:
        spec = library.load(f"{library.prefix}.{name}")
        if spec is None:
            continue
        out.imported += 1
        out.documented += bool(spec.summary)
        pins_in = [p for p in spec.inputs + spec.config]
        typed_in = [p for p in pins_in if not isinstance(p.type, T.AnyT)]
        typed_out = [p for p in spec.outputs if not isinstance(p.type, T.AnyT)]
        out.inputs += len(pins_in)
        out.typed_inputs += len(typed_in)
        out.outputs += len(spec.outputs)
        out.typed_outputs += len(typed_out)
        out.fully_typed += (len(typed_in) == len(pins_in) and len(typed_out) == len(spec.outputs))
        for pin in typed_in + typed_out:
            key = pin.source or "unknown"
            out.sources[key] = out.sources.get(key, 0) + 1
    skipped = "skipped by an override"
    out.skipped = sum(reason == skipped for reason in library.failures.values())
    out.failures = {n: r for n, r in library.failures.items() if r != skipped}
    return out
