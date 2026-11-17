"""NodeSpec: the one record that describes a kind of node.

The importer (or a hand-written palette) produces NodeSpecs; the checker, the
runtime and later the editor and front panel only read them.
"""

from __future__ import annotations

import importlib
import inspect
import keyword
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from .. import types as T

__all__ = ["MISSING", "PinSpec", "NodeSpec", "SpecError", "NodeResultError",
           "spec_from_function"]


class SpecError(ValueError):
    """A NodeSpec is inconsistent."""


class NodeResultError(TypeError):
    """A node function returned something its output pins cannot hold."""


class _Missing:
    """Sentinel for "this input has no default: it must be wired or set"."""

    _instance: "Optional[_Missing]" = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False

    def __copy__(self) -> "_Missing":
        return self

    def __deepcopy__(self, memo: dict) -> "_Missing":
        return self

    def __reduce__(self) -> str:
        return "MISSING"


MISSING = _Missing()

_ACCESS = ("read", "mutate")
_KINDS = ("keyword", "positional")
_EXECUTORS = ("inline", "thread", "process")
_TIERS = ("auto", "annotated", "curated")


def _valid_name(name: Any) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


@dataclass(frozen=True)
class PinSpec:
    """One input, output or configuration pin."""

    name: str
    type: T.Type = T.ANY
    default: Any = MISSING           # inputs and config only
    doc: str = ""
    access: str = "read"             # inputs: "read" or "mutate"
    aliases: Optional[str] = None    # outputs: name of the input it shares memory with
    widget: Optional[str] = None     # front-panel hint
    kind: str = "keyword"            # how the argument is passed to the function
    hidden: bool = False             # optional input not shown until asked for

    def __post_init__(self) -> None:
        if not _valid_name(self.name):
            raise SpecError(f"invalid pin name {self.name!r}")
        if not isinstance(self.type, T.Type):
            object.__setattr__(self, "type", T.parse_type(self.type))
        if self.access not in _ACCESS:
            raise SpecError(f"pin {self.name!r}: access must be one of {_ACCESS}")
        if self.kind not in _KINDS:
            raise SpecError(f"pin {self.name!r}: kind must be one of {_KINDS}")

    @property
    def required(self) -> bool:
        return self.default is MISSING


@dataclass(frozen=True)
class NodeSpec:
    """Immutable description of a kind of node."""

    id: str
    inputs: tuple[PinSpec, ...] = ()
    outputs: tuple[PinSpec, ...] = ()
    config: tuple[PinSpec, ...] = ()
    fn: Optional[Callable[..., Any]] = field(default=None, compare=False, repr=False)
    target: str = ""                 # "module:qualname", used when fn is None
    version: str = ""
    summary: str = ""
    description: str = ""
    category: str = ""
    pure: bool = True                # same inputs always give the same outputs
    thread_safe: bool = True
    executor: str = "thread"
    tier: str = "auto"
    #: Optional type function for polymorphic nodes:
    #: ``infer(input_types, config_values) -> {output_pin: Type}``.
    #: ``input_types`` maps pin names to a Type, or ``None`` when unconnected.
    infer: Optional[Callable[[Mapping[str, Optional[T.Type]], Mapping[str, Any]],
                             Mapping[str, T.Type]]] = field(
        default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or any(
                not part.isidentifier() for part in self.id.split(".")):
            raise SpecError(f"invalid spec id {self.id!r}")
        for attr in ("inputs", "outputs", "config"):
            pins = tuple(getattr(self, attr))
            if not all(isinstance(p, PinSpec) for p in pins):
                raise SpecError(f"{self.id}: {attr} must be PinSpec objects")
            object.__setattr__(self, attr, pins)
        argument_names = [p.name for p in self.inputs + self.config]
        if len(set(argument_names)) != len(argument_names):
            raise SpecError(f"{self.id}: duplicate input or config pin name")
        output_names = [p.name for p in self.outputs]
        if len(set(output_names)) != len(output_names):
            raise SpecError(f"{self.id}: duplicate output pin name")
        for pin in self.outputs:
            if pin.aliases is not None and pin.aliases not in {p.name for p in self.inputs}:
                raise SpecError(
                    f"{self.id}: output {pin.name!r} aliases unknown input {pin.aliases!r}")
        seen_keyword = False
        for pin in self.inputs:
            if pin.kind == "keyword":
                seen_keyword = True
            elif seen_keyword:
                raise SpecError(
                    f"{self.id}: positional input {pin.name!r} follows a keyword input")
        for pin in self.config:
            if pin.kind != "keyword":
                raise SpecError(f"{self.id}: config pin {pin.name!r} must be a keyword")
        if self.executor not in _EXECUTORS:
            raise SpecError(f"{self.id}: executor must be one of {_EXECUTORS}")
        if self.tier not in _TIERS:
            raise SpecError(f"{self.id}: tier must be one of {_TIERS}")
        if self.fn is None and not self.target:
            raise SpecError(f"{self.id}: needs a function or a target")
        if self.fn is not None and not callable(self.fn):
            raise SpecError(f"{self.id}: fn is not callable")

    # -- lookups ------------------------------------------------------------

    def input(self, name: str) -> Optional[PinSpec]:
        return next((p for p in self.inputs if p.name == name), None)

    def output(self, name: str) -> Optional[PinSpec]:
        return next((p for p in self.outputs if p.name == name), None)

    def config_pin(self, name: str) -> Optional[PinSpec]:
        return next((p for p in self.config if p.name == name), None)

    # -- calling ------------------------------------------------------------

    def resolve(self) -> Callable[..., Any]:
        """The Python callable behind this node."""
        if self.fn is not None:
            return self.fn
        module, _, qualname = self.target.partition(":")
        if not module or not qualname:
            raise SpecError(f"{self.id}: target {self.target!r} is not 'module:qualname'")
        try:
            obj: Any = importlib.import_module(module)
            for part in qualname.split("."):
                obj = getattr(obj, part)
        except (ImportError, AttributeError) as exc:
            raise SpecError(f"{self.id}: cannot import {self.target!r}: {exc}") from exc
        if not callable(obj):
            raise SpecError(f"{self.id}: target {self.target!r} is not callable")
        object.__setattr__(self, "fn", obj)  # cache; fn is excluded from equality
        return obj

    def call(self, inputs: Mapping[str, Any],
             config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Call the function and spread its result over the output pins.

        ``inputs`` holds a value for every input that should be passed; an
        input left out falls back to the function's own default.
        """
        fn = self.resolve()
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        gap: Optional[str] = None
        for pin in self.inputs:
            if pin.name not in inputs:
                if pin.kind == "positional":
                    gap = gap or pin.name
                continue
            if pin.kind == "positional":
                if gap is not None:
                    raise TypeError(
                        f"{self.id}: positional input {pin.name!r} is set but "
                        f"{gap!r} before it is not")
                args.append(inputs[pin.name])
            else:
                kwargs[pin.name] = inputs[pin.name]
        for pin in self.config:
            if config is not None and pin.name in config:
                kwargs[pin.name] = config[pin.name]
        return self.spread(fn(*args, **kwargs))

    def spread(self, result: Any) -> dict[str, Any]:
        """Distribute a function's return value over the output pins."""
        count = len(self.outputs)
        if count == 0:
            return {}
        if count == 1:
            return {self.outputs[0].name: result}
        if isinstance(result, Mapping):
            names = {p.name for p in self.outputs}
            if set(result) != names:
                raise NodeResultError(
                    f"{self.id}: returned keys {sorted(map(str, result))}, "
                    f"expected {sorted(names)}")
            return {p.name: result[p.name] for p in self.outputs}
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            if len(result) != count:
                raise NodeResultError(
                    f"{self.id}: returned {len(result)} values for {count} outputs")
            return {p.name: v for p, v in zip(self.outputs, result)}
        raise NodeResultError(
            f"{self.id}: returned {type(result).__name__} for {count} outputs; "
            "expected a tuple")


def _first_line(doc: Optional[str]) -> str:
    for line in (doc or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _annotations(fn: Callable[..., Any]) -> dict[str, Any]:
    """Annotations of ``fn``, resolved one by one.

    ``typing.get_type_hints`` is all or nothing: one unresolvable name loses
    every annotation.  Falling back per parameter keeps the good ones.
    """
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception:
        pass
    raw = dict(getattr(fn, "__annotations__", {}) or {})
    namespace = getattr(fn, "__globals__", {})
    hints: dict[str, Any] = {}
    for name, annotation in raw.items():
        if isinstance(annotation, str):
            try:
                annotation = eval(annotation, namespace)  # what typing itself does
            except Exception:
                pass  # left as text: it may be OpenRamen type syntax
        hints[name] = annotation
    return hints


def spec_from_function(
    fn: Callable[..., Any],
    id: Optional[str] = None,
    *,
    outputs: "Sequence[str] | str | None" = None,
    types: Optional[Mapping[str, Any]] = None,
    config: Sequence[str] = (),
    mutates: Sequence[str] = (),
    docs: Optional[Mapping[str, str]] = None,
    **fields: Any,
) -> NodeSpec:
    """Build a NodeSpec from a Python function and its annotations.

    * ``outputs`` names the output pins.  One name (the default, ``"out"``)
      takes the whole return value; several names unpack a returned tuple.
      A function annotated ``-> None`` has no outputs.
    * ``types`` overrides the type of any pin, by name.
    * ``config`` lists parameters that are configuration, not wirable inputs.
    * ``mutates`` lists inputs the function writes to in place.
    * Remaining keyword arguments are NodeSpec fields (``category``, ...).
    """
    if not callable(fn):
        raise SpecError(f"{fn!r} is not callable")
    types = dict(types or {})
    docs = dict(docs or {})
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"cannot read the signature of {fn!r}: {exc}") from exc
    hints = _annotations(fn)

    def pin_type(name: str, default: Any = MISSING) -> T.Type:
        if name in types:
            return T.parse_type(types[name])
        if name in hints:
            return T.from_annotation(hints[name])
        if default is not MISSING and default is not None:
            return T.type_of(default)
        return T.ANY

    unknown = (set(config) | set(mutates)) - set(signature.parameters)
    if unknown:
        raise SpecError(f"{fn!r} has no parameter named {sorted(unknown)[0]!r}")

    inputs: list[PinSpec] = []
    config_pins: list[PinSpec] = []
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise SpecError(
                f"{getattr(fn, '__name__', fn)!s}: variadic parameter "
                f"{name!r} is not supported yet")
        default = MISSING if param.default is param.empty else param.default
        kind = "positional" if param.kind is param.POSITIONAL_ONLY else "keyword"
        ptype = pin_type(name, default)
        if name in config:
            if kind != "keyword":
                raise SpecError(f"config parameter {name!r} cannot be positional-only")
            config_pins.append(PinSpec(name, ptype, default, docs.get(name, "")))
        else:
            inputs.append(PinSpec(
                name, ptype, default, docs.get(name, ""),
                access="mutate" if name in mutates else "read", kind=kind))

    returns = hints.get("return", inspect.Signature.empty)
    if isinstance(outputs, str):
        outputs = (outputs,)
    if outputs is None:
        outputs = () if returns is None or returns is type(None) else ("out",)
    outputs = tuple(outputs)
    return_type = T.ANY if returns is inspect.Signature.empty else T.from_annotation(returns)
    if len(outputs) == 1:
        element_types = [return_type]
    else:
        element_types = [T.ANY] * len(outputs)
        if (isinstance(return_type, T.TupleT) and not return_type.variadic
                and len(return_type.elems) == len(outputs)):
            element_types = list(return_type.elems)
        elif isinstance(return_type, T.RecordT) and [n for n, _ in return_type.fields] == list(outputs):
            element_types = [ft for _, ft in return_type.fields]
    output_pins = tuple(
        PinSpec(name, T.parse_type(types[name]) if name in types else etype,
                doc=docs.get(name, ""))
        for name, etype in zip(outputs, element_types))

    if id is None:
        module = getattr(fn, "__module__", None) or "user"
        id = f"{module}.{getattr(fn, '__name__', 'node')}"
    fields.setdefault("summary", _first_line(getattr(fn, "__doc__", None)))
    fields.setdefault("tier", "curated")
    return NodeSpec(id=id, inputs=tuple(inputs), outputs=output_pins,
                    config=tuple(config_pins), fn=fn, **fields)
