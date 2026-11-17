"""The OpenRamen type algebra.

Every pin in a graph has a type from this module.  The algebra is deliberately
smaller than Python's ``typing``: it only has what a dataflow wire needs.

Main entry points:

* :func:`parse_type` / :func:`format_type` -- text form used in files and docs.
* :func:`check` -- is a wire from ``src`` to ``dst`` allowed, and how?
* :func:`unify` / :func:`substitute` -- resolve type variables of generic nodes.
* :func:`join` -- smallest common type of two branches.
* :func:`type_of` / :func:`conforms` / :func:`accept` -- run-time side.
* :func:`from_annotation` -- translate a Python annotation.
"""

from __future__ import annotations

import collections.abc as cabc
import dataclasses
import enum
import functools
import importlib
import itertools
import os
import re
import types as _pytypes
import typing
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Callable, Iterable, Optional

import numpy as np

__all__ = [
    "Type", "ScalarT", "AnyT", "VarT", "EnumT", "ArrayT", "ListT", "TupleT",
    "DictT", "UnionT", "RecordT", "ObjectT", "CallableT", "IteratorT",
    "BOOL", "INT", "FLOAT", "COMPLEX", "STR", "BYTES", "PATH", "NONE", "ANY",
    "union", "optional", "parse_type", "format_type",
    "Verdict", "Compat", "check", "delivered", "join", "unify", "substitute", "free_vars",
    "type_of", "conforms", "accept", "from_annotation", "dtype_name",
    "plain_scalar",
    "TypeSyntaxError", "TypeMismatch", "UnifyError",
]

#: How many elements of a container are inspected by run-time checks.
CHECK_LIMIT = 4096

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


class TypeSyntaxError(ValueError):
    """Raised when a type string cannot be parsed."""


class TypeMismatch(TypeError):
    """Raised when a run-time value does not fit the type of a pin."""


class UnifyError(Exception):
    """Raised when a type variable would need two incompatible bindings."""

    def __init__(self, var: str, first: Any, second: Any):
        super().__init__(f"conflicting types for ~{var}: {first} and {second}")
        self.var = var
        self.first = first
        self.second = second


# --------------------------------------------------------------------------
# Type objects
# --------------------------------------------------------------------------

class Type:
    """Base class of all OpenRamen types.  Instances are immutable."""

    __slots__ = ()

    def __str__(self) -> str:
        return format_type(self)

    def __repr__(self) -> str:
        return f"<Type {format_type(self)}>"


@dataclass(frozen=True, repr=False)
class ScalarT(Type):
    name: str


@dataclass(frozen=True, repr=False)
class AnyT(Type):
    """Unknown until run time."""


@dataclass(frozen=True, repr=False)
class VarT(Type):
    """A type variable, written ``~T``."""

    name: str


@dataclass(frozen=True, repr=False)
class EnumT(Type):
    name: str
    members: tuple[str, ...] = ()


@dataclass(frozen=True, repr=False)
class ArrayT(Type):
    """A NumPy array.  ``None`` in a slot means "not known statically"."""

    dtype: "str | VarT | None" = None
    ndim: "int | VarT | None" = None

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str) and not _IDENT.match(self.dtype):
            raise ValueError(f"invalid dtype name {self.dtype!r}")
        if isinstance(self.ndim, bool) or (
            isinstance(self.ndim, int) and self.ndim < 0
        ):
            raise ValueError(f"invalid ndim {self.ndim!r}")


@dataclass(frozen=True, repr=False)
class ListT(Type):
    elem: Type = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.elem is None:
            object.__setattr__(self, "elem", ANY)


@dataclass(frozen=True, repr=False)
class TupleT(Type):
    """Fixed-length tuple, or ``Tuple[T, ...]`` when ``variadic`` is set."""

    elems: tuple[Type, ...] = ()
    variadic: bool = False

    def __post_init__(self) -> None:
        if self.variadic and len(self.elems) != 1:
            raise ValueError("a variadic tuple has exactly one element type")


@dataclass(frozen=True, repr=False)
class DictT(Type):
    key: Type = None  # type: ignore[assignment]
    value: Type = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.key is None:
            object.__setattr__(self, "key", ANY)
        if self.value is None:
            object.__setattr__(self, "value", ANY)


@dataclass(frozen=True, repr=False)
class UnionT(Type):
    """Use :func:`union` to build one; it normalises the options."""

    options: tuple[Type, ...]


@dataclass(frozen=True, repr=False)
class RecordT(Type):
    """Named fields: a dataclass, NamedTuple or TypedDict."""

    name: str
    fields: tuple[tuple[str, Type], ...] = ()


@dataclass(frozen=True, repr=False)
class ObjectT(Type):
    """An opaque class instance with reference semantics."""

    cls: str  # "module:QualName"


@dataclass(frozen=True, repr=False)
class CallableT(Type):
    """A function on a wire.  ``params is None`` means unknown signature."""

    params: Optional[tuple[Type, ...]] = None
    result: Type = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.result is None:
            object.__setattr__(self, "result", ANY)


@dataclass(frozen=True, repr=False)
class IteratorT(Type):
    elem: Type = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.elem is None:
            object.__setattr__(self, "elem", ANY)


BOOL = ScalarT("Bool")
INT = ScalarT("Int")
FLOAT = ScalarT("Float")
COMPLEX = ScalarT("Complex")
STR = ScalarT("Str")
BYTES = ScalarT("Bytes")
PATH = ScalarT("Path")
NONE = ScalarT("None")
ANY = AnyT()

_SCALARS = {t.name: t for t in (BOOL, INT, FLOAT, COMPLEX, STR, BYTES, PATH, NONE)}


def union(*options: Type) -> Type:
    """Build a normalised union: flattened, de-duplicated, order-independent."""
    flat: list[Type] = []
    for opt in options:
        if isinstance(opt, UnionT):
            flat.extend(opt.options)
        else:
            flat.append(opt)
    if not flat:
        raise ValueError("a union needs at least one option")
    if any(isinstance(o, AnyT) for o in flat):
        return ANY
    unique = sorted(set(flat), key=format_type)
    if len(unique) == 1:
        return unique[0]
    return UnionT(tuple(unique))


def optional(t: Type) -> Type:
    return union(t, NONE)


# --------------------------------------------------------------------------
# Text form
# --------------------------------------------------------------------------

def _fmt_slot(slot: Any) -> str:
    if slot is None:
        return "_"
    if isinstance(slot, VarT):
        return "~" + slot.name
    return str(slot)


def format_type(t: Type) -> str:
    """Canonical text form of a type; :func:`parse_type` reads it back."""
    if isinstance(t, ScalarT):
        return t.name
    if isinstance(t, AnyT):
        return "Any"
    if isinstance(t, VarT):
        return "~" + t.name
    if isinstance(t, EnumT):
        if not t.members:
            return f"Enum[{t.name}]"
        return f"Enum[{t.name}: {', '.join(t.members)}]"
    if isinstance(t, ArrayT):
        return f"Array[{_fmt_slot(t.dtype)}, {_fmt_slot(t.ndim)}]"
    if isinstance(t, ListT):
        return f"List[{format_type(t.elem)}]"
    if isinstance(t, TupleT):
        if t.variadic:
            return f"Tuple[{format_type(t.elems[0])}, ...]"
        return f"Tuple[{', '.join(format_type(e) for e in t.elems)}]"
    if isinstance(t, DictT):
        return f"Dict[{format_type(t.key)}, {format_type(t.value)}]"
    if isinstance(t, UnionT):
        if len(t.options) == 2 and NONE in t.options:
            other = t.options[0] if t.options[1] == NONE else t.options[1]
            return f"Optional[{format_type(other)}]"
        return f"Union[{', '.join(format_type(o) for o in t.options)}]"
    if isinstance(t, RecordT):
        if not t.fields:
            return f"Record[{t.name}]"
        body = ", ".join(f"{n}={format_type(ft)}" for n, ft in t.fields)
        return f"Record[{t.name}: {body}]"
    if isinstance(t, ObjectT):
        return f"Object[{t.cls}]"
    if isinstance(t, CallableT):
        if t.params is None:
            return f"Callable[..., {format_type(t.result)}]"
        params = ", ".join(format_type(p) for p in t.params)
        return f"Callable[[{params}], {format_type(t.result)}]"
    if isinstance(t, IteratorT):
        return f"Iterator[{format_type(t.elem)}]"
    raise TypeError(f"not an OpenRamen type: {t!r}")


_TOKEN = re.compile(
    r"\s*(?:(?P<dots>\.\.\.)|(?P<ident>[A-Za-z_][A-Za-z0-9_.]*)"
    r"|(?P<int>\d+)|(?P<punct>[\[\],:=~]))"
)


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens: list[tuple[str, str]] = []
        pos = 0
        end = len(text)
        while pos < end:
            if text[pos:].strip() == "":
                break
            m = _TOKEN.match(text, pos)
            if m is None:
                raise TypeSyntaxError(
                    f"unexpected character {text[pos:].lstrip()[:1]!r} in type {text!r}"
                )
            kind = m.lastgroup
            assert kind is not None
            self.tokens.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0

    # -- token helpers
    def peek(self) -> tuple[str, str]:
        if self.i < len(self.tokens):
            return self.tokens[self.i]
        return ("end", "")

    def next(self) -> tuple[str, str]:
        tok = self.peek()
        self.i += 1
        return tok

    def fail(self, expected: str) -> "typing.NoReturn":
        kind, value = self.peek()
        got = "end of input" if kind == "end" else repr(value)
        raise TypeSyntaxError(f"expected {expected}, got {got} in type {self.text!r}")

    def expect(self, value: str) -> None:
        if self.peek() != ("punct", value):
            self.fail(repr(value))
        self.i += 1

    def accept_punct(self, value: str) -> bool:
        if self.peek() == ("punct", value):
            self.i += 1
            return True
        return False

    def ident(self, what: str = "a name") -> str:
        kind, value = self.peek()
        if kind != "ident":
            self.fail(what)
        self.i += 1
        return value

    def dotted(self, what: str) -> str:
        name = self.ident(what)
        if not _DOTTED.match(name):
            raise TypeSyntaxError(f"invalid {what[2:].lstrip()} {name!r}")
        return name

    def plain_ident(self, what: str) -> str:
        name = self.ident(what)
        if not _IDENT.match(name):
            raise TypeSyntaxError(f"{what} may not contain dots: {name!r}")
        return name

    # -- grammar
    def parse(self) -> Type:
        t = self.type()
        if self.peek()[0] != "end":
            self.fail("end of input")
        return t

    def type(self) -> Type:
        if self.accept_punct("~"):
            return VarT(self.plain_ident("a type variable name"))
        name = self.ident("a type")
        if name in _SCALARS:
            return _SCALARS[name]
        if name == "Any":
            return ANY
        handler = getattr(self, "_p_" + name, None)
        if handler is None:
            raise TypeSyntaxError(f"unknown type {name!r} in {self.text!r}")
        return handler()

    def _slot(self, want_int: bool) -> Any:
        kind, value = self.peek()
        if self.accept_punct("~"):
            return VarT(self.plain_ident("a variable name"))
        if kind == "ident" and value == "_":
            self.i += 1
            return None
        if want_int:
            if kind != "int":
                self.fail("a number of dimensions, '_' or a variable")
            self.i += 1
            return int(value)
        return self.plain_ident("a dtype name, '_' or a variable")

    def _p_Array(self) -> Type:
        if not self.accept_punct("["):
            return ArrayT()
        dtype = self._slot(False)
        self.expect(",")
        ndim = self._slot(True)
        self.expect("]")
        return ArrayT(dtype, ndim)

    def _p_List(self) -> Type:
        if not self.accept_punct("["):
            return ListT(ANY)
        elem = self.type()
        self.expect("]")
        return ListT(elem)

    def _p_Iterator(self) -> Type:
        if not self.accept_punct("["):
            return IteratorT(ANY)
        elem = self.type()
        self.expect("]")
        return IteratorT(elem)

    def _p_Tuple(self) -> Type:
        if not self.accept_punct("["):
            return TupleT((ANY,), variadic=True)
        if self.accept_punct("]"):
            return TupleT(())
        elems = [self.type()]
        while self.accept_punct(","):
            if self.peek()[0] == "dots":
                self.i += 1
                self.expect("]")
                if len(elems) != 1:
                    raise TypeSyntaxError(
                        f"'...' needs exactly one element type in {self.text!r}"
                    )
                return TupleT((elems[0],), variadic=True)
            elems.append(self.type())
        self.expect("]")
        return TupleT(tuple(elems))

    def _p_Dict(self) -> Type:
        if not self.accept_punct("["):
            return DictT(ANY, ANY)
        key = self.type()
        self.expect(",")
        value = self.type()
        self.expect("]")
        return DictT(key, value)

    def _p_Union(self) -> Type:
        self.expect("[")
        options = [self.type()]
        while self.accept_punct(","):
            options.append(self.type())
        self.expect("]")
        return union(*options)

    def _p_Optional(self) -> Type:
        self.expect("[")
        inner = self.type()
        self.expect("]")
        return optional(inner)

    def _p_Enum(self) -> Type:
        self.expect("[")
        name = self.dotted("an enum name")
        members: list[str] = []
        if self.accept_punct(":"):
            members.append(self.plain_ident("a member name"))
            while self.accept_punct(","):
                members.append(self.plain_ident("a member name"))
        self.expect("]")
        return EnumT(name, tuple(members))

    def _p_Record(self) -> Type:
        self.expect("[")
        name = self.dotted("a record name")
        fields: list[tuple[str, Type]] = []
        if self.accept_punct(":"):
            while True:
                fname = self.plain_ident("a field name")
                self.expect("=")
                fields.append((fname, self.type()))
                if not self.accept_punct(","):
                    break
        self.expect("]")
        seen = [n for n, _ in fields]
        if len(set(seen)) != len(seen):
            raise TypeSyntaxError(f"duplicate field in {self.text!r}")
        return RecordT(name, tuple(fields))

    def _p_Object(self) -> Type:
        self.expect("[")
        module = self.dotted("a module path")
        self.expect(":")
        qual = self.dotted("a class name")
        self.expect("]")
        return ObjectT(f"{module}:{qual}")

    def _p_Callable(self) -> Type:
        if not self.accept_punct("["):
            return CallableT(None, ANY)
        params: Optional[tuple[Type, ...]]
        if self.peek()[0] == "dots":
            self.i += 1
            params = None
        else:
            self.expect("[")
            items: list[Type] = []
            if not self.accept_punct("]"):
                items.append(self.type())
                while self.accept_punct(","):
                    items.append(self.type())
                self.expect("]")
            params = tuple(items)
        self.expect(",")
        result = self.type()
        self.expect("]")
        return CallableT(params, result)


def parse_type(text: "str | Type") -> Type:
    """Parse the text form of a type, e.g. ``"Array[float64, 2]"``."""
    if isinstance(text, Type):
        return text
    if not isinstance(text, str):
        raise TypeError(f"expected a type string, got {type(text).__name__}")
    return _Parser(text).parse()


# --------------------------------------------------------------------------
# NumPy helpers
# --------------------------------------------------------------------------

def dtype_name(dtype: Any) -> str:
    """A dtype name that is always a plain identifier."""
    dt = np.dtype(dtype)
    name = dt.name
    if _IDENT.match(name):
        return name
    return dt.type.__name__  # e.g. "datetime64" for "datetime64[ns]"


def _np_dtype(name: Any) -> "np.dtype | None":
    if not isinstance(name, str):
        return None
    try:
        return np.dtype(name)
    except TypeError:
        return None


_SCALAR_CLASSES: dict[str, tuple[type, ...]] = {
    "Bool": (bool, np.bool_), "Int": (int, np.integer),
    "Float": (float, np.floating), "Complex": (complex, np.complexfloating),
    "Str": (str,), "Bytes": (bytes,), "Path": (PurePath,), "None": (type(None),),
}
_TOWER = ("Int", "Float", "Complex")


# --------------------------------------------------------------------------
# Class paths (Enum, Record and Object names)
# --------------------------------------------------------------------------

_CLASSES: "weakref.WeakValueDictionary[str, type]" = weakref.WeakValueDictionary()


def _qualname(cls: type) -> str:
    return cls.__qualname__.replace("<locals>", "_locals_")


def _dotted_name(cls: type) -> str:
    name = f"{cls.__module__}.{_qualname(cls)}"
    _remember(name, cls)
    return name


def _class_path(cls: type) -> str:
    path = f"{cls.__module__}:{_qualname(cls)}"
    _remember(path, cls)
    return path


def _remember(key: str, cls: type) -> None:
    try:
        _CLASSES[key] = cls
    except TypeError:  # class without weakref support
        pass


def _resolve_class(path: str) -> "type | None":
    cls = _CLASSES.get(path)
    if cls is not None:
        return cls
    module, _, qual = path.partition(":")
    if not qual:
        return None
    try:
        obj: Any = importlib.import_module(module)
        for part in qual.split("."):
            obj = getattr(obj, part)
    except Exception:
        return None
    if isinstance(obj, type):
        _remember(path, obj)
        return obj
    return None


# --------------------------------------------------------------------------
# Compatibility of a wire
# --------------------------------------------------------------------------

class Verdict(enum.IntEnum):
    """Outcome of connecting ``src`` to ``dst``, from best to worst."""

    EXACT = 0         # plain wire
    COERCION = 1      # a known, lossless conversion is inserted
    DYNAMIC = 2       # decided at run time by :func:`accept`
    INCOMPATIBLE = 3  # broken wire


@dataclass(frozen=True)
class Compat:
    verdict: Verdict
    convert: Optional[Callable[[Any], Any]] = dataclasses.field(
        default=None, compare=False
    )
    reason: str = ""
    #: For a coercion into a union: the option the value is converted to.
    target: Optional[Type] = None

    @property
    def ok(self) -> bool:
        return self.verdict is not Verdict.INCOMPATIBLE


_EXACT = Compat(Verdict.EXACT)
_DYNAMIC = Compat(Verdict.DYNAMIC)


def _no(src: Type, dst: Type, why: str = "") -> Compat:
    reason = f"{format_type(src)} does not fit {format_type(dst)}"
    if why:
        reason += f" ({why})"
    return Compat(Verdict.INCOMPATIBLE, None, reason)


_SCALAR_COERCIONS: dict[tuple[str, str], Callable[[Any], Any]] = {
    ("Int", "Float"): float,
    ("Int", "Complex"): complex,
    ("Float", "Complex"): complex,
    ("Path", "Str"): os.fspath,
    ("Str", "Path"): Path,
}


def check(src: Type, dst: Type) -> Compat:
    """Can a value of type ``src`` flow into a pin of type ``dst``?"""
    if isinstance(dst, (AnyT, VarT)) or src == dst:
        return _EXACT
    if isinstance(src, (AnyT, VarT)):
        return _DYNAMIC
    if isinstance(src, UnionT):
        results = [check(o, dst) for o in src.options]
        if all(r.verdict is Verdict.EXACT for r in results):
            return _EXACT
        if all(r.verdict is Verdict.INCOMPATIBLE for r in results):
            return _no(src, dst)
        return _DYNAMIC
    if isinstance(dst, UnionT):
        option, best = min(((o, check(src, o)) for o in dst.options),
                           key=lambda pair: pair[1].verdict)
        if not best.ok:
            return _no(src, dst)
        if best.verdict is Verdict.COERCION:
            return dataclasses.replace(best, target=best.target or option)
        return best
    if isinstance(src, RecordT) and not isinstance(dst, (RecordT, ObjectT)):
        # NamedTuples are tuples and TypedDicts are dicts at run time.
        cls = _CLASSES.get(src.name)
        if cls is None:
            if isinstance(dst, (TupleT, ListT, DictT)):
                return _DYNAMIC
        elif _is_namedtuple_class(cls):
            return check(TupleT(tuple(ft for _, ft in src.fields)), dst)
        elif typing.is_typeddict(cls):
            values = union(*(ft for _, ft in src.fields)) if src.fields else ANY
            return check(DictT(STR, values), dst)
        return _no(src, dst)
    if isinstance(src, ObjectT) and not isinstance(dst, ObjectT):
        # A class-typed value flowing into a structural pin can only be
        # decided at run time, and only if the classes are related at all.
        a = _resolve_class(src.cls)
        classes = _static_classes(dst)
        if a is not None and (classes is None or any(_related(a, c) for c in classes)):
            return _DYNAMIC
        return _no(src, dst)
    if isinstance(dst, ScalarT):
        if isinstance(src, ScalarT):
            conv = _SCALAR_COERCIONS.get((src.name, dst.name))
            if conv is not None:
                return Compat(Verdict.COERCION, conv)
        return _no(src, dst)
    if isinstance(dst, ArrayT):
        return _check_array(src, dst)
    if isinstance(dst, ListT):
        return _check_list(src, dst)
    if isinstance(dst, TupleT):
        return _check_tuple(src, dst)
    if isinstance(dst, DictT):
        return _check_dict(src, dst)
    if isinstance(dst, IteratorT):
        return _check_iterator(src, dst)
    if isinstance(dst, RecordT):
        return _check_record(src, dst)
    if isinstance(dst, ObjectT):
        return _check_object(src, dst)
    if isinstance(dst, CallableT):
        return _check_callable(src, dst)
    return _no(src, dst)  # EnumT and anything unequal


def delivered(src: Type, dst: Type, compat: Optional[Compat] = None) -> Type:
    """Type of the value a ``dst`` pin receives from a ``src`` wire.

    An exact wire delivers ``src`` itself, which may be more precise than the
    pin; a coerced or run-time checked wire delivers the pin's type.
    """
    if compat is None:
        compat = check(src, dst)
    if compat.verdict is Verdict.EXACT and not isinstance(src, (AnyT, VarT)):
        return src
    return compat.target or dst


def _check_array(src: Type, dst: ArrayT) -> Compat:
    if not isinstance(src, ArrayT):
        return _no(src, dst)
    verdict = Verdict.EXACT
    convert: Optional[Callable[[Any], Any]] = None
    if isinstance(dst.ndim, int):
        if isinstance(src.ndim, int):
            if src.ndim != dst.ndim:
                return _no(src, dst, "different number of dimensions")
        else:
            verdict = Verdict.DYNAMIC
    if isinstance(dst.dtype, str):
        if isinstance(src.dtype, str):
            if src.dtype != dst.dtype:
                a, b = _np_dtype(src.dtype), _np_dtype(dst.dtype)
                if a is None or b is None or not np.can_cast(a, b, "safe"):
                    return _no(src, dst, "lossy dtype conversion")
                convert = lambda v, b=b: v.astype(b)  # noqa: E731
                verdict = max(verdict, Verdict.COERCION)
        else:
            verdict = Verdict.DYNAMIC
    return Compat(verdict, convert if verdict is Verdict.COERCION else None)


def _lift(elem: Compat, src: Type, dst: Type, build: Callable[..., Any]) -> Compat:
    """Compatibility of a homogeneous container from that of its elements."""
    if not elem.ok:
        return _no(src, dst, elem.reason)
    if elem.verdict is Verdict.COERCION and elem.convert is not None:
        return Compat(Verdict.COERCION, lambda xs, f=elem.convert: build(f, xs))
    return Compat(elem.verdict)


def _check_list(src: Type, dst: ListT) -> Compat:
    if isinstance(src, ListT):
        return _lift(check(src.elem, dst.elem), src, dst,
                     lambda f, xs: [f(x) for x in xs])
    if isinstance(src, TupleT):
        parts = [check(e, dst.elem) for e in src.elems]
        if not all(p.ok for p in parts):
            return _no(src, dst)
        if any(p.verdict is Verdict.DYNAMIC for p in parts):
            return _DYNAMIC
        if src.variadic:
            f = parts[0].convert
            if f is None:
                return Compat(Verdict.COERCION, list)
            return Compat(Verdict.COERCION, lambda xs, f=f: [f(x) for x in xs])
        convs = [p.convert for p in parts]
        return Compat(
            Verdict.COERCION,
            lambda xs, convs=convs: [c(x) if c else x for c, x in zip(convs, xs)],
        )
    return _no(src, dst)


def _check_tuple(src: Type, dst: TupleT) -> Compat:
    if not isinstance(src, TupleT):
        return _no(src, dst)
    if dst.variadic:
        parts = [check(e, dst.elems[0]) for e in src.elems]
        if not all(p.ok for p in parts):
            return _no(src, dst)
        worst = max((p.verdict for p in parts), default=Verdict.EXACT)
        if worst is Verdict.COERCION:
            if src.variadic:
                f = parts[0].convert
                return Compat(Verdict.COERCION,
                              lambda xs, f=f: tuple(f(x) for x in xs))
            convs = [p.convert for p in parts]
            return Compat(
                Verdict.COERCION,
                lambda xs, convs=convs: tuple(
                    c(x) if c else x for c, x in zip(convs, xs)),
            )
        return Compat(worst)
    if src.variadic:
        if not all(check(src.elems[0], d).ok for d in dst.elems):
            return _no(src, dst)
        return _DYNAMIC  # the length is only known at run time
    if len(src.elems) != len(dst.elems):
        return _no(src, dst, "different lengths")
    parts = [check(s, d) for s, d in zip(src.elems, dst.elems)]
    if not all(p.ok for p in parts):
        return _no(src, dst)
    worst = max((p.verdict for p in parts), default=Verdict.EXACT)
    if worst is Verdict.COERCION:
        convs = [p.convert for p in parts]
        return Compat(
            Verdict.COERCION,
            lambda xs, convs=convs: tuple(
                c(x) if c else x for c, x in zip(convs, xs)),
        )
    return Compat(worst)


def _check_dict(src: Type, dst: DictT) -> Compat:
    if not isinstance(src, DictT):
        return _no(src, dst)
    key = check(src.key, dst.key)
    if not key.ok or key.verdict is Verdict.COERCION:
        return _no(src, dst, "keys are never converted")
    value = _lift(check(src.value, dst.value), src, dst,
                  lambda f, d: {k: f(v) for k, v in d.items()})
    if not value.ok:
        return value
    if key.verdict is Verdict.DYNAMIC:
        return _DYNAMIC
    return value


def _check_iterator(src: Type, dst: IteratorT) -> Compat:
    if isinstance(src, IteratorT):
        return _lift(check(src.elem, dst.elem), src, dst, lambda f, it: map(f, it))
    if isinstance(src, ListT) or (isinstance(src, TupleT) and src.variadic):
        elem_t = src.elem if isinstance(src, ListT) else src.elems[0]
        elem = check(elem_t, dst.elem)
        if not elem.ok:
            return _no(src, dst)
        if elem.verdict is Verdict.DYNAMIC:
            return _DYNAMIC
        if elem.convert is not None:
            return Compat(Verdict.COERCION, lambda xs, f=elem.convert: map(f, xs))
        return Compat(Verdict.COERCION, iter)
    return _no(src, dst)


def _check_record(src: Type, dst: RecordT) -> Compat:
    if isinstance(src, DictT):
        # A plain dict may be an instance of a TypedDict record.
        cls = _CLASSES.get(dst.name)
        if (cls is None or typing.is_typeddict(cls)) and check(src.key, STR).ok:
            return _DYNAMIC
        return _no(src, dst)
    if not isinstance(src, RecordT) or src.name != dst.name:
        return _no(src, dst)
    src_fields = dict(src.fields)
    if set(src_fields) != {n for n, _ in dst.fields}:
        return _no(src, dst, "different fields")
    verdict = Verdict.EXACT
    for name, dst_t in dst.fields:
        part = check(src_fields[name], dst_t)
        if not part.ok or part.verdict is Verdict.COERCION:
            return _no(src, dst, f"field {name!r} needs an explicit conversion")
        verdict = max(verdict, part.verdict)
    return Compat(verdict)


def _static_classes(t: Type) -> "tuple[type, ...] | None":
    """Python classes the values of a non-Object type can have, if known."""
    if isinstance(t, ScalarT):
        return _SCALAR_CLASSES[t.name]
    if isinstance(t, ListT):
        return (list,)
    if isinstance(t, TupleT):
        return (tuple,)
    if isinstance(t, DictT):
        return (dict,)
    if isinstance(t, ArrayT):
        return (np.ndarray,)
    if isinstance(t, (EnumT, RecordT)):
        cls = _CLASSES.get(t.name)
        if cls is None:
            return None
        return (dict,) if typing.is_typeddict(cls) else (cls,)
    return None


def _related(a: type, b: type) -> bool:
    return issubclass(a, b) or issubclass(b, a)


def _check_object(src: Type, dst: ObjectT) -> Compat:
    b = _resolve_class(dst.cls)
    if not isinstance(src, ObjectT):
        # A structural type flowing into a class-typed pin: decided by the
        # Python classes its values have (a List value is a ``list``).
        if b is None:
            return _no(src, dst)
        classes = _static_classes(src)
        if classes is None:
            return _DYNAMIC
        if all(issubclass(c, b) for c in classes):
            return _EXACT
        if any(_related(c, b) for c in classes):
            return _DYNAMIC
        return _no(src, dst)
    a = _resolve_class(src.cls)
    if a is None or b is None:
        return _no(src, dst)
    if issubclass(a, b):
        return _EXACT
    if issubclass(b, a):
        return _DYNAMIC
    return _no(src, dst)


def _check_callable(src: Type, dst: CallableT) -> Compat:
    if not isinstance(src, CallableT):
        return _no(src, dst)
    result = check(src.result, dst.result)
    if not result.ok or result.verdict is Verdict.COERCION:
        return _no(src, dst, "results differ")
    verdict = result.verdict
    if dst.params is not None:
        if src.params is None:
            verdict = Verdict.DYNAMIC
        else:
            if len(src.params) != len(dst.params):
                return _no(src, dst, "different number of parameters")
            for s, d in zip(src.params, dst.params):
                part = check(d, s)  # parameters are contravariant
                if not part.ok or part.verdict is Verdict.COERCION:
                    return _no(src, dst, "parameters differ")
                verdict = max(verdict, part.verdict)
    return Compat(verdict)


# --------------------------------------------------------------------------
# Join, unification, substitution
# --------------------------------------------------------------------------

def join(a: Type, b: Type) -> Optional[Type]:
    """Smallest type both ``a`` and ``b`` widen to, or ``None`` if unrelated."""
    if a == b:
        return a
    if isinstance(a, AnyT) or isinstance(b, AnyT):
        return ANY
    if isinstance(a, ScalarT) and isinstance(b, ScalarT):
        if a.name in _TOWER and b.name in _TOWER:
            return a if _TOWER.index(a.name) > _TOWER.index(b.name) else b
        if a == NONE or b == NONE:
            return optional(b if a == NONE else a)
        return None
    if a == NONE or b == NONE:
        return optional(b if a == NONE else a)
    if isinstance(a, UnionT) or isinstance(b, UnionT):
        if isinstance(a, UnionT) and check(b, a).verdict is Verdict.EXACT:
            return a
        if isinstance(b, UnionT) and check(a, b).verdict is Verdict.EXACT:
            return b
        return None
    if isinstance(a, ArrayT) and isinstance(b, ArrayT):
        if isinstance(a.ndim, int) and isinstance(b.ndim, int) and a.ndim != b.ndim:
            return None
        ndim = a.ndim if a.ndim == b.ndim else None
        dtype = _join_dtype(a.dtype, b.dtype)
        if dtype is False:
            return None
        return ArrayT(dtype, ndim)  # type: ignore[arg-type]
    if isinstance(a, ListT) and isinstance(b, ListT):
        elem = join(a.elem, b.elem)
        return None if elem is None else ListT(elem)
    if isinstance(a, IteratorT) and isinstance(b, IteratorT):
        elem = join(a.elem, b.elem)
        return None if elem is None else IteratorT(elem)
    if isinstance(a, DictT) and isinstance(b, DictT):
        if a.key != b.key:
            return None
        value = join(a.value, b.value)
        return None if value is None else DictT(a.key, value)
    if isinstance(a, TupleT) and isinstance(b, TupleT):
        if a.variadic != b.variadic or len(a.elems) != len(b.elems):
            return None
        elems = [join(x, y) for x, y in zip(a.elems, b.elems)]
        if any(e is None for e in elems):
            return None
        return TupleT(tuple(elems), a.variadic)  # type: ignore[arg-type]
    if isinstance(a, ObjectT) and isinstance(b, ObjectT):
        if check(a, b).verdict is Verdict.EXACT:
            return b
        if check(b, a).verdict is Verdict.EXACT:
            return a
    return None


def _join_dtype(a: Any, b: Any) -> Any:
    """Joined dtype slot; ``False`` signals "unrelated"."""
    if a == b:
        return a
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    da, db = _np_dtype(a), _np_dtype(b)
    if da is None or db is None:
        return False
    try:
        return dtype_name(np.promote_types(da, db))
    except TypeError:
        return False


def _bind(bindings: dict[str, Any], name: str, new: Any) -> None:
    old = bindings.get(name)
    if old is None or old == new:
        bindings[name] = new
        return
    if isinstance(old, Type) and isinstance(new, Type):
        joined = join(old, new)
        if joined is None:
            raise UnifyError(name, old, new)
        bindings[name] = joined
        return
    if isinstance(old, str) and isinstance(new, str):
        joined_dtype = _join_dtype(old, new)
        if not joined_dtype:
            raise UnifyError(name, old, new)
        bindings[name] = joined_dtype
        return
    raise UnifyError(name, old, new)


def unify(pattern: Type, actual: Optional[Type], bindings: dict[str, Any]) -> None:
    """Bind the type variables of ``pattern`` so that it matches ``actual``.

    ``bindings`` is updated in place.  A variable seen twice is bound to the
    join of both types (``Int`` and ``Float`` give ``Float``).  Structural
    mismatches are ignored here: :func:`check` reports them afterwards.
    """
    if actual is None or isinstance(actual, (AnyT, VarT)):
        return
    if isinstance(pattern, VarT):
        _bind(bindings, pattern.name, actual)
    elif isinstance(pattern, ArrayT):
        if isinstance(actual, ArrayT):
            if isinstance(pattern.dtype, VarT) and isinstance(actual.dtype, str):
                _bind(bindings, pattern.dtype.name, actual.dtype)
            if isinstance(pattern.ndim, VarT) and isinstance(actual.ndim, int):
                _bind(bindings, pattern.ndim.name, actual.ndim)
    elif isinstance(pattern, ListT):
        if isinstance(actual, ListT):
            unify(pattern.elem, actual.elem, bindings)
        elif isinstance(actual, TupleT):
            for e in actual.elems:
                unify(pattern.elem, e, bindings)
    elif isinstance(pattern, IteratorT):
        if isinstance(actual, (IteratorT, ListT)):
            unify(pattern.elem, actual.elem, bindings)
    elif isinstance(pattern, TupleT):
        if isinstance(actual, TupleT):
            if pattern.variadic:
                for e in actual.elems:
                    unify(pattern.elems[0], e, bindings)
            elif not actual.variadic and len(actual.elems) == len(pattern.elems):
                for p, e in zip(pattern.elems, actual.elems):
                    unify(p, e, bindings)
    elif isinstance(pattern, DictT):
        if isinstance(actual, DictT):
            unify(pattern.key, actual.key, bindings)
            unify(pattern.value, actual.value, bindings)
    elif isinstance(pattern, UnionT):
        # Only the common ``Optional[pattern]`` shape is solved.
        if len(pattern.options) == 2 and NONE in pattern.options:
            inner = pattern.options[0] if pattern.options[1] == NONE else pattern.options[1]
            if actual == NONE:
                return
            if isinstance(actual, UnionT):
                rest = [o for o in actual.options if o != NONE]
                actual = union(*rest) if rest else None
            unify(inner, actual, bindings)
    elif isinstance(pattern, CallableT):
        if isinstance(actual, CallableT):
            unify(pattern.result, actual.result, bindings)
            if (pattern.params is not None and actual.params is not None
                    and len(pattern.params) == len(actual.params)):
                for p, e in zip(pattern.params, actual.params):
                    unify(p, e, bindings)


@functools.lru_cache(maxsize=8192)
def _has_vars(t: Type) -> bool:
    return bool(free_vars(t))


def substitute(t: Type, bindings: "cabc.Mapping[str, Any]") -> Type:
    """Replace bound type variables; unbound ones become unknown."""
    if not _has_vars(t):
        return t
    return _substitute(t, bindings)


def _substitute(t: Type, bindings: "cabc.Mapping[str, Any]") -> Type:
    if isinstance(t, VarT):
        bound = bindings.get(t.name)
        return bound if isinstance(bound, Type) else ANY
    if isinstance(t, ArrayT):
        dtype: Any = t.dtype
        ndim: Any = t.ndim
        if isinstance(dtype, VarT):
            bound = bindings.get(dtype.name)
            dtype = bound if isinstance(bound, str) else None
        if isinstance(ndim, VarT):
            bound = bindings.get(ndim.name)
            ndim = bound if isinstance(bound, int) and not isinstance(bound, bool) else None
        return ArrayT(dtype, ndim)
    if isinstance(t, ListT):
        return ListT(_substitute(t.elem, bindings))
    if isinstance(t, IteratorT):
        return IteratorT(_substitute(t.elem, bindings))
    if isinstance(t, TupleT):
        return TupleT(tuple(_substitute(e, bindings) for e in t.elems), t.variadic)
    if isinstance(t, DictT):
        return DictT(_substitute(t.key, bindings), _substitute(t.value, bindings))
    if isinstance(t, UnionT):
        return union(*(_substitute(o, bindings) for o in t.options))
    if isinstance(t, RecordT):
        return RecordT(t.name, tuple((n, _substitute(ft, bindings)) for n, ft in t.fields))
    if isinstance(t, CallableT):
        params = None if t.params is None else tuple(
            _substitute(p, bindings) for p in t.params)
        return CallableT(params, _substitute(t.result, bindings))
    return t


def free_vars(t: Type) -> set[str]:
    """Names of all type variables that occur in ``t``."""
    out: set[str] = set()
    if isinstance(t, VarT):
        out.add(t.name)
    elif isinstance(t, ArrayT):
        for slot in (t.dtype, t.ndim):
            if isinstance(slot, VarT):
                out.add(slot.name)
    elif isinstance(t, (ListT, IteratorT)):
        out |= free_vars(t.elem)
    elif isinstance(t, TupleT):
        for e in t.elems:
            out |= free_vars(e)
    elif isinstance(t, DictT):
        out |= free_vars(t.key) | free_vars(t.value)
    elif isinstance(t, UnionT):
        for o in t.options:
            out |= free_vars(o)
    elif isinstance(t, RecordT):
        for _, ft in t.fields:
            out |= free_vars(ft)
    elif isinstance(t, CallableT):
        out |= free_vars(t.result)
        for p in t.params or ():
            out |= free_vars(p)
    return out


# --------------------------------------------------------------------------
# Run-time side
# --------------------------------------------------------------------------

def plain_scalar(value: Any) -> Any:
    """Turn a NumPy number or boolean into the Python scalar it stands for.

    Scalars travel on wires as Python objects; only arrays keep a dtype.  This
    keeps static typing honest, because NumPy promotes ``np.float32(1) + x``
    differently from ``1.0 + x``.
    """
    if isinstance(value, (np.bool_, np.integer, np.floating, np.complexfloating)):
        return value.item()
    return value


def _union_all(values: Iterable[Any]) -> Type:
    """Exact element type of a container: the union of what it holds.

    Unlike a join, this never claims more than the values deliver, so
    ``conforms(v, type_of(v))`` always holds.  Very mixed content is ``Any``.
    """
    seen: set[Type] = set()
    for v in itertools.islice(values, CHECK_LIMIT):
        seen.add(type_of(v))
        if len(seen) > 4:
            return ANY
    return union(*seen) if seen else ANY


def _is_namedtuple_class(cls: Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, tuple) and hasattr(cls, "_fields")


def type_of(value: Any) -> Type:
    """The type of a concrete Python value."""
    if value is None:
        return NONE
    if isinstance(value, (bool, np.bool_)):
        return BOOL
    if isinstance(value, enum.Enum):
        return _enum_type(type(value))
    if isinstance(value, (int, np.integer)):
        return INT
    if isinstance(value, (float, np.floating)):
        return FLOAT
    if isinstance(value, (complex, np.complexfloating)):
        return COMPLEX
    if isinstance(value, str):
        return STR
    if isinstance(value, bytes):
        return BYTES
    if isinstance(value, PurePath):
        return PATH
    if isinstance(value, np.ndarray):
        return ArrayT(dtype_name(value.dtype), value.ndim)
    if isinstance(value, list):
        return ListT(_union_all(value))
    if isinstance(value, tuple):
        if _is_namedtuple_class(type(value)):
            return from_annotation(type(value))
        if len(value) <= 16:
            return TupleT(tuple(type_of(v) for v in value))
        return TupleT((_union_all(value),), variadic=True)
    if isinstance(value, dict):
        return DictT(_union_all(value.keys()), _union_all(value.values()))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return from_annotation(type(value))
    if isinstance(value, cabc.Iterator):
        return IteratorT(ANY)
    if callable(value) and not isinstance(value, type):
        return CallableT(None, ANY)
    return ObjectT(_class_path(type(value)))


def conforms(value: Any, t: Type) -> bool:
    """Does ``value`` fit type ``t`` without any conversion?

    Containers are inspected up to :data:`CHECK_LIMIT` elements.
    """
    if isinstance(t, (AnyT, VarT)):
        return True
    if isinstance(t, ScalarT):
        name = t.name
        if name == "None":
            return value is None
        if name == "Bool":
            return isinstance(value, (bool, np.bool_))
        if name == "Int":
            return isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, enum.Enum))
        if name == "Float":
            return isinstance(value, (float, np.floating))
        if name == "Complex":
            return isinstance(value, (complex, np.complexfloating))
        if name == "Str":
            return isinstance(value, str) and not isinstance(value, enum.Enum)
        if name == "Bytes":
            return isinstance(value, bytes)
        if name == "Path":
            return isinstance(value, PurePath)
        return False
    if isinstance(t, UnionT):
        return any(conforms(value, o) for o in t.options)
    if isinstance(t, ArrayT):
        if not isinstance(value, np.ndarray):
            return False
        if isinstance(t.dtype, str) and dtype_name(value.dtype) != t.dtype:
            return False
        if isinstance(t.ndim, int) and value.ndim != t.ndim:
            return False
        return True
    if isinstance(t, ListT):
        return isinstance(value, list) and all(
            conforms(v, t.elem) for v in itertools.islice(value, CHECK_LIMIT))
    if isinstance(t, TupleT):
        if not isinstance(value, tuple):
            return False
        if t.variadic:
            return all(conforms(v, t.elems[0])
                       for v in itertools.islice(value, CHECK_LIMIT))
        return len(value) == len(t.elems) and all(
            conforms(v, e) for v, e in zip(value, t.elems))
    if isinstance(t, DictT):
        return isinstance(value, dict) and all(
            conforms(k, t.key) and conforms(v, t.value)
            for k, v in itertools.islice(value.items(), CHECK_LIMIT))
    if isinstance(t, EnumT):
        return (isinstance(value, enum.Enum)
                and _dotted_name(type(value)) == t.name
                and value.name in t.members)
    if isinstance(t, RecordT):
        return _conforms_record(value, t)
    if isinstance(t, ObjectT):
        cls = _resolve_class(t.cls)
        if cls is not None:
            return isinstance(value, cls)
        return _class_path(type(value)) == t.cls
    if isinstance(t, CallableT):
        return callable(value)
    if isinstance(t, IteratorT):
        return isinstance(value, cabc.Iterator)
    return False


def _conforms_record(value: Any, t: RecordT) -> bool:
    if isinstance(value, dict):  # TypedDict instances are plain dicts
        cls = _CLASSES.get(t.name)
        if cls is not None and not typing.is_typeddict(cls):
            return False
        return set(value) == {n for n, _ in t.fields} and all(
            conforms(value[n], ft) for n, ft in t.fields)
    is_record = (dataclasses.is_dataclass(value) and not isinstance(value, type)) \
        or _is_namedtuple_class(type(value))
    if not is_record or _dotted_name(type(value)) != t.name:
        return False
    return all(hasattr(value, n) and conforms(getattr(value, n), ft)
               for n, ft in t.fields)


def _mismatch(value: Any, t: Type) -> TypeMismatch:
    return TypeMismatch(f"expected {format_type(t)}, got {format_type(type_of(value))}")


def accept(value: Any, t: Type) -> Any:
    """Return ``value`` as a ``t``, applying lossless coercions if needed.

    This is the run-time counterpart of :func:`check`; it resolves wires whose
    verdict was ``DYNAMIC``.  Raises :class:`TypeMismatch` if nothing fits.
    """
    if conforms(value, t):
        return value
    if isinstance(t, UnionT):
        for option in t.options:
            try:
                return accept(value, option)
            except TypeMismatch:
                continue
        raise _mismatch(value, t)
    if isinstance(t, ListT) and isinstance(value, (list, tuple)):
        return [accept(v, t.elem) for v in value]
    if isinstance(t, TupleT) and isinstance(value, tuple):
        if t.variadic:
            return tuple(accept(v, t.elems[0]) for v in value)
        if len(value) != len(t.elems):
            raise _mismatch(value, t)
        return tuple(accept(v, e) for v, e in zip(value, t.elems))
    if isinstance(t, DictT) and isinstance(value, dict):
        for k in value:
            if not conforms(k, t.key):
                raise _mismatch(value, t)
        return {k: accept(v, t.value) for k, v in value.items()}
    compat = check(type_of(value), t)
    if compat.verdict is Verdict.COERCION and compat.convert is not None:
        try:
            return compat.convert(value)
        except TypeMismatch:
            raise
        except Exception as exc:  # e.g. an integer too large for a float
            raise TypeMismatch(
                f"cannot convert {format_type(type_of(value))} to {format_type(t)}: {exc}"
            ) from exc
    raise _mismatch(value, t)


# --------------------------------------------------------------------------
# Python annotations
# --------------------------------------------------------------------------

def _enum_type(cls: "type[enum.Enum]") -> EnumT:
    return EnumT(_dotted_name(cls), tuple(m.name for m in cls))


def _record_type(cls: type, seen: frozenset[type]) -> Type:
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:
        hints = dict(getattr(cls, "__annotations__", {}))
    if dataclasses.is_dataclass(cls):
        names = [f.name for f in dataclasses.fields(cls)]
    elif _is_namedtuple_class(cls):
        names = list(cls._fields)  # type: ignore[attr-defined]
    else:
        names = list(hints)
    if not all(_IDENT.match(n) for n in names):
        return DictT(STR, ANY)
    fields = tuple(
        (n, _from_annotation(hints.get(n, Any), seen | {cls})) for n in names)
    return RecordT(_dotted_name(cls), fields)


def from_annotation(annotation: Any) -> Type:
    """Translate a Python annotation into an OpenRamen type.

    Unknown or missing annotations give ``Any``; this function never raises.
    """
    try:
        return _from_annotation(annotation, frozenset())
    except Exception:
        return ANY


_SIMPLE: dict[Any, Type] = {
    bool: BOOL, int: INT, float: FLOAT, complex: COMPLEX, str: STR,
    bytes: BYTES, type(None): NONE, None: NONE, Any: ANY, object: ANY,
    list: ListT(ANY), dict: DictT(ANY, ANY), tuple: TupleT((ANY,), True),
    np.ndarray: ArrayT(), os.PathLike: PATH,
}

_ALIAS = getattr(typing, "TypeAliasType", None)  # Python 3.12+
_LIST_ORIGINS = (list, cabc.Sequence, cabc.MutableSequence)
_DICT_ORIGINS = (dict, cabc.Mapping, cabc.MutableMapping)
_ITER_ORIGINS = (cabc.Iterator, cabc.Iterable, cabc.Generator)


def _from_annotation(ann: Any, seen: frozenset[type]) -> Type:
    if isinstance(ann, Type):
        return ann
    if isinstance(ann, str):
        try:
            return parse_type(ann)
        except TypeSyntaxError:
            return ANY
    try:
        simple = _SIMPLE.get(ann)
    except TypeError:  # unhashable annotation
        simple = None
    if simple is not None:
        return simple
    if isinstance(ann, typing.TypeVar):
        return VarT(ann.__name__)
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if _ALIAS is not None:
        if isinstance(ann, _ALIAS):
            return _from_annotation(ann.__value__, seen)
        if isinstance(origin, _ALIAS):
            value = origin.__value__
            try:
                value = value[args]
            except TypeError:
                pass
            return _from_annotation(value, seen)
    if origin is typing.Annotated:
        for meta in args[1:]:
            if isinstance(meta, Type):
                return meta
        return _from_annotation(args[0], seen)
    if origin is typing.Union or origin is _pytypes.UnionType:
        return union(*(_from_annotation(a, seen) for a in args))
    if origin is typing.Literal:
        return _union_all(args)
    if origin is np.ndarray:
        dtype = None
        if len(args) == 2:
            inner = typing.get_args(args[1])
            if inner and isinstance(inner[0], type) and issubclass(inner[0], np.generic):
                try:
                    dtype = dtype_name(inner[0])
                except TypeError:
                    dtype = None
        return ArrayT(dtype, None)
    if origin in _LIST_ORIGINS or ann in _LIST_ORIGINS:
        return ListT(_from_annotation(args[0], seen) if args else ANY)
    if origin is tuple:
        if not args:  # bare ``typing.Tuple``
            return TupleT((ANY,), variadic=True)
        if len(args) == 2 and args[1] is Ellipsis:
            return TupleT((_from_annotation(args[0], seen),), variadic=True)
        return TupleT(tuple(_from_annotation(a, seen) for a in args))
    if origin in _DICT_ORIGINS or ann in _DICT_ORIGINS:
        if len(args) == 2:
            return DictT(_from_annotation(args[0], seen), _from_annotation(args[1], seen))
        return DictT(ANY, ANY)
    if origin in _ITER_ORIGINS or ann in _ITER_ORIGINS:
        return IteratorT(_from_annotation(args[0], seen) if args else ANY)
    if origin is cabc.Callable or ann is cabc.Callable or ann is typing.Callable:
        if len(args) != 2:
            return CallableT(None, ANY)
        params, result = args
        if isinstance(params, (list, tuple)):
            return CallableT(tuple(_from_annotation(p, seen) for p in params),
                             _from_annotation(result, seen))
        return CallableT(None, _from_annotation(result, seen))
    if origin is not None and isinstance(origin, type):
        ann = origin  # e.g. a parametrised user generic: fall back to its class
    if not isinstance(ann, type):
        return ANY
    if issubclass(ann, enum.Enum):
        return _enum_type(ann)
    if issubclass(ann, np.generic):
        if issubclass(ann, np.bool_):
            return BOOL
        if issubclass(ann, np.integer):
            return INT
        if issubclass(ann, np.floating):
            return FLOAT
        if issubclass(ann, np.complexfloating):
            return COMPLEX
        return ObjectT(_class_path(ann))
    if issubclass(ann, PurePath):
        return PATH
    if ann in seen:
        return ObjectT(_class_path(ann))
    if dataclasses.is_dataclass(ann) or _is_namedtuple_class(ann) or typing.is_typeddict(ann):
        return _record_type(ann, seen)
    if issubclass(ann, np.ndarray):
        return ArrayT()
    return ObjectT(_class_path(ann))
