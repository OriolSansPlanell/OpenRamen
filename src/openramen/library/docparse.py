"""Reading docstrings: parameters, returns, signature lines and type phrases.

Three styles are understood: numpydoc (NumPy, SciPy, scikit-image), Google
and reStructuredText fields.  Nothing here raises on odd input; whatever
cannot be understood is simply left out.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .. import types as T
from .spec import DEFAULT

__all__ = ["Doc", "DocItem", "parse_docstring", "signature_from_text", "type_from_phrase",
           "ARRAY_LIKE", "ARRAY_OR_SCALAR", "SCALAR"]

SCALAR = T.union(T.BOOL, T.INT, T.FLOAT, T.COMPLEX)
#: What NumPy calls array_like: anything ``np.asarray`` accepts.
ARRAY_LIKE = T.union(SCALAR, T.ArrayT(), T.ListT(T.ANY), T.TupleT((T.ANY,), variadic=True))
#: What array functions return: an array, or a scalar for 0-d results.
ARRAY_OR_SCALAR = T.union(SCALAR, T.ArrayT())


@dataclass
class DocItem:
    """One documented parameter or return value."""

    name: str            # "" for an unnamed return value
    type: str = ""       # the type phrase as written
    doc: str = ""
    optional: bool = False


@dataclass
class Doc:
    summary: str = ""
    description: str = ""
    params: dict[str, DocItem] = field(default_factory=dict)
    returns: list[DocItem] = field(default_factory=list)
    signature: str = ""  # a leading "name(args)" line, if the docstring has one
    has_returns_section: bool = False


_SIGNATURE_LINE = re.compile(r"^\s*[A-Za-z_][\w.]*\s*(\(.*\))\s*(?:->.*)?$")
_NUMPY_SECTIONS = {
    "parameters": "params", "other parameters": "params", "returns": "returns",
    "yields": "returns", "return": "returns", "params": "params",
}
_GOOGLE_SECTIONS = {
    "args": "params", "arguments": "params", "parameters": "params", "params": "params",
    "keyword args": "params", "keyword arguments": "params", "other parameters": "params",
    "returns": "returns", "return": "returns", "yields": "returns",
}
_GOOGLE_HEADER = re.compile(r"^([A-Z][A-Za-z ]*):\s*$")
_UNDERLINE = re.compile(r"^\s*(-{3,}|={3,}|~{3,})\s*$")
_REST_FIELD = re.compile(r"^:(\w+)(?:\s+([^:]*?))?:\s*(.*)$")


def parse_docstring(text: Optional[str]) -> Doc:
    """Extract what a node needs from a docstring."""
    doc = Doc()
    if not text or not isinstance(text, str):
        return doc
    try:
        lines = inspect.cleandoc(text).splitlines()
        _parse(lines, doc)
    except Exception:  # documentation must never break an import
        pass
    return doc


def _parse(lines: list[str], doc: Doc) -> None:
    # Leading signature lines, as C functions have them.  A long one may be
    # wrapped over several lines.
    lines = _join_wrapped_signature(lines)
    while lines and _SIGNATURE_LINE.match(lines[0]):
        if not doc.signature:
            doc.signature = lines[0].strip()
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]

    sections = _split_sections(lines)
    intro = sections.pop(0)[2]
    paragraphs = _paragraphs(intro)
    intro_fields = [l for l in intro if _REST_FIELD.match(l.strip())]
    if paragraphs:
        doc.summary = " ".join(paragraphs[0])
        doc.description = "\n\n".join(" ".join(p) for p in paragraphs[1:])
    for style, kind, body in sections:
        items = _numpy_items(body) if style == "numpy" else _google_items(body, kind)
        if kind == "params":
            for item in items:
                for name in _split_names(item.name):
                    doc.params.setdefault(name, DocItem(name, item.type, item.doc, item.optional))
        else:
            doc.has_returns_section = True
            for item in items:
                if style == "numpy" and not item.type:
                    # numpydoc: a return entry without " : " is a bare type
                    item.type, item.optional = _clean_type(item.name)
                    item.name = ""
            doc.returns.extend(items)
    if intro_fields or any(_REST_FIELD.match(l.strip()) for l in lines):
        _rest_fields(lines, doc)


_SIGNATURE_START = re.compile(r"^\s*[A-Za-z_][\w.]*\s*\(")


def _join_wrapped_signature(lines: list[str]) -> list[str]:
    if not lines or not _SIGNATURE_START.match(lines[0]) or _SIGNATURE_LINE.match(lines[0]):
        return lines
    joined = lines[0].strip()
    for count, line in enumerate(lines[1:8], start=2):
        if not line.strip():
            break
        joined += " " + line.strip()
        if joined.count("(") == joined.count(")"):
            if _SIGNATURE_LINE.match(joined):
                return [joined] + lines[count:]
            break
    return lines


def _paragraphs(lines: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _REST_FIELD.match(stripped):
            break
        if stripped:
            current.append(stripped)
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def _split_sections(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    """``[(style, kind, lines)]``; the first entry is the text before any section."""
    sections: list[tuple[str, str, list[str]]] = [("", "intro", [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        title = line.strip().lower()
        if (i + 1 < len(lines) and _UNDERLINE.match(lines[i + 1]) and title
                and not line.startswith(" ")):
            sections.append(("numpy", _NUMPY_SECTIONS.get(title, "other"), []))
            i += 2
            continue
        google = _GOOGLE_HEADER.match(line)
        if google and not line.startswith(" "):
            sections.append(("google", _GOOGLE_SECTIONS.get(google.group(1).lower(), "other"), []))
            i += 1
            continue
        sections[-1][2].append(line)
        i += 1
    return [s for n, s in enumerate(sections) if n == 0 or s[1] in ("params", "returns")]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _clean_type(text: str) -> tuple[str, bool]:
    """Strip "optional" and "default ..." from a type phrase."""
    optional = bool(re.search(r"\boptional\b", text, flags=re.I))
    text = re.sub(r",?\s*\boptional\b", "", text, flags=re.I)
    text = re.sub(r",?\s*\(?\bdefault\b\s*(?:is|=|:)?\s*[^,)]*\)?", "", text, flags=re.I)
    return text.strip(" ,;."), optional


def _numpy_items(lines: list[str]) -> list[DocItem]:
    items: list[DocItem] = []
    body: list[str] = []
    for line in lines:
        if not line.strip():
            if items:
                body.append("")
            continue
        if _indent(line) == 0:
            if items:
                items[-1].doc = _join(body)
            body = []
            head = line.strip()
            if " : " in head:
                name, _, type_text = head.partition(" : ")
            elif head.endswith(" :") or head.endswith(":"):
                name, type_text = head.rstrip(": "), ""
            else:
                name, type_text = head, ""
            cleaned, optional = _clean_type(type_text)
            items.append(DocItem(name.strip(), cleaned, "", optional))
        elif items:
            body.append(line.strip())
    if items:
        items[-1].doc = _join(body)
    return items


_GOOGLE_PARAM = re.compile(r"^(\*{0,2}[A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")


def _google_items(lines: list[str], kind: str) -> list[DocItem]:
    items: list[DocItem] = []
    filled = [l for l in lines if l.strip()]
    if not filled:
        return items
    base = min(_indent(l) for l in filled)
    body: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        if _indent(line) > base and items:
            body.append(line.strip())
            continue
        if items:
            items[-1].doc = _join([items[-1].doc] + body)
        body = []
        head = line.strip()
        if kind == "params":
            match = _GOOGLE_PARAM.match(head)
            if not match:
                continue
            cleaned, optional = _clean_type(match.group(2) or "")
            items.append(DocItem(match.group(1).lstrip("*"), cleaned, match.group(3), optional))
        else:
            type_text, sep, rest = head.partition(":")
            if sep and (not re.search(r"\s", type_text.strip())
                        or type_from_phrase(type_text) is not None):
                cleaned, optional = _clean_type(type_text)
                items.append(DocItem("", cleaned, rest.strip(), optional))
            else:
                items.append(DocItem("", "", head))
    if items:
        items[-1].doc = _join([items[-1].doc] + body)
    if kind == "returns" and len(items) > 1:
        # Google style documents one return value; extra lines are prose.
        items = [DocItem("", items[0].type, _join([i.doc for i in items]), items[0].optional)]
    return items


def _rest_fields(lines: list[str], doc: Doc) -> None:
    current: Optional[tuple[str, str]] = None
    store: dict[tuple[str, str], str] = {}
    for line in lines:
        match = _REST_FIELD.match(line.strip())
        if match:
            role, arg, text = match.group(1).lower(), (match.group(2) or "").strip(), match.group(3)
            current = (role, arg)
            store[current] = text.strip()
        elif current is not None and line.strip() and line.startswith(" "):
            store[current] = (store[current] + " " + line.strip()).strip()
        elif not line.strip():
            continue
        else:
            current = None
    return_doc, return_type = "", ""
    for (role, arg), text in store.items():
        if role in ("param", "parameter", "arg", "argument", "key", "keyword"):
            parts = arg.split()
            if not parts:
                continue
            name = parts[-1].lstrip("*")
            item = doc.params.setdefault(name, DocItem(name))
            item.doc = item.doc or text
            if len(parts) > 1 and not item.type:
                item.type, item.optional = _clean_type(" ".join(parts[:-1]))
        elif role == "type" and arg:
            name = arg.lstrip("*")
            item = doc.params.setdefault(name, DocItem(name))
            item.type, item.optional = _clean_type(text)
        elif role in ("return", "returns"):
            return_doc = text
        elif role == "rtype":
            return_type = text
    if return_doc or return_type:
        cleaned, optional = _clean_type(return_type)
        doc.returns.append(DocItem("", cleaned, return_doc, optional))
        doc.has_returns_section = True


def _join(lines: list[str]) -> str:
    out: list[str] = []
    for line in lines:
        if line:
            out.append(line)
        elif out and out[-1] != "\n\n":
            out.append("\n\n")
    return re.sub(r" ?\n\n ?", "\n\n", " ".join(out)).strip()


def _split_names(text: str) -> list[str]:
    """``"x1, x2"`` documents two parameters; ``"*args"`` is named ``args``."""
    names = []
    for part in text.split(","):
        name = part.strip().lstrip("*")
        if name.isidentifier():
            names.append(name)
    return names


# --------------------------------------------------------------------------
# Signatures written as text
# --------------------------------------------------------------------------

_SAFE_NAMES: dict[str, Any] = {
    "None": None, "True": True, "False": False, "int": int, "float": float,
    "bool": bool, "str": str, "complex": complex, "object": object,
    "inf": float("inf"), "nan": float("nan"),
}


def _default_value(node: ast.expr, namespace: Mapping[str, Any]) -> Any:
    """The value of a default written in documentation, or ``DEFAULT``."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        return namespace.get(node.id, DEFAULT)
    if isinstance(node, ast.Attribute):  # e.g. np.float64: only plain attribute chains
        chain = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name) and current.id in namespace:
            obj = namespace[current.id]
            for attr in reversed(chain):
                if attr.startswith("_") or not hasattr(obj, attr):
                    return DEFAULT
                obj = getattr(obj, attr)
            return obj
    return DEFAULT


def signature_from_text(text: str, namespace: Optional[Mapping[str, Any]] = None
                        ) -> Optional[inspect.Signature]:
    """Turn ``"name(a, b=1, *, c=None)"`` into a Signature, if it is that simple.

    Old-style optional brackets (``"f(x[, out])"``) are understood; defaults
    that cannot be evaluated safely become ``DEFAULT``.  Nothing is executed.
    """
    match = _SIGNATURE_LINE.match(text or "")
    if not match:
        return None
    expanded = _expand_optional(match.group(1)[1:-1])
    if expanded is None:
        return None
    params = expanded
    try:
        tree = ast.parse(f"def _f({params}): pass")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        return None
    args = func.args
    namespace = dict(namespace or {})

    def value(node: Optional[ast.expr]) -> Any:
        if node is None:
            return inspect.Parameter.empty
        if isinstance(node, ast.Name) and node.id == "__default__":
            return DEFAULT
        return _default_value(node, namespace)

    P = inspect.Parameter
    out: list[inspect.Parameter] = []
    positional = args.posonlyargs + args.args
    defaults: list[Optional[ast.expr]] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        kind = P.POSITIONAL_ONLY if index < len(args.posonlyargs) else P.POSITIONAL_OR_KEYWORD
        out.append(P(arg.arg, kind, default=value(default)))
    if args.vararg is not None:
        out.append(P(args.vararg.arg, P.VAR_POSITIONAL))
    for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        out.append(P(arg.arg, P.KEYWORD_ONLY, default=value(kw_default)))
    if args.kwarg is not None:
        out.append(P(args.kwarg.arg, P.VAR_KEYWORD))
    try:
        return inspect.Signature(out)
    except ValueError:
        return None


def _expand_optional(text: str) -> Optional[str]:
    """Rewrite old-style optional markers: ``"a[, b[, c]]"`` becomes
    ``"a, b=__default__, c=__default__"``.

    Brackets that follow ``=`` are list literals and are left alone.  Markers
    that do not start with a comma (``"[start,] stop"``) cannot be expressed as
    a Python signature, so the answer is ``None``.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "'\"":                      # skip string literals
            end = text.find(ch, i + 1)
            if end < 0:
                return None
            out.append(text[i:end + 1])
            i = end + 1
            continue
        if ch != "[":
            out.append(ch)
            i += 1
            continue
        depth, j = 1, i + 1
        while j < len(text) and depth:
            depth += {"[": 1, "]": -1}.get(text[j], 0)
            j += 1
        if depth:
            return None
        inner = text[i + 1:j - 1]
        before = "".join(out).rstrip()
        if before.endswith(("=", "(", ",", "[")) and not inner.lstrip().startswith(","):
            if before.endswith("="):
                out.append(text[i:j])        # a list literal used as a default
                i = j
                continue
            return None
        if not inner.lstrip().startswith(","):
            return None
        expanded = _expand_optional(inner)
        if expanded is None:
            return None
        pieces = []
        for piece in _split_top(expanded, (",",)):
            if "=" not in piece and piece.strip().lstrip("*").isidentifier() \
                    and not piece.strip().startswith("*"):
                piece = f"{piece.strip()}=__default__"
            pieces.append(piece)
        out.append(", " + ", ".join(pieces) if pieces else "")
        i = j
    return "".join(out)


# --------------------------------------------------------------------------
# Type phrases
# --------------------------------------------------------------------------

_SEQUENCE = r"(?:sequence|list|tuple|iterable|array|ndarray)s?"
_WORDS: list[tuple[str, T.Type]] = [
    (r"(?:array[_\- ]like|arraylike)(?: object)?s?", ARRAY_LIKE),
    (r"(?:np\.|numpy\.)?matrix", T.ArrayT()),
    (r"bool(?:ean)?s?", T.BOOL),
    (r"int(?:eger)?s?|non-?negative int(?:eger)?s?|positive int(?:eger)?s?", T.INT),
    (r"floats?|real|double", T.FLOAT),
    (r"complex", T.COMPLEX),
    (r"str(?:ing)?s?|unicode|text", T.STR),
    (r"bytes", T.BYTES),
    (r"none|nonetype", T.NONE),
    (r"scalars?|numbers?|numeric|number-like|real number", T.union(T.INT, T.FLOAT, T.COMPLEX)),
    (r"callables?|functions?", T.CallableT(None, T.ANY)),
    (r"dict(?:ionary)?|mapping", T.DictT(T.ANY, T.ANY)),
    (r"lists?", T.ListT(T.ANY)),
    (r"tuples?", T.TupleT((T.ANY,), variadic=True)),
    (r"sequences?", T.union(T.ListT(T.ANY), T.TupleT((T.ANY,), variadic=True))),
    (r"(?:pathlib\.)?(?:pure)?path(?:[- ]like)?|os\.pathlike", T.union(T.STR, T.PATH)),
    (r"object|any|(?:data[- ])?d?type(?:[_ -]like)?|file(?:[- ]like)?(?: object)?|"
     r"generator|rng|iterable|iterator|slice|ellipsis", T.ANY),
]
_WORD_TABLE = [(re.compile(rf"^(?:{pattern})$", re.I), t) for pattern, t in _WORDS]
_OF = re.compile(rf"^({_SEQUENCE}) of (.+)$", re.I)
_SHAPE_PREFIX = re.compile(r"^\(?[\w\s,.+*-]*\)?\s*(?=(?:nd)?array|array_like)", re.I)
_DIMS = re.compile(r"\b([0-9nNmM])[- ]?[dD]\b")


def _split_top(text: str, separators: tuple[str, ...]) -> list[str]:
    """Split on separators that are not inside brackets, braces or quotes."""
    parts, depth, quote, start, i = [], 0, "", 0, 0
    while i < len(text):
        ch = text[i]
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            for sep in separators:
                if text.startswith(sep, i):
                    parts.append(text[start:i])
                    start = i + len(sep)
                    i = start - 1
                    break
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def type_from_phrase(text: str) -> Optional[T.Type]:
    """The type a documentation phrase describes, or ``None`` if unclear.

    ``"int or tuple of ints"`` gives ``Union[Int, Tuple[Int, ...]]``.  The
    answer is ``None`` whenever any part of the phrase is not understood: a
    type that is too narrow would reject correct wiring.
    """
    if not text or not isinstance(text, str):
        return None
    phrase, _ = _clean_type(text.strip())
    phrase = re.sub(r":(?:class|obj|func|py:\w+):`~?([^`]+)`", r"\1", phrase)
    phrase = phrase.replace("`", "").strip()
    # "ndarray, shape (N, M)" and "array of dtype float" describe one array
    phrase = re.sub(r",?\s*(?:(?:of|with|has)\s+)?shape\s*(?:=|is|:)?\s*\([^)]*\)", "", phrase,
                    flags=re.I)
    phrase = re.sub(r",?\s*(?:(?:of|with)\s+)?dtype\s*(?:=|is|:)?\s*[\w.]+", "", phrase,
                    flags=re.I)
    phrase = phrase.strip(" ,")
    if not phrase:
        return None
    try:
        return _phrase(phrase)
    except Exception:
        return None


def _phrase(phrase: str) -> Optional[T.Type]:
    phrase = phrase.strip().strip(".").strip()
    if not phrase:
        return None
    if phrase.startswith("{") and phrase.endswith("}"):
        return _choice_set(phrase[1:-1])
    alternatives = _split_top(phrase, (" or ", " | ", ", or ", ","))
    if len(alternatives) > 1:
        parts = [_phrase(a) for a in alternatives]
        if any(p is None for p in parts):
            return None
        return T.union(*parts)  # type: ignore[arg-type]
    try:  # already OpenRamen syntax, e.g. in a hand-written docstring
        return T.parse_type(phrase)
    except T.TypeSyntaxError:
        pass
    of = _OF.match(phrase)
    if of:
        container, inner_text = of.group(1).lower().rstrip("s"), of.group(2)
        if container in ("array", "ndarray"):
            return T.ArrayT()
        inner = _phrase(inner_text)
        if inner is None:
            return None
        if container == "list":
            return T.ListT(inner)
        if container == "tuple":
            return T.TupleT((inner,), variadic=True)
        if container == "iterable":
            return T.ANY
        return T.union(T.ListT(inner), T.TupleT((inner,), variadic=True))
    lowered = phrase.lower()
    is_array = bool(re.search(r"(?:nd)?array|array_like", lowered))
    if is_array:
        core = _SHAPE_PREFIX.sub("", lowered).strip()
        core = re.sub(r"^(?:[0-9nm][- ]?d|n-dimensional|input|output|numpy|np\.)\s+", "", core)
        core = re.sub(r"\s*(?:,\s*)?(?:(?:of|with|in)\s|\().*$", "", core).strip()
        core = re.sub(r"\s+objects?$", "", core)
        like = "like" in core
        base = ARRAY_LIKE if like else T.ArrayT()
        dims = _DIMS.search(lowered)
        if not like and dims and dims.group(1).isdigit():
            return T.ArrayT(None, int(dims.group(1)))
        if re.fullmatch(r"(?:(?:np|numpy)\.)?(?:nd)?arrays?|array[_\- ]like|arraylike", core):
            return base
        return None
    for pattern, word_type in _WORD_TABLE:
        if pattern.match(lowered):
            return word_type
    return None


def _choice_set(body: str) -> Optional[T.Type]:
    """``{'reflect', 'wrap', None}`` is a text choice, possibly with None."""
    options = _split_top(body, (",",))
    if not options:
        return None
    types = []
    for option in options:
        try:
            types.append(T.type_of(ast.literal_eval(option)))
        except (ValueError, SyntaxError):
            word = _phrase(option)    # braces sometimes list types: {int, sequence}
            if word is not None:
                types.append(word)
            elif option.isidentifier():
                types.append(T.STR)   # unquoted choices: {C, F, A}
            else:
                return None
    return T.union(*types)
