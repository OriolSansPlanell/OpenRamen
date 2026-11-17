"""Docstring parsing, signatures written as text, and type phrases."""

import inspect

import numpy as np
import pytest

from openramen import types as T
from openramen.library import DEFAULT
from openramen.library.docparse import (
    ARRAY_LIKE, ARRAY_OR_SCALAR, SCALAR, parse_docstring, signature_from_text,
    type_from_phrase,
)

NUMPYDOC = '''
    Compute the histogram of a dataset.

    A longer explanation that
    spans two lines.

    Second paragraph.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over
        the flattened array.

        A second paragraph of the description.
    bins : int or sequence of scalars or str, optional
        Number of bins.
    range : (float, float), optional
        The range.
    x1, x2 : array_like
        Two at once.
    *args : tuple
        Extra.
    flag
        No type given.

    Other Parameters
    ----------------
    weights : array_like, optional, default: None
        Weights.

    Returns
    -------
    hist : array
        The values of the histogram.
    bin_edges : array of dtype float
        The bin edges ``(length(hist)+1)``.

    See Also
    --------
    histogramdd, bincount

    Notes
    -----
    Not : a parameter
        Should be ignored.

    Examples
    --------
    >>> np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    '''


class TestNumpydoc:
    def test_summary_and_description(self):
        doc = parse_docstring(NUMPYDOC)
        assert doc.summary == "Compute the histogram of a dataset."
        assert doc.description == "A longer explanation that spans two lines.\n\nSecond paragraph."
        assert doc.signature == ""

    def test_parameters(self):
        doc = parse_docstring(NUMPYDOC)
        assert list(doc.params) == ["a", "bins", "range", "x1", "x2", "args", "flag", "weights"]
        a = doc.params["a"]
        assert (a.type, a.optional) == ("array_like", False)
        assert a.doc == ("Input data. The histogram is computed over the flattened array."
                         "\n\nA second paragraph of the description.")
        bins = doc.params["bins"]
        assert (bins.type, bins.optional) == ("int or sequence of scalars or str", True)
        assert doc.params["x1"].type == doc.params["x2"].type == "array_like"
        assert doc.params["flag"].type == "" and doc.params["flag"].doc == "No type given."
        assert (doc.params["weights"].type, doc.params["weights"].optional) == ("array_like", True)
        assert "Not" not in doc.params

    def test_returns(self):
        doc = parse_docstring(NUMPYDOC)
        assert doc.has_returns_section
        assert [(r.name, r.type) for r in doc.returns] == [
            ("hist", "array"), ("bin_edges", "array of dtype float")]
        assert doc.returns[0].doc == "The values of the histogram."

    def test_unnamed_and_optional_returns(self):
        doc = parse_docstring("""Do.

        Returns
        -------
        ndarray
            The result.
        counts : ndarray, optional
            Only if asked for.
        """)
        assert [(r.name, r.type, r.optional) for r in doc.returns] == [
            ("", "ndarray", False), ("counts", "ndarray", True)]

    def test_yields_and_no_returns(self):
        assert parse_docstring("X.\n\nYields\n------\nint\n    A number.").returns[0].type == "int"
        doc = parse_docstring("X.\n\nParameters\n----------\na : int\n    A.")
        assert doc.returns == [] and not doc.has_returns_section

    def test_leading_signature_lines(self):
        doc = parse_docstring("zeros(shape, dtype=float, order='C', *, like=None)\n\n"
                              "Return a new array of given shape.\n\nParameters\n----------\n"
                              "shape : int or tuple of ints\n    Shape.")
        assert doc.signature == "zeros(shape, dtype=float, order='C', *, like=None)"
        assert doc.summary == "Return a new array of given shape."
        assert doc.params["shape"].type == "int or tuple of ints"
        two = parse_docstring("f(a)\nf(a, b)\n\nSummary.")
        assert two.signature == "f(a)" and two.summary == "Summary."


class TestGoogle:
    TEXT = '''Fetch rows from a table.

    More detail here.

    Args:
        table (str): Name of the table,
            continued on a second line.
        limit (int, optional): Maximum number of rows.
        verbose: Print progress.
        *args: Extra positional arguments.
        **kwargs (dict): Extra options.

    Returns:
        list of str: The rows, one text
        per row.

    Raises:
        ValueError: If the table is missing.
    '''

    def test_parameters(self):
        doc = parse_docstring(self.TEXT)
        assert doc.summary == "Fetch rows from a table." and doc.description == "More detail here."
        assert list(doc.params) == ["table", "limit", "verbose", "args", "kwargs"]
        assert doc.params["table"].type == "str"
        assert doc.params["table"].doc == "Name of the table, continued on a second line."
        assert (doc.params["limit"].type, doc.params["limit"].optional) == ("int", True)
        assert doc.params["verbose"].type == "" and doc.params["verbose"].doc == "Print progress."
        assert "ValueError" not in doc.params

    def test_returns(self):
        doc = parse_docstring(self.TEXT)
        assert len(doc.returns) == 1 and doc.returns[0].type == "list of str"
        assert doc.returns[0].doc == "The rows, one text per row."
        prose = parse_docstring("X.\n\nReturns:\n    The number of rows that were found.")
        assert prose.returns[0].type == "" and "number of rows" in prose.returns[0].doc
        dotted = parse_docstring("X.\n\nReturns:\n    np.ndarray: The image.")
        assert dotted.returns[0].type == "np.ndarray"


class TestRest:
    TEXT = '''Resize an image.

    :param image: The input
        image.
    :param int width: New width.
    :type image: numpy.ndarray
    :param height: New height.
    :type height: int, optional
    :keyword mode: Interpolation mode.
    :returns: The resized image.
    :rtype: numpy.ndarray
    :raises ValueError: On bad sizes.
    '''

    def test_fields(self):
        doc = parse_docstring(self.TEXT)
        assert doc.summary == "Resize an image."
        assert list(doc.params) == ["image", "width", "height", "mode"]
        assert doc.params["image"].type == "numpy.ndarray"
        assert doc.params["image"].doc == "The input image."
        assert doc.params["width"].type == "int"
        assert (doc.params["height"].type, doc.params["height"].optional) == ("int", True)
        assert [(r.type, r.doc) for r in doc.returns] == [("numpy.ndarray", "The resized image.")]

    def test_return_without_type(self):
        doc = parse_docstring("X.\n\n:return: Something.")
        assert doc.returns[0].type == "" and doc.returns[0].doc == "Something."


@pytest.mark.parametrize("text", [None, "", "   ", "\n\n", 42, b"bytes", "Just a summary.",
                                  "Parameters\n----------", "Args:", ":param:", "a : b",
                                  "Returns\n-------\n", "----\n----\n----", "f(", "x" * 10000])
def test_parser_never_raises(text):
    doc = parse_docstring(text)
    assert isinstance(doc.summary, str) and isinstance(doc.params, dict)


def test_parser_handles_every_numpy_docstring():
    count = 0
    for name in dir(np):
        obj = getattr(np, name)
        if callable(obj) and isinstance(getattr(obj, "__doc__", None), str):
            doc = parse_docstring(obj.__doc__)
            count += 1
            for item in list(doc.params.values()) + doc.returns:
                type_from_phrase(item.type)      # must not raise either
    assert count > 300
    doc = parse_docstring(np.histogram.__doc__)
    assert doc.summary.startswith("Compute the histogram")
    assert {"a", "bins", "range", "density", "weights"} <= set(doc.params)
    assert [r.name for r in doc.returns] == ["hist", "bin_edges"]


class TestSignatureFromText:
    def kinds(self, sig):
        return [(p.name, p.kind.name, p.default) for p in sig.parameters.values()]

    def test_plain(self):
        E = inspect.Parameter.empty
        sig = signature_from_text("zeros(shape, dtype=float, order='C', *, like=None)")
        assert self.kinds(sig) == [
            ("shape", "POSITIONAL_OR_KEYWORD", E), ("dtype", "POSITIONAL_OR_KEYWORD", float),
            ("order", "POSITIONAL_OR_KEYWORD", "C"), ("like", "KEYWORD_ONLY", None)]

    def test_all_parameter_kinds(self):
        sig = signature_from_text("f(a, b=2, /, c=[1, 2], *args, d, e=-1.5, **kw)")
        assert [(n, k) for n, k, _ in self.kinds(sig)] == [
            ("a", "POSITIONAL_ONLY"), ("b", "POSITIONAL_ONLY"), ("c", "POSITIONAL_OR_KEYWORD"),
            ("args", "VAR_POSITIONAL"), ("d", "KEYWORD_ONLY"), ("e", "KEYWORD_ONLY"),
            ("kw", "VAR_KEYWORD")]
        assert sig.parameters["c"].default == [1, 2] and sig.parameters["e"].default == -1.5

    def test_old_style_optional_brackets(self):
        sig = signature_from_text("add(x1, x2, /, out=None, *, where=True[, signature])")
        assert sig.parameters["signature"].default is DEFAULT
        assert sig.parameters["signature"].kind.name == "KEYWORD_ONLY"
        sig = signature_from_text("f(a[, b[, c]])")
        assert [p.default for p in sig.parameters.values()] == [
            inspect.Parameter.empty, DEFAULT, DEFAULT]
        assert signature_from_text("f(a[, b=3])").parameters["b"].default == 3

    def test_defaults_are_never_executed(self):
        sig = signature_from_text("f(a=np.float64, b=some_call(), c=__import__('os'), d=np._core, "
                                  "e=unknown, f=inf, g=np.pi)", {"np": np})
        d = {p.name: p.default for p in sig.parameters.values()}
        assert d["a"] is np.float64 and d["g"] == np.pi and d["f"] == float("inf")
        assert d["b"] is DEFAULT and d["c"] is DEFAULT and d["d"] is DEFAULT and d["e"] is DEFAULT

    def test_return_annotation_and_dotted_names_are_accepted(self):
        assert list(signature_from_text("numpy.dot(a, b, out=None) -> ndarray").parameters) == [
            "a", "b", "out"]
        assert list(signature_from_text("f()").parameters) == []

    @pytest.mark.parametrize("text", [
        "", None, "Just a sentence.", "f(", "f(a,, b)", "f(a b)", "f(a=)", "f(1)",
        "arange([start,] stop[, step,], dtype=None)", "f(a, a)", "f(*, )", "(a, b)",
        "f(a)[0]", "class Foo(Bar)", "f(a=1, b)",
    ])
    def test_unusable_text_gives_none(self, text):
        assert signature_from_text(text) is None


def P(text):
    return T.parse_type(text)


LIST_OR_TUPLE_OF_INT = T.union(P("List[Int]"), P("Tuple[Int, ...]"))


@pytest.mark.parametrize("phrase, expected", [
    ("int", T.INT), ("integer", T.INT), ("ints", T.INT), ("float", T.FLOAT), ("bool", T.BOOL),
    ("boolean", T.BOOL), ("str", T.STR), ("string", T.STR), ("complex", T.COMPLEX),
    ("None", T.NONE), ("bytes", T.BYTES), ("scalar", T.union(T.INT, T.FLOAT, T.COMPLEX)),
    ("callable", P("Callable")), ("function", P("Callable")), ("dict", P("Dict")),
    ("list", P("List")), ("tuple", P("Tuple")), ("object", T.ANY), ("dtype", T.ANY),
    ("data-type", T.ANY), ("data type", T.ANY), ("file-like object", T.ANY),
    ("array_like", ARRAY_LIKE), ("array-like", ARRAY_LIKE), ("array_like object", ARRAY_LIKE),
    ("ndarray", T.ArrayT()), ("array", T.ArrayT()), ("np.ndarray", T.ArrayT()),
    ("numpy.ndarray", T.ArrayT()), ("ndarrays", T.ArrayT()), ("matrix", T.ArrayT()),
    ("(N,) ndarray", T.ArrayT()), ("(M, N) array_like", ARRAY_LIKE),
    ("ndarray, shape (N, M)", T.ArrayT()), ("ndarray of shape (N,)", T.ArrayT()),
    ("array of dtype float", T.ArrayT()), ("ndarray of ints", T.ArrayT()),
    ("array_like of floats", ARRAY_LIKE), ("ndarray of dtype bool", T.ArrayT()),
    ("2-D array", T.ArrayT(None, 2)), ("1-D ndarray", T.ArrayT(None, 1)),
    ("2D array", T.ArrayT(None, 2)), ("N-D array", T.ArrayT()), ("1-D array_like", ARRAY_LIKE),
    ("input ndarray", T.ArrayT()),
    ("int, optional", T.INT), ("float, default 1.0", T.FLOAT), ("bool (default: True)", T.BOOL),
    ("str, default='reflect'", T.STR), ("int, optional, default: 0", T.INT),
    ("int or float", T.union(T.INT, T.FLOAT)), ("int or None", T.optional(T.INT)),
    ("None or int or tuple of ints", T.union(T.NONE, T.INT, P("Tuple[Int, ...]"))),
    ("int | None", T.optional(T.INT)), ("int, float, or str", T.union(T.INT, T.FLOAT, T.STR)),
    ("scalar or array_like", ARRAY_LIKE), ("ndarray or scalar", ARRAY_OR_SCALAR.__class__ and
     T.union(T.ArrayT(), T.INT, T.FLOAT, T.COMPLEX)),
    ("tuple of ints", P("Tuple[Int, ...]")), ("list of str", P("List[Str]")),
    ("sequence of ints", LIST_OR_TUPLE_OF_INT), ("int or sequence of ints",
                                                 T.union(T.INT, LIST_OR_TUPLE_OF_INT)),
    ("tuple of ndarrays", P("Tuple[Array, ...]")), ("list of arrays", P("List[Array]")),
    ("sequence of array_like", T.union(T.ListT(ARRAY_LIKE), T.TupleT((ARRAY_LIKE,), True))),
    ("{'raise', 'wrap', 'clip'}", T.STR), ("{'C', 'F', 'A'}, optional", T.STR),
    ("{C, F, A}", T.STR), ("{'valid', 'same', None}", T.optional(T.STR)),
    ("{0, 1}", T.INT), ("{True, False, 'auto'}", T.union(T.BOOL, T.STR)),
    ("str or path-like", T.union(T.STR, T.PATH)), ("pathlib.Path", T.union(T.STR, T.PATH)),
    ("str or callable", T.union(T.STR, P("Callable"))),
    ("int or dtype", T.ANY),                       # Any absorbs: nothing is rejected
    ("Array[float64, 2]", P("Array[float64, 2]")), ("List[Int] or None", P("Optional[List[Int]]")),
    (":class:`numpy.ndarray`", T.ArrayT()), ("`ndarray`", T.ArrayT()),
    ("INT", T.INT), ("  float  ", T.FLOAT), ("float.", T.FLOAT),
])
def test_type_from_phrase(phrase, expected):
    assert type_from_phrase(phrase) == expected


@pytest.mark.parametrize("phrase", [
    "", None, "   ", "optional", "see below", "MyClass", "numpy.random.Generator",
    "int or MyClass", "a thing that walks", "{", "{}", "{foo bar}", "tuple of things",
    "sequence of widgets or int", "instance of Foo", "same as input", "(float, float)",
    "ndarray subclass instance", "array of some special kind, see notes", 42,
])
def test_unclear_phrases_give_none(phrase):
    """A phrase is only trusted when every part of it is understood."""
    assert type_from_phrase(phrase) is None


def test_helper_types():
    assert T.check(T.INT, ARRAY_LIKE).verdict is T.Verdict.EXACT
    assert T.check(P("Array[uint8, 2]"), ARRAY_LIKE).verdict is T.Verdict.EXACT
    assert T.check(P("List[List[Float]]"), ARRAY_LIKE).verdict is T.Verdict.EXACT
    assert T.check(P("Tuple[Int, Int]"), ARRAY_LIKE).verdict is T.Verdict.EXACT
    assert T.check(T.STR, ARRAY_LIKE).verdict is T.Verdict.INCOMPATIBLE
    assert T.check(P("Array"), ARRAY_OR_SCALAR).verdict is T.Verdict.EXACT
    assert T.check(ARRAY_OR_SCALAR, P("Array")).verdict is T.Verdict.DYNAMIC
    assert T.check(T.BOOL, SCALAR).verdict is T.Verdict.EXACT
