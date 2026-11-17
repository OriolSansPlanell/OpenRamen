"""The hand-written example graphs load, check, run and survive a rewrite."""

import json
from pathlib import Path

import numpy as np
import pytest

import openramen as ramen
from openramen import Graph, analyze, run
from openramen.__main__ import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FILES = sorted(EXAMPLES.glob("*.json"))


def test_there_are_at_least_three_examples():
    assert len(FILES) >= 3


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_example_is_valid(path):
    graph = Graph.load(path)
    analysis = analyze(graph)
    assert analysis.problems == []
    result = run(graph, check_outputs=True)
    assert set(result.durations) == {n.id for n in graph}
    # every example is stored exactly as OpenRamen itself would write it
    assert graph.dumps() == path.read_text(encoding="utf-8")
    # and only uses nodes of the base palette
    registry = ramen.default_registry()
    assert all(node.spec in registry for node in graph)


def test_arithmetic():
    graph = Graph.load(EXAMPLES / "arithmetic.json")
    analysis = analyze(graph)
    result = run(graph)
    assert result["sum"] == 2.5 and result["product"] == 10.0
    assert result["text"] == "result = 10.0"
    assert str(analysis.out_type("sum", "out")) == "Float"
    assert analysis.wire_verdict("sum", "a") is ramen.types.Verdict.EXACT


def test_lists_and_select():
    graph = Graph.load(EXAMPLES / "lists_and_select.json")
    analysis = analyze(graph)
    result = run(graph)
    assert result["numbers"] == [1, 2, 3, 4, 5] and result["reversed"] == [5, 4, 3, 2, 1]
    assert result["label"] == "long" and result["picked"] == 5.0
    assert result["with_extra"] == [1.0, 2.0, 3.0, 4.0, 5.0, 2.5]
    assert str(analysis.out_type("with_extra", "out")) == "List[Float]"
    assert str(analysis.out_type("picked", "out")) == "Float"
    assert str(analysis.out_type("label", "out")) == "Str"
    assert analysis.wire_verdict("with_extra", "items") is ramen.types.Verdict.COERCION


def test_array_statistics():
    graph = Graph.load(EXAMPLES / "array_statistics.json")
    analysis = analyze(graph)
    result = run(graph)
    x = np.linspace(0.0, 2 * np.pi, 9)
    assert np.allclose(result["scaled"], 3 * np.sin(x))
    assert result["peak"] == pytest.approx(3.0) and abs(result["mean"]) < 1e-12
    assert result["bytes_total"] == 21 and result["shape"] == (2, 3)
    assert result["first_row"].tolist() == [1, 2, 3]
    types = {n: str(analysis.out_type(n, "out")) for n in
             ["x", "wave", "scaled", "positive", "mean", "as_bytes", "first_row", "bytes_total"]}
    assert types == {
        "x": "Array[float64, 1]", "wave": "Array[float64, 1]", "scaled": "Array[float64, 1]",
        "positive": "Array[bool, 1]", "mean": "Float", "as_bytes": "Array[uint8, 2]",
        "first_row": "Array[uint8, 1]", "bytes_total": "Int"}


class TestCommandLine:
    def test_check_ok(self, capsys):
        assert main(["check", str(EXAMPLES / "arithmetic.json")]) == 0
        assert "ok (7 nodes, 6 wires)" in capsys.readouterr().out

    def test_check_reports_problems(self, tmp_path, capsys):
        graph = Graph()
        graph.add_node("base.add", "n")
        path = tmp_path / "bad.json"
        graph.save(path)
        assert main(["check", str(path)]) == 1
        out = capsys.readouterr().out
        assert out.count("error:") == 2 and "ok" not in out

    def test_run_prints_the_ends_of_the_diagram(self, capsys):
        assert main(["run", str(EXAMPLES / "arithmetic.json")]) == 0
        assert capsys.readouterr().out == "text.out = 'result = 10.0'\n"

    def test_run_selected_nodes(self, capsys):
        path = str(EXAMPLES / "arithmetic.json")
        assert main(["run", path, "--node", "sum", "--node", "a"]) == 0
        assert capsys.readouterr().out == "sum.out = 2.5\na.out = 2\n"
        assert main(["run", path, "--node", "ghost"]) == 2
        assert "ghost" in capsys.readouterr().err

    def test_run_refuses_a_broken_graph(self, tmp_path, capsys):
        graph = Graph()
        graph.add_node("base.add", "n")
        graph.save(tmp_path / "bad.json")
        assert main(["run", str(tmp_path / "bad.json")]) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and "error:" in captured.err

    def test_run_reports_a_failing_node(self, tmp_path, capsys):
        graph = Graph()
        graph.add_node("base.divide", "n", values={"a": 1, "b": 0})
        graph.save(tmp_path / "zero.json")
        assert main(["run", str(tmp_path / "zero.json")]) == 1
        assert "node 'n' (base.divide): ZeroDivisionError" in capsys.readouterr().err

    @pytest.mark.parametrize("content", [None, "not json", "{}", json.dumps({"format": "x"})])
    def test_unreadable_files(self, tmp_path, capsys, content):
        path = tmp_path / "file.json"
        if content is not None:
            path.write_text(content)
        for command in ("check", "run"):
            assert main([command, str(path)]) == 2
            assert "cannot load" in capsys.readouterr().err

    def test_nodes_lists_the_palette(self, capsys):
        assert main(["nodes"]) == 0
        out = capsys.readouterr().out
        assert "Numeric" in out and "base.add" in out and "Add two numbers or arrays." in out

    def test_usage_errors(self, capsys):
        for argv in ([], ["bogus"], ["run"]):
            with pytest.raises(SystemExit) as err:
                main(argv)
            assert err.value.code == 2
        with pytest.raises(SystemExit) as err:
            main(["--version"])
        assert err.value.code == 0 and ramen.__version__ in capsys.readouterr().out

    def test_module_entry_point(self):
        import subprocess
        import sys
        src = Path(__file__).resolve().parent.parent / "src"
        proc = subprocess.run(
            [sys.executable, "-m", "openramen", "run", str(EXAMPLES / "arithmetic.json")],
            capture_output=True, text=True, env={"PYTHONPATH": str(src), "PATH": ""})
        assert proc.returncode == 0 and proc.stdout.strip() == "text.out = 'result = 10.0'"


def test_public_api_is_importable_and_documented():
    for name in ramen.__all__:
        assert hasattr(ramen, name), name
    g = ramen.Graph("demo")
    a = g.add_node("base.constant", config={"value": 2})
    b = g.add_node("base.add", values={"b": 0.5})
    g.connect(a, "out", b, "a")
    assert ramen.run(g)[b.id] == 2.5       # the example from the package docstring


def test_readme_code_runs(tmp_path, monkeypatch):
    """Every Python block in the README is executed, in order, in one namespace."""
    import re
    readme = (EXAMPLES.parent / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    assert len(blocks) >= 2
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, "README.md", "exec"), namespace)
    assert (tmp_path / "demo.json").exists()
    registry = namespace["registry"]
    spec = registry.get("my.scale")
    assert str(spec.input("image").type) == "Array[float64, 2]"
    assert str(spec.input("factor").type) == "Float" and spec.category == "Demo"
    g = Graph()
    g.add_node("base.constant", "img", config={"value": np.ones((2, 2))})
    g.add_node("my.scale", "n")
    g.connect("img", "out", "n", "image")
    assert run(g, registry)["n"].tolist() == [[2.0, 2.0], [2.0, 2.0]]


def test_spec_document_matches_the_code():
    """The tables in docs/spec.md name things that really exist."""
    import re
    from openramen import check as check_module
    spec_text = (EXAMPLES.parent / "docs" / "spec.md").read_text(encoding="utf-8")
    source = Path(check_module.__file__).read_text(encoding="utf-8")
    for code in re.findall(r'error\("([a-z-]+)"', source) + ["infer-failed", "cycle"]:
        if code != "cycle":      # cycles cannot be built or loaded in the first place
            assert f"`{code}`" in spec_text, code
    table = spec_text[spec_text.index("| Family |"):spec_text.index("### Connecting")]
    syntax_column = [row.split("|")[2] for row in table.splitlines() if row.startswith("| ")]
    examples = re.findall(r"`([^`]+)`", "\n".join(syntax_column))
    assert len(examples) >= 25
    for type_text in examples:
        ramen.types.parse_type(type_text)          # every type in the syntax table parses
    embedded = re.search(r"```json\n(.*?)```", spec_text, flags=re.S).group(1)
    graph = Graph.loads(embedded)
    assert analyze(graph).ok and run(graph)["sum"] == 2.5
    assert graph.dumps() == embedded
