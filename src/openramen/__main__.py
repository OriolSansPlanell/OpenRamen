"""Command line: ``python -m openramen {check,run,nodes,import} ...``."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from . import __version__
from .check import analyze
from .library import default_registry
from .model import Graph, GraphError
from .runtime import NodeError, run


def _load(path: str) -> Optional[Graph]:
    try:
        return Graph.load(path)
    except (OSError, GraphError) as exc:
        print(f"error: cannot load {path}: {exc}", file=sys.stderr)
        return None


def _check(args: argparse.Namespace) -> int:
    graph = _load(args.file)
    if graph is None:
        return 2
    analysis = analyze(graph)
    for problem in analysis.problems:
        print(problem)
    if analysis.ok:
        print(f"{args.file}: ok ({len(graph)} nodes, {len(graph.wires)} wires)")
        return 0
    return 1


def _run(args: argparse.Namespace) -> int:
    graph = _load(args.file)
    if graph is None:
        return 2
    analysis = analyze(graph)
    if not analysis.ok:
        for problem in analysis.errors:
            print(problem, file=sys.stderr)
        return 1
    try:
        result = run(graph, targets=args.node or None,
                     detect_mutation=args.detect_mutation)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    except NodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.node:
        shown = list(args.node)
    else:  # by default show the ends of the diagram
        shown = [n.id for n in graph if not graph.outgoing(n)]
    for node in shown:
        for pin, value in result.outputs(node).items():
            print(f"{node}.{pin} = {value!r}")
    return 0


def _nodes(args: argparse.Namespace) -> int:
    for category, specs in default_registry().categories().items():
        print(category or "(no category)")
        for spec in specs:
            print(f"  {spec.id:<28} {spec.summary}")
    return 0


def _import(args: argparse.Namespace) -> int:
    from .library import ImportFailure, Registry, import_module, report
    registry = Registry()
    try:
        library = import_module(registry, args.module, pack=not args.no_pack)
    except ImportFailure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = report(library)
    if args.list:
        for spec in registry:
            print(f"{spec.id:<36} {spec.summary}")
        return 0
    print("\n".join(summary.lines()))
    for name, reason in sorted(summary.failures.items()):
        print(f"  not imported: {name}: {reason}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="openramen", description=__doc__)
    parser.add_argument("--version", action="version", version=f"openramen {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check", help="check a graph file without running it")
    p_check.add_argument("file")
    p_check.set_defaults(func=_check)
    p_run = sub.add_parser("run", help="run a graph file and print its results")
    p_run.add_argument("file")
    p_run.add_argument("--node", action="append", metavar="ID",
                       help="only run and print this node (repeatable)")
    p_run.add_argument("--detect-mutation", action="store_true",
                       help="fail if a node changes a list or dict it did not declare it writes")
    p_run.set_defaults(func=_run)
    p_nodes = sub.add_parser("nodes", help="list the base palette")
    p_nodes.set_defaults(func=_nodes)
    p_import = sub.add_parser(
        "import", help="show what importing a Python module as nodes gives")
    p_import.add_argument("module", help="e.g. numpy or scipy.ndimage")
    p_import.add_argument("--list", action="store_true", help="list the nodes instead")
    p_import.add_argument("--no-pack", action="store_true",
                          help="ignore the overrides bundled with OpenRamen")
    p_import.set_defaults(func=_import)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
