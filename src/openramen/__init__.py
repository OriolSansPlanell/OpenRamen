"""OpenRamen: a LabVIEW-style visual dataflow framework on top of Python.

The public surface of the headless core::

    import openramen as ramen

    g = ramen.Graph("demo")
    a = g.add_node("base.constant", config={"value": 2})
    b = g.add_node("base.add", values={"b": 0.5})
    g.connect(a, "out", b, "a")
    print(ramen.run(g)[b.id])   # 2.5
"""

from . import types
from .check import Analysis, GraphCheckError, Problem, analyze, registry_for
from .library import (
    DEFAULT, MISSING, ImportFailure, NodeSpec, PinSpec, Registry, SpecError,
    UnknownSpecError, default_registry, import_callable, import_module, pin_outputs,
    spec_from_function,
)
from .model import CycleError, Graph, GraphError, Node, Structure, Tunnel, Wire
from .runtime import (
    Cache, Cancelled, Event, NodeError, Plan, Run, RunResult, compile_graph, run, start,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "types", "Graph", "Node", "Wire", "Structure", "Tunnel", "GraphError", "CycleError",
    "NodeSpec", "PinSpec", "Registry", "MISSING", "SpecError", "UnknownSpecError",
    "default_registry", "spec_from_function", "DEFAULT", "ImportFailure",
    "import_callable", "import_module", "pin_outputs",
    "analyze", "Analysis", "Problem", "GraphCheckError", "registry_for",
    "run", "compile_graph", "Plan", "RunResult", "Event", "NodeError", "Cancelled", "Run", "start", "Cache",
    "__version__",
]
