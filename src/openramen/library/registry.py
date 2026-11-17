"""The registry: every NodeSpec a session knows, by id."""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

from .spec import NodeSpec, SpecError, spec_from_function

__all__ = ["Registry", "UnknownSpecError", "default_registry"]


class UnknownSpecError(KeyError):
    """No NodeSpec with that id is registered."""

    def __str__(self) -> str:
        return f"unknown node spec {self.args[0]!r}"


class Registry:
    """A collection of NodeSpecs."""

    def __init__(self) -> None:
        self._specs: dict[str, NodeSpec] = {}
        self._providers: list[Any] = []

    def add_provider(self, provider: Any) -> None:
        """Attach a lazy source of specs, such as an imported module.

        A provider has a ``prefix``, lists candidate ids with ``ids()`` and
        builds one spec with ``load(id)`` (``None`` if it cannot).  Specs are
        built the first time they are asked for.
        """
        self._providers = [p for p in self._providers if p.prefix != provider.prefix]
        self._providers.append(provider)
        self._providers.sort(key=lambda p: len(p.prefix), reverse=True)

    @property
    def providers(self) -> list[Any]:
        return list(self._providers)

    def import_module(self, module: str, **options: Any) -> Any:
        """Make the functions of a Python module available as nodes.

        See :func:`openramen.library.importer.import_module`.
        """
        from .importer import import_module
        return import_module(self, module, **options)

    def register(self, spec: NodeSpec, *, replace: bool = False) -> NodeSpec:
        if not isinstance(spec, NodeSpec):
            raise SpecError(f"expected a NodeSpec, got {type(spec).__name__}")
        if spec.id in self._specs and not replace:
            raise SpecError(f"a node spec with id {spec.id!r} is already registered")
        self._specs[spec.id] = spec
        return spec

    def node(self, id: Optional[str] = None, *, replace: bool = False,
             **options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register a function as a node.

        The function itself is returned unchanged, with its spec attached as
        ``fn.spec``.  Options are those of :func:`spec_from_function`.
        """
        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            spec = self.register(spec_from_function(fn, id, **options), replace=replace)
            try:
                fn.spec = spec  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                pass  # builtins do not take attributes
            return fn
        return decorate

    def get(self, id: str) -> NodeSpec:
        spec = self.find(id)
        if spec is None:
            raise UnknownSpecError(id)
        return spec

    def find(self, id: str) -> Optional[NodeSpec]:
        spec = self._specs.get(id)
        if spec is not None or not isinstance(id, str):
            return spec
        for provider in self._providers:      # longest prefix first
            if id.startswith(provider.prefix + "."):
                spec = provider.load(id)
                if spec is not None:
                    self._specs[id] = spec
                    return spec
        return None

    def __contains__(self, id: object) -> bool:
        return isinstance(id, str) and self.find(id) is not None

    def __iter__(self) -> Iterator[NodeSpec]:
        """Every spec, building the lazily provided ones on the way."""
        for id in self.ids():
            spec = self.find(id)
            if spec is not None:
                yield spec

    def __len__(self) -> int:
        return len(self.ids())

    def ids(self) -> list[str]:
        """All known ids.  Lazily provided ones are candidates until built."""
        ids = set(self._specs)
        for provider in self._providers:
            ids.update(provider.ids())
        return sorted(ids)

    def categories(self) -> dict[str, list[NodeSpec]]:
        """Specs grouped by category, both levels sorted."""
        groups: dict[str, list[NodeSpec]] = {}
        for spec in self:
            groups.setdefault(spec.category, []).append(spec)
        return dict(sorted(groups.items()))

    def copy(self) -> "Registry":
        clone = Registry()
        clone._specs = dict(self._specs)
        clone._providers = list(self._providers)
        return clone

    def update(self, other: "Registry", *, replace: bool = False) -> None:
        for spec in list(other._specs.values()):
            self.register(spec, replace=replace)
        for provider in other._providers:
            self.add_provider(provider)


_default: Optional[Registry] = None


def default_registry() -> Registry:
    """A fresh registry holding the base palette."""
    global _default
    if _default is None:
        from . import base
        _default = Registry()
        base.register(_default)
    return _default.copy()
