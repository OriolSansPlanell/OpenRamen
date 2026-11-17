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
        try:
            return self._specs[id]
        except KeyError:
            raise UnknownSpecError(id) from None

    def find(self, id: str) -> Optional[NodeSpec]:
        return self._specs.get(id)

    def __contains__(self, id: object) -> bool:
        return id in self._specs

    def __iter__(self) -> Iterator[NodeSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def ids(self) -> list[str]:
        return sorted(self._specs)

    def categories(self) -> dict[str, list[NodeSpec]]:
        """Specs grouped by category, both levels sorted."""
        groups: dict[str, list[NodeSpec]] = {}
        for spec in sorted(self._specs.values(), key=lambda s: s.id):
            groups.setdefault(spec.category, []).append(spec)
        return dict(sorted(groups.items()))

    def copy(self) -> "Registry":
        clone = Registry()
        clone._specs = dict(self._specs)
        return clone

    def update(self, other: "Registry", *, replace: bool = False) -> None:
        for spec in other:
            self.register(spec, replace=replace)


_default: Optional[Registry] = None


def default_registry() -> Registry:
    """A fresh registry holding the base palette."""
    global _default
    if _default is None:
        from . import base
        _default = Registry()
        base.register(_default)
    return _default.copy()
