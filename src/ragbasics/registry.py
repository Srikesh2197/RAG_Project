"""Name-to-factory registry, so a YAML config can pick a component by name.

    @register("chunker", "fixed")
    class FixedChunker: ...

    chunker = build("chunker", "fixed", size=512)

Swapping a component is then a one-line config change. This is the whole mechanism
behind "plugins" in most frameworks: a dictionary and a decorator.
"""

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {}


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a class or factory function under (kind, name)."""

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        names = _REGISTRY.setdefault(kind, {})
        if name in names:
            raise ValueError(f"{kind} '{name}' is already registered")
        names[name] = factory
        return factory

    return decorator


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


def build(kind: str, name: str, **params: Any) -> Any:
    """Instantiate the component registered under (kind, name) with `params`."""
    try:
        factory = _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"unknown {kind} '{name}'; available: {available(kind)}") from None
    return factory(**params)
