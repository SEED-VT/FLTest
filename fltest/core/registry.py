"""Name-keyed plugin registries for the four extension points of FLTest.

Frameworks, attacks, defenses, and metric listeners all register here by name so the
orchestrator can wire a run purely from a YAML config (``framework: flwr``,
``attacks: [label_flip]`` …) without importing concrete classes. This is what makes
the testbed extensible: dropping a new ``@register_attack("foo")`` makes ``foo`` usable
from config immediately.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A simple name -> object registry with a decorator-based ``register``."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            key = name.lower()
            if key in self._items:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._items[key] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        key = name.lower()
        if key not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}"
            )
        return self._items[key]

    def names(self):
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items


# Global registries. Modules populate these on import.
FRAMEWORKS: "Registry[Type]" = Registry("framework")
ATTACKS: "Registry[Type]" = Registry("attack")
DEFENSES: "Registry[Type]" = Registry("defense")
METRICS: "Registry[Type]" = Registry("metric")


def register_framework(name: str):
    return FRAMEWORKS.register(name)


def register_attack(name: str):
    return ATTACKS.register(name)


def register_defense(name: str):
    return DEFENSES.register(name)


def register_metric(name: str):
    return METRICS.register(name)
