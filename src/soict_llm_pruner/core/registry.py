from __future__ import annotations

from typing import Generic, Tuple, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple named registry used for extensibility points."""

    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, T] = {}

    def register(self, name: str, item: T | None = None):
        if item is not None:
            self._register(name=name, item=item)
            return item

        def decorator(obj: T) -> T:
            self._register(name=name, item=obj)
            return obj

        return decorator

    def _register(self, name: str, item: T) -> None:
        if name in self._entries:
            raise KeyError("Duplicate registration for {} in {}.".format(name, self.name))
        self._entries[name] = item

    def get(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(
                "Unknown {} '{}'. Available: {}.".format(
                    self.name,
                    name,
                    ", ".join(self.names()) or "<empty>",
                )
            ) from exc

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._entries))

    def items(self) -> Tuple[tuple[str, T], ...]:
        return tuple((name, self._entries[name]) for name in self.names())

    def __contains__(self, name: str) -> bool:
        return name in self._entries


ESTIMATOR_REGISTRY: Registry[type] = Registry("estimator")
PRUNER_REGISTRY: Registry[type] = Registry("pruner")
PRUNING_STRATEGY_REGISTRY: Registry[type] = Registry("pruning strategy")
