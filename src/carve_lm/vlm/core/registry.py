from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, Tuple, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple named registry used for extensibility points."""

    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, T] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        item: T | None = None,
        *,
        aliases: Iterable[str] = (),
    ):
        if item is not None:
            self._register(name=name, item=item, aliases=aliases)
            return item

        def decorator(obj: T) -> T:
            self._register(name=name, item=obj, aliases=aliases)
            return obj

        return decorator

    def _register(
        self,
        name: str,
        item: T,
        *,
        aliases: Iterable[str] = (),
    ) -> None:
        if name in self._entries or name in self._aliases:
            raise KeyError("Duplicate registration for {} in {}.".format(name, self.name))
        self._entries[name] = item
        for alias in aliases:
            if alias in self._entries or alias in self._aliases:
                raise KeyError("Duplicate registration for {} in {}.".format(alias, self.name))
            self._aliases[alias] = name

    def canonical_name(self, name: str) -> str:
        if name in self._entries:
            return name
        try:
            return self._aliases[name]
        except KeyError as exc:
            raise KeyError(
                "Unknown {} '{}'. Available: {}.".format(
                    self.name,
                    name,
                    ", ".join(self.names(include_aliases=True)) or "<empty>",
                )
            ) from exc

    def aliases_for(self, name: str) -> Tuple[str, ...]:
        canonical_name = self.canonical_name(name)
        return tuple(
            sorted(alias for alias, canonical in self._aliases.items() if canonical == canonical_name)
        )

    def alias_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._aliases))

    def is_alias(self, name: str) -> bool:
        return name in self._aliases

    def get(self, name: str) -> T:
        return self._entries[self.canonical_name(name)]

    def names(self, include_aliases: bool = False) -> Tuple[str, ...]:
        if not include_aliases:
            return tuple(sorted(self._entries))
        return tuple(sorted((*self._entries.keys(), *self._aliases.keys())))

    def items(self, include_aliases: bool = False) -> Tuple[tuple[str, T], ...]:
        return tuple((name, self.get(name)) for name in self.names(include_aliases=include_aliases))

    def __contains__(self, name: str) -> bool:
        return name in self._entries or name in self._aliases


ESTIMATOR_REGISTRY: Registry[type] = Registry("estimator")
PRUNER_REGISTRY: Registry[type] = Registry("pruner")
PRUNING_STRATEGY_REGISTRY: Registry[type] = Registry("pruning strategy")
