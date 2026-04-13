from __future__ import annotations

import warnings

from ..core import PRUNER_REGISTRY


def create_pruner(name: str, *args, **kwargs):
    if PRUNER_REGISTRY.is_alias(name):
        warnings.warn(
            "Pruner '{}' is deprecated; use '{}' instead.".format(
                name,
                PRUNER_REGISTRY.canonical_name(name),
            ),
            DeprecationWarning,
            stacklevel=2,
        )
    pruner_cls = PRUNER_REGISTRY.get(name)
    return pruner_cls(*args, **kwargs)


def available_pruners(
    prefix: str | None = None,
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    names = PRUNER_REGISTRY.names(include_aliases=include_aliases)
    if prefix is None:
        return names
    return tuple(name for name in names if name.startswith(prefix))


__all__ = ["available_pruners", "create_pruner"]
