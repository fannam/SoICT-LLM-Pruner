from __future__ import annotations

import warnings

from ..core import ESTIMATOR_REGISTRY


def create_estimator(name: str, *args, **kwargs):
    if ESTIMATOR_REGISTRY.is_alias(name):
        warnings.warn(
            "Estimator '{}' is deprecated; use '{}' instead.".format(
                name,
                ESTIMATOR_REGISTRY.canonical_name(name),
            ),
            DeprecationWarning,
            stacklevel=2,
        )
    estimator_cls = ESTIMATOR_REGISTRY.get(name)
    return estimator_cls(*args, **kwargs)


def available_estimators(
    prefix: str | None = None,
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    names = ESTIMATOR_REGISTRY.names(include_aliases=include_aliases)
    if prefix is None:
        return names
    return tuple(name for name in names if name.startswith(prefix))


__all__ = ["available_estimators", "create_estimator"]
