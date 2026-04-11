from __future__ import annotations

import warnings


def warn_pruner_alias(alias: str, canonical: str, *, stacklevel: int = 2) -> None:
    warnings.warn(
        "Pruner '{}' is deprecated; use '{}' instead.".format(alias, canonical),
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def warn_config_alias(alias: str, canonical: str, *, stacklevel: int = 2) -> None:
    warnings.warn(
        "Config '{}' is deprecated; use '{}' instead.".format(alias, canonical),
        DeprecationWarning,
        stacklevel=stacklevel,
    )
