from __future__ import annotations

import warnings


def warn_estimator_alias(alias: str, canonical: str, *, stacklevel: int = 2) -> None:
    warnings.warn(
        "Estimator '{}' is deprecated; use '{}' instead.".format(alias, canonical),
        DeprecationWarning,
        stacklevel=stacklevel,
    )
