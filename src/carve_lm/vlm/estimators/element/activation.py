from __future__ import annotations

from ...core import ESTIMATOR_REGISTRY
from .._base import _BaseActivationElementEstimator
from .._compat import warn_estimator_alias


@ESTIMATOR_REGISTRY.register("activation.element", aliases=("element.activation",))
class ActivationEstimator(_BaseActivationElementEstimator):
    """Adapter-backed activation estimator for element-level scores."""


class ActivationElementEstimator(ActivationEstimator):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_estimator_alias("ActivationElementEstimator", "ActivationEstimator", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3ActivationElementEstimator(ActivationElementEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2ActivationElementEstimator(ActivationElementEstimator):
    """Backward-compatible alias for legacy code."""


class MistralActivationElementEstimator(ActivationElementEstimator):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "ActivationEstimator",
    "ActivationElementEstimator",
    "Llama3ActivationElementEstimator",
    "Qwen2ActivationElementEstimator",
    "MistralActivationElementEstimator",
]
