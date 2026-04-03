from __future__ import annotations

from soict_llm_pruner_core import ESTIMATOR_REGISTRY

from ._shared import _BaseActivationElementEstimator, _BaseWeightMagnitudeEstimator


@ESTIMATOR_REGISTRY.register("element.activation")
class ActivationElementEstimator(_BaseActivationElementEstimator):
    """Adapter-backed activation importance estimator."""


@ESTIMATOR_REGISTRY.register("element.weight_magnitude")
class WeightMagnitudeElementEstimator(_BaseWeightMagnitudeEstimator):
    """Adapter-backed weight magnitude importance estimator."""


class Llama3ActivationElementEstimator(ActivationElementEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2ActivationElementEstimator(ActivationElementEstimator):
    """Backward-compatible alias for legacy code."""


class Llama3WeightMagnitudeEstimator(WeightMagnitudeElementEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2WeightMagnitudeEstimator(WeightMagnitudeElementEstimator):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "ActivationElementEstimator",
    "WeightMagnitudeElementEstimator",
    "Llama3ActivationElementEstimator",
    "Qwen2ActivationElementEstimator",
    "Llama3WeightMagnitudeEstimator",
    "Qwen2WeightMagnitudeEstimator",
]
