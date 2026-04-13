from __future__ import annotations

from ...core import ESTIMATOR_REGISTRY
from .._base import _BaseLayerPerplexityEstimator
from .._compat import warn_estimator_alias


@ESTIMATOR_REGISTRY.register("perplexity.layer", aliases=("layer.perplexity",))
class LayerPerplexityEstimator(_BaseLayerPerplexityEstimator):
    """
    Perplexity-delta importance estimator for attention and MLP sublayers.

    Temporarily replaces each attention or MLP sublayer with an identity
    module and measures the perplexity increase. Higher delta means the
    component is more important for next-token prediction.
    """


class PerplexityLayerEstimator(LayerPerplexityEstimator):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_estimator_alias("PerplexityLayerEstimator", "LayerPerplexityEstimator", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3LayerPerplexityEstimator(PerplexityLayerEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2LayerPerplexityEstimator(PerplexityLayerEstimator):
    """Backward-compatible alias for legacy code."""


class MistralLayerPerplexityEstimator(PerplexityLayerEstimator):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "LayerPerplexityEstimator",
    "PerplexityLayerEstimator",
    "Llama3LayerPerplexityEstimator",
    "Qwen2LayerPerplexityEstimator",
    "MistralLayerPerplexityEstimator",
]
