from __future__ import annotations

from ...core import ESTIMATOR_REGISTRY
from .._base import _BaseSimilarityLayerEstimator
from .._compat import warn_estimator_alias


@ESTIMATOR_REGISTRY.register("similarity.layer", aliases=("layer.similarity",))
class LayerSimilarityEstimator(_BaseSimilarityLayerEstimator):
    """
    Cosine-distance importance estimator for attention and MLP sublayers.

    Scores each sublayer by how much the residual stream changes when the
    sublayer is active (1 - cosine_sim between input and input+output).
    Higher score → sublayer changes the representation more → more important.
    """


class SimilarityLayerEstimator(LayerSimilarityEstimator):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_estimator_alias("SimilarityLayerEstimator", "LayerSimilarityEstimator", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3SimilarityLayerEstimator(SimilarityLayerEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2SimilarityLayerEstimator(SimilarityLayerEstimator):
    """Backward-compatible alias for legacy code."""


class MistralSimilarityLayerEstimator(SimilarityLayerEstimator):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "LayerSimilarityEstimator",
    "SimilarityLayerEstimator",
    "Llama3SimilarityLayerEstimator",
    "Qwen2SimilarityLayerEstimator",
    "MistralSimilarityLayerEstimator",
]
