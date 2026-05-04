from __future__ import annotations

from ..core import ESTIMATOR_REGISTRY
from ._base import _BaseVisionSimilarityBlockEstimator, _BaseVisionSimilarityLayerEstimator


@ESTIMATOR_REGISTRY.register("similarity.layer")
class LayerSimilarityEstimator(_BaseVisionSimilarityLayerEstimator):
    """Similarity estimator for attention and MLP sublayers in vision blocks."""


@ESTIMATOR_REGISTRY.register("similarity.block")
class BlockSimilarityEstimator(_BaseVisionSimilarityBlockEstimator):
    """Similarity estimator for contiguous vision-transformer blocks."""


__all__ = ["BlockSimilarityEstimator", "LayerSimilarityEstimator"]
