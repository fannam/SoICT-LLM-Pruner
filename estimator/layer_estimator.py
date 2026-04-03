from __future__ import annotations

from soict_llm_pruner_core import ESTIMATOR_REGISTRY

from ._shared import _BaseSimilarityLayerEstimator


@ESTIMATOR_REGISTRY.register("layer.similarity")
class SimilarityLayerEstimator(_BaseSimilarityLayerEstimator):
    """Adapter-backed layer similarity estimator."""


class Llama3SimilarityLayerEstimator(SimilarityLayerEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2SimilarityLayerEstimator(SimilarityLayerEstimator):
    """Backward-compatible alias for legacy code."""


class MistralSimilarityLayerEstimator(SimilarityLayerEstimator):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "SimilarityLayerEstimator",
    "Llama3SimilarityLayerEstimator",
    "Qwen2SimilarityLayerEstimator",
    "MistralSimilarityLayerEstimator",
]
