from __future__ import annotations

from soict_llm_pruner_core import ESTIMATOR_REGISTRY

from .block_estimator import (
    BlockPerplexityEstimator,
    Llama3BlockPerplexityEstimator,
    Llama3SimilarityBlockEstimator,
    MistralBlockPerplexityEstimator,
    MistralSimilarityBlockEstimator,
    Qwen2BlockPerplexityEstimator,
    Qwen2SimilarityBlockEstimator,
    SimilarityBlockEstimator,
)
from .element_estimator import (
    ActivationElementEstimator,
    Llama3ActivationElementEstimator,
    Llama3WeightMagnitudeEstimator,
    MistralActivationElementEstimator,
    MistralWeightMagnitudeEstimator,
    Qwen2ActivationElementEstimator,
    Qwen2WeightMagnitudeEstimator,
    WeightMagnitudeElementEstimator,
)
from .layer_estimator import (
    Llama3SimilarityLayerEstimator,
    MistralSimilarityLayerEstimator,
    Qwen2SimilarityLayerEstimator,
    SimilarityLayerEstimator,
)


def create_estimator(name: str, *args, **kwargs):
    estimator_cls = ESTIMATOR_REGISTRY.get(name)
    return estimator_cls(*args, **kwargs)


def available_estimators(prefix: str | None = None) -> tuple[str, ...]:
    names = ESTIMATOR_REGISTRY.names()
    if prefix is None:
        return names
    return tuple(name for name in names if name.startswith(prefix))


__all__ = [
    "ActivationElementEstimator",
    "WeightMagnitudeElementEstimator",
    "SimilarityLayerEstimator",
    "SimilarityBlockEstimator",
    "BlockPerplexityEstimator",
    "Llama3ActivationElementEstimator",
    "Qwen2ActivationElementEstimator",
    "MistralActivationElementEstimator",
    "Llama3WeightMagnitudeEstimator",
    "Qwen2WeightMagnitudeEstimator",
    "MistralWeightMagnitudeEstimator",
    "Llama3SimilarityLayerEstimator",
    "Qwen2SimilarityLayerEstimator",
    "MistralSimilarityLayerEstimator",
    "Llama3SimilarityBlockEstimator",
    "Qwen2SimilarityBlockEstimator",
    "MistralSimilarityBlockEstimator",
    "Llama3BlockPerplexityEstimator",
    "Qwen2BlockPerplexityEstimator",
    "MistralBlockPerplexityEstimator",
    "available_estimators",
    "create_estimator",
]
