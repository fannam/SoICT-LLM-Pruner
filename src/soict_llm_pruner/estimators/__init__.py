from __future__ import annotations

import warnings

from ..core import ESTIMATOR_REGISTRY
from .activation import (
    ActivationElementEstimator,
    ActivationEstimator,
    Llama3ActivationElementEstimator,
    MistralActivationElementEstimator,
    Qwen2ActivationElementEstimator,
)
from .block import (
    BlockPerplexityEstimator,
    Llama3BlockPerplexityEstimator,
    Llama3SimilarityBlockEstimator,
    MistralBlockPerplexityEstimator,
    MistralSimilarityBlockEstimator,
    Qwen2BlockPerplexityEstimator,
    Qwen2SimilarityBlockEstimator,
    SimilarityBlockEstimator,
)
from .layer import (
    Llama3SimilarityLayerEstimator,
    MistralSimilarityLayerEstimator,
    Qwen2SimilarityLayerEstimator,
    SimilarityLayerEstimator,
)
from .magnitude import (
    Llama3WeightMagnitudeEstimator,
    MagnitudeChannelEstimator,
    MagnitudeEstimator,
    MagnitudeGroupEstimator,
    MistralWeightMagnitudeEstimator,
    Qwen2WeightMagnitudeEstimator,
    WeightMagnitudeElementEstimator,
)
from .perplexity import PerplexityEstimator
from .random import RandomGroupEstimator
from .similarity import (
    BlockSimilarityEstimator,
    LayerSimilarityEstimator,
    SimilarityEstimator,
)
from .taylor import TaylorGroupEstimator


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


__all__ = [
    "ActivationEstimator",
    "MagnitudeEstimator",
    "SimilarityEstimator",
    "PerplexityEstimator",
    "RandomGroupEstimator",
    "MagnitudeGroupEstimator",
    "MagnitudeChannelEstimator",
    "TaylorGroupEstimator",
    "LayerSimilarityEstimator",
    "BlockSimilarityEstimator",
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
