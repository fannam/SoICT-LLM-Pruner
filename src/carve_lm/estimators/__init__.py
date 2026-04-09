"""
Tri-level importance estimators for structured LLM pruning.

Levels
------
* element  (Level 1) – neurons, embedding channels, attention heads, GQA groups
* layer    (Level 2) – attention sublayer and MLP sublayer within a decoder block
* block    (Level 3) – contiguous decoder blocks
"""
from __future__ import annotations

import warnings

from ..core import ESTIMATOR_REGISTRY
from .block import (
    BlockPerplexityEstimator,
    BlockSimilarityEstimator,
    Llama3BlockPerplexityEstimator,
    Llama3SimilarityBlockEstimator,
    MistralBlockPerplexityEstimator,
    MistralSimilarityBlockEstimator,
    PerplexityEstimator,
    Qwen2BlockPerplexityEstimator,
    Qwen2SimilarityBlockEstimator,
    SimilarityBlockEstimator,
)
from .element import (
    ActivationElementEstimator,
    ActivationEstimator,
    Llama3ActivationElementEstimator,
    Llama3WeightMagnitudeEstimator,
    MagnitudeChannelEstimator,
    MagnitudeEstimator,
    MagnitudeGroupEstimator,
    MistralActivationElementEstimator,
    MistralWeightMagnitudeEstimator,
    Qwen2ActivationElementEstimator,
    Qwen2WeightMagnitudeEstimator,
    RandomGroupEstimator,
    TaylorGroupEstimator,
    WeightMagnitudeElementEstimator,
)
from .layer import (
    LayerSimilarityEstimator,
    Llama3SimilarityLayerEstimator,
    MistralSimilarityLayerEstimator,
    Qwen2SimilarityLayerEstimator,
    SimilarityLayerEstimator,
)
from ._base import _BaseEstimator


class SimilarityEstimator(_BaseEstimator):
    """Facade exposing both layer-wise and block-wise similarity scoring."""

    def estimate_layers(self, dataloader):
        estimator = LayerSimilarityEstimator(
            self.model,
            device=self.device,
            model_adapter=self.adapter,
        )
        return estimator.estimate(dataloader)

    def estimate_blocks(self, dataloader, *, block_size: int):
        from .block.similarity import BlockSimilarityEstimator as _BlockSim
        estimator = _BlockSim(
            self.model,
            block_size=block_size,
            device=self.device,
            model_adapter=self.adapter,
        )
        return estimator.estimate(dataloader)


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
    # Level 1 – element
    "ActivationEstimator",
    "MagnitudeEstimator",
    "MagnitudeGroupEstimator",
    "MagnitudeChannelEstimator",
    "TaylorGroupEstimator",
    "RandomGroupEstimator",
    # Level 2 – layer
    "LayerSimilarityEstimator",
    # Level 3 – block
    "BlockSimilarityEstimator",
    "PerplexityEstimator",
    # Facade
    "SimilarityEstimator",
    # Backward-compat aliases
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
    # Factory helpers
    "available_estimators",
    "create_estimator",
]
