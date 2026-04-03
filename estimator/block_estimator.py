from __future__ import annotations

from soict_llm_pruner_core import ESTIMATOR_REGISTRY

from ._shared import _BaseBlockPerplexityEstimator, _BaseSimilarityBlockEstimator


@ESTIMATOR_REGISTRY.register("block.similarity")
class SimilarityBlockEstimator(_BaseSimilarityBlockEstimator):
    """Adapter-backed block similarity estimator."""


@ESTIMATOR_REGISTRY.register("block.perplexity")
class BlockPerplexityEstimator(_BaseBlockPerplexityEstimator):
    """Adapter-backed block perplexity estimator."""


class Llama3SimilarityBlockEstimator(SimilarityBlockEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2SimilarityBlockEstimator(SimilarityBlockEstimator):
    """Backward-compatible alias for legacy code."""


class MistralSimilarityBlockEstimator(SimilarityBlockEstimator):
    """Backward-compatible alias for legacy code."""


class Llama3BlockPerplexityEstimator(BlockPerplexityEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2BlockPerplexityEstimator(BlockPerplexityEstimator):
    """Backward-compatible alias for legacy code."""


class MistralBlockPerplexityEstimator(BlockPerplexityEstimator):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "SimilarityBlockEstimator",
    "BlockPerplexityEstimator",
    "Llama3SimilarityBlockEstimator",
    "Qwen2SimilarityBlockEstimator",
    "MistralSimilarityBlockEstimator",
    "Llama3BlockPerplexityEstimator",
    "Qwen2BlockPerplexityEstimator",
    "MistralBlockPerplexityEstimator",
]
