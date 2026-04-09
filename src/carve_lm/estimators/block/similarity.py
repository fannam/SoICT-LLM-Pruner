from __future__ import annotations

from ...core import ESTIMATOR_REGISTRY
from .._compat import warn_estimator_alias
from .._base import _BaseEstimator, _BaseSimilarityBlockEstimator


@ESTIMATOR_REGISTRY.register("similarity.block", aliases=("block.similarity",))
class BlockSimilarityEstimator(_BaseSimilarityBlockEstimator):
    """
    Cosine-distance importance estimator for contiguous decoder blocks.

    For each possible starting index, scores the block of `block_size`
    consecutive decoder layers by the cosine distance between the block
    input and the block output.  Lower score → block barely changes the
    representation → safer to prune.
    """


class SimilarityBlockEstimator(BlockSimilarityEstimator):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_estimator_alias("SimilarityBlockEstimator", "BlockSimilarityEstimator", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3SimilarityBlockEstimator(SimilarityBlockEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2SimilarityBlockEstimator(SimilarityBlockEstimator):
    """Backward-compatible alias for legacy code."""


class MistralSimilarityBlockEstimator(SimilarityBlockEstimator):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "BlockSimilarityEstimator",
    "SimilarityBlockEstimator",
    "Llama3SimilarityBlockEstimator",
    "Qwen2SimilarityBlockEstimator",
    "MistralSimilarityBlockEstimator",
]
