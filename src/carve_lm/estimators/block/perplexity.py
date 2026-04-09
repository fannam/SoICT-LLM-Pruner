from __future__ import annotations

from ...core import ESTIMATOR_REGISTRY
from .._compat import warn_estimator_alias
from .._base import _BaseBlockPerplexityEstimator


@ESTIMATOR_REGISTRY.register("perplexity.block", aliases=("block.perplexity",))
class PerplexityEstimator(_BaseBlockPerplexityEstimator):
    """
    Perplexity-delta importance estimator for contiguous decoder blocks.

    Temporarily replaces each candidate block with identity modules and
    measures how much perplexity increases.  Higher delta → block is more
    important for language modelling → keep it.
    """


class BlockPerplexityEstimator(PerplexityEstimator):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_estimator_alias("BlockPerplexityEstimator", "PerplexityEstimator", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3BlockPerplexityEstimator(BlockPerplexityEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2BlockPerplexityEstimator(BlockPerplexityEstimator):
    """Backward-compatible alias for legacy code."""


class MistralBlockPerplexityEstimator(BlockPerplexityEstimator):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "PerplexityEstimator",
    "BlockPerplexityEstimator",
    "Llama3BlockPerplexityEstimator",
    "Qwen2BlockPerplexityEstimator",
    "MistralBlockPerplexityEstimator",
]
