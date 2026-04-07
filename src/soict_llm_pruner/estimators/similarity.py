from __future__ import annotations

from ..core import ESTIMATOR_REGISTRY
from ._compat import warn_estimator_alias
from ._shared import _BaseEstimator, _BaseSimilarityBlockEstimator, _BaseSimilarityLayerEstimator


@ESTIMATOR_REGISTRY.register("similarity.layer", aliases=("layer.similarity",))
class LayerSimilarityEstimator(_BaseSimilarityLayerEstimator):
    """Adapter-backed layer similarity estimator."""


@ESTIMATOR_REGISTRY.register("similarity.block", aliases=("block.similarity",))
class BlockSimilarityEstimator(_BaseSimilarityBlockEstimator):
    """Adapter-backed block similarity estimator."""


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
        estimator = BlockSimilarityEstimator(
            self.model,
            block_size=block_size,
            device=self.device,
            model_adapter=self.adapter,
        )
        return estimator.estimate(dataloader)


class SimilarityLayerEstimator(LayerSimilarityEstimator):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_estimator_alias("SimilarityLayerEstimator", "LayerSimilarityEstimator", stacklevel=3)
        super().__init__(*args, **kwargs)


class SimilarityBlockEstimator(BlockSimilarityEstimator):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_estimator_alias("SimilarityBlockEstimator", "BlockSimilarityEstimator", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3SimilarityLayerEstimator(SimilarityLayerEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2SimilarityLayerEstimator(SimilarityLayerEstimator):
    """Backward-compatible alias for legacy code."""


class MistralSimilarityLayerEstimator(SimilarityLayerEstimator):
    """Backward-compatible alias for legacy code."""


class Llama3SimilarityBlockEstimator(SimilarityBlockEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2SimilarityBlockEstimator(SimilarityBlockEstimator):
    """Backward-compatible alias for legacy code."""


class MistralSimilarityBlockEstimator(SimilarityBlockEstimator):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "SimilarityEstimator",
    "LayerSimilarityEstimator",
    "BlockSimilarityEstimator",
    "SimilarityLayerEstimator",
    "SimilarityBlockEstimator",
    "Llama3SimilarityLayerEstimator",
    "Qwen2SimilarityLayerEstimator",
    "MistralSimilarityLayerEstimator",
    "Llama3SimilarityBlockEstimator",
    "Qwen2SimilarityBlockEstimator",
    "MistralSimilarityBlockEstimator",
]
