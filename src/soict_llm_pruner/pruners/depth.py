from __future__ import annotations

from ..core import PRUNER_REGISTRY
from ._engine.facade import DepthLayerPruner as _EngineDepthLayerPruner
from .block import BlockPruner, DepthBlockPruner, Llama3BlockPruner, MistralBlockPruner, Qwen2BlockPruner


@PRUNER_REGISTRY.register("depth.layer")
class DepthLayerPruner(_EngineDepthLayerPruner):
    """Adapter-backed depth-layer pruner."""


__all__ = [
    "DepthBlockPruner",
    "DepthLayerPruner",
    "BlockPruner",
    "Llama3BlockPruner",
    "Qwen2BlockPruner",
    "MistralBlockPruner",
]
