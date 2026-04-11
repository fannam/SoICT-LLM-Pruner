"""
Level-3 (block-level) pruners.

Remove entire decoder blocks (contiguous sequences of one or more
decoder layers) from the model, reducing the network depth.
"""
from __future__ import annotations

from .depth import (
    BlockPruner,
    DepthBlockPruner,
    DepthLayerPruner,
    Llama3BlockPruner,
    MistralBlockPruner,
    Qwen2BlockPruner,
)

__all__ = [
    "DepthBlockPruner",
    "DepthLayerPruner",
    "BlockPruner",
    "Llama3BlockPruner",
    "Qwen2BlockPruner",
    "MistralBlockPruner",
]
