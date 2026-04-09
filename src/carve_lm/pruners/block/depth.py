"""
Level-3 (block-level) depth pruners.

Remove contiguous slices of decoder blocks from the model stack,
reducing the total number of layers (depth pruning).

* ``DepthBlockPruner`` – greedily selects non-overlapping low-importance
  windows of ``block_size`` consecutive layers and deletes them.
* ``DepthLayerPruner`` – reduces the layer count by a fixed number using
  the engine-level layerwise plan (prefix-keep strategy).
"""
from __future__ import annotations

from ...core import PRUNER_REGISTRY
from .._compat import warn_pruner_alias
from .._base import _BaseBlockPruner
from .._engine.facade import DepthLayerPruner as _EngineDepthLayerPruner


@PRUNER_REGISTRY.register("depth.block", aliases=("block",))
class DepthBlockPruner(_BaseBlockPruner):
    """
    Adapter-backed depth-block pruner.

    Selects non-overlapping contiguous blocks of decoder layers with the
    lowest importance scores and removes them entirely from the model.
    """


@PRUNER_REGISTRY.register("depth.layer")
class DepthLayerPruner(_EngineDepthLayerPruner):
    """
    Adapter-backed depth-layer pruner.

    Reduces the total number of hidden layers using the engine-level
    layerwise plan (keeps a prefix of the most important layers).
    """


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

class BlockPruner(DepthBlockPruner):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_pruner_alias("BlockPruner", "DepthBlockPruner", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3BlockPruner(BlockPruner):
    """Backward-compatible alias for legacy code."""


class Qwen2BlockPruner(BlockPruner):
    """Backward-compatible alias for legacy code."""


class MistralBlockPruner(BlockPruner):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "DepthBlockPruner",
    "DepthLayerPruner",
    "BlockPruner",
    "Llama3BlockPruner",
    "Qwen2BlockPruner",
    "MistralBlockPruner",
]
