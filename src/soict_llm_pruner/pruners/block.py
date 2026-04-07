from __future__ import annotations

from ..core import PRUNER_REGISTRY
from ._compat import warn_pruner_alias
from ._shared import _BaseBlockPruner


@PRUNER_REGISTRY.register("depth.block", aliases=("block",))
class DepthBlockPruner(_BaseBlockPruner):
    """Adapter-backed depth-block pruner."""


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
    "BlockPruner",
    "Llama3BlockPruner",
    "Qwen2BlockPruner",
    "MistralBlockPruner",
]
