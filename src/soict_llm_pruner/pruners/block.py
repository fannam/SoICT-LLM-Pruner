from __future__ import annotations

from ..core import PRUNER_REGISTRY
from ._shared import _BaseBlockPruner


@PRUNER_REGISTRY.register("block")
class BlockPruner(_BaseBlockPruner):
    """Adapter-backed block pruner."""


class Llama3BlockPruner(BlockPruner):
    """Backward-compatible alias for legacy code."""


class Qwen2BlockPruner(BlockPruner):
    """Backward-compatible alias for legacy code."""


class MistralBlockPruner(BlockPruner):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "BlockPruner",
    "Llama3BlockPruner",
    "Qwen2BlockPruner",
    "MistralBlockPruner",
]
