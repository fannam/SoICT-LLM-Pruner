"""
Tri-level structured pruners for decoder-only LLMs.

Levels
------
* element  (Level 1) – neurons, embedding channels, attention heads, GQA groups
* layer    (Level 2) – attention sublayer and MLP sublayer within a decoder block
* block    (Level 3) – contiguous decoder blocks (depth pruning)
"""
from __future__ import annotations

import warnings

from ..core import PRUNER_REGISTRY
from ._engine.config import DepthLayerConfig, EstimatorSpec, ImportanceConfig, WidthChannelConfig, WidthGroupConfig
from ._engine.types import PruningResult
from .block import (
    BlockPruner,
    DepthBlockPruner,
    DepthLayerPruner,
    Llama3BlockPruner,
    MistralBlockPruner,
    Qwen2BlockPruner,
)
from .element import (
    ElementPruner,
    Llama3ElementPruner,
    MistralElementPruner,
    Qwen2ElementPruner,
    WidthChannelPruner,
    WidthGroupPruner,
    WidthPruner,
    available_element_pruning_strategies,
)
from .layer import (
    ComponentPruner,
    LayerPruner,
    Llama3LayerPruner,
    MistralLayerPruner,
    Qwen2LayerPruner,
)
from .structured import StructuredBlockPruner, StructuredChannelPruner, StructuredLayerPruner


def create_pruner(name: str, *args, **kwargs):
    if PRUNER_REGISTRY.is_alias(name):
        warnings.warn(
            "Pruner '{}' is deprecated; use '{}' instead.".format(
                name,
                PRUNER_REGISTRY.canonical_name(name),
            ),
            DeprecationWarning,
            stacklevel=2,
        )
    pruner_cls = PRUNER_REGISTRY.get(name)
    return pruner_cls(*args, **kwargs)


def available_pruners(
    prefix: str | None = None,
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    names = PRUNER_REGISTRY.names(include_aliases=include_aliases)
    if prefix is None:
        return names
    return tuple(name for name in names if name.startswith(prefix))


__all__ = [
    # Config / result types
    "EstimatorSpec",
    "WidthGroupConfig",
    "WidthChannelConfig",
    "DepthLayerConfig",
    "ImportanceConfig",
    "PruningResult",
    # Level 1 – element
    "WidthPruner",
    "WidthGroupPruner",
    "WidthChannelPruner",
    # Level 2 – layer
    "ComponentPruner",
    # Level 3 – block
    "DepthBlockPruner",
    "DepthLayerPruner",
    # Backward-compat aliases
    "ElementPruner",
    "LayerPruner",
    "BlockPruner",
    "Llama3ElementPruner",
    "Qwen2ElementPruner",
    "MistralElementPruner",
    "Llama3LayerPruner",
    "Qwen2LayerPruner",
    "MistralLayerPruner",
    "Llama3BlockPruner",
    "Qwen2BlockPruner",
    "MistralBlockPruner",
    "StructuredBlockPruner",
    "StructuredChannelPruner",
    "StructuredLayerPruner",
    "available_element_pruning_strategies",
    # Factory helpers
    "available_pruners",
    "create_pruner",
]
