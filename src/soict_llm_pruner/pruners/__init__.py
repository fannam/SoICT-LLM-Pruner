from __future__ import annotations

import warnings

from ..core import PRUNER_REGISTRY
from ._engine.config import DepthLayerConfig, EstimatorSpec, ImportanceConfig, WidthChannelConfig, WidthGroupConfig
from ._engine.types import PruningResult
from .block import BlockPruner, DepthBlockPruner, Llama3BlockPruner, MistralBlockPruner, Qwen2BlockPruner
from .component import ComponentPruner
from .depth import DepthLayerPruner
from .element import (
    ElementPruner,
    Llama3ElementPruner,
    MistralElementPruner,
    Qwen2ElementPruner,
    WidthPruner,
    available_element_pruning_strategies,
)
from .layer import LayerPruner, Llama3LayerPruner, MistralLayerPruner, Qwen2LayerPruner
from .structured import StructuredBlockPruner, StructuredChannelPruner, StructuredLayerPruner
from .width import WidthChannelPruner, WidthGroupPruner


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
    "EstimatorSpec",
    "WidthGroupConfig",
    "WidthChannelConfig",
    "DepthLayerConfig",
    "ImportanceConfig",
    "PruningResult",
    "WidthPruner",
    "WidthGroupPruner",
    "WidthChannelPruner",
    "ComponentPruner",
    "DepthBlockPruner",
    "DepthLayerPruner",
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
    "available_pruners",
    "create_pruner",
]
