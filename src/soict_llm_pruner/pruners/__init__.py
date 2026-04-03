from __future__ import annotations

from ..core import PRUNER_REGISTRY
from .block import BlockPruner, Llama3BlockPruner, MistralBlockPruner, Qwen2BlockPruner
from .element import (
    ElementPruner,
    Llama3ElementPruner,
    MistralElementPruner,
    Qwen2ElementPruner,
    available_element_pruning_strategies,
)
from .layer import LayerPruner, Llama3LayerPruner, MistralLayerPruner, Qwen2LayerPruner
from .structured import StructuredBlockPruner, StructuredChannelPruner, StructuredLayerPruner


def create_pruner(name: str, *args, **kwargs):
    pruner_cls = PRUNER_REGISTRY.get(name)
    return pruner_cls(*args, **kwargs)


def available_pruners(prefix: str | None = None) -> tuple[str, ...]:
    names = PRUNER_REGISTRY.names()
    if prefix is None:
        return names
    return tuple(name for name in names if name.startswith(prefix))


__all__ = [
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
