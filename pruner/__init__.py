from __future__ import annotations

from soict_llm_pruner_core import PRUNER_REGISTRY

from .block_level_pruner import BlockPruner, Llama3BlockPruner, Qwen2BlockPruner
from .element_level_pruner import (
    ElementPruner,
    Llama3ElementPruner,
    Qwen2ElementPruner,
    available_element_pruning_strategies,
)
from .layer_level_pruner import LayerPruner, Llama3LayerPruner, Qwen2LayerPruner


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
    "Llama3LayerPruner",
    "Qwen2LayerPruner",
    "Llama3BlockPruner",
    "Qwen2BlockPruner",
    "available_element_pruning_strategies",
    "available_pruners",
    "create_pruner",
]
