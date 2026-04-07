from __future__ import annotations

from ..core import PRUNER_REGISTRY
from ._engine.facade import WidthChannelPruner as _EngineWidthChannelPruner
from ._engine.facade import WidthGroupPruner as _EngineWidthGroupPruner
from .element import (
    ElementPruner,
    Llama3ElementPruner,
    MistralElementPruner,
    Qwen2ElementPruner,
    WidthPruner,
    available_element_pruning_strategies,
)


@PRUNER_REGISTRY.register("width.group")
class WidthGroupPruner(_EngineWidthGroupPruner):
    """Adapter-backed width-group pruner."""


@PRUNER_REGISTRY.register("width.channel")
class WidthChannelPruner(_EngineWidthChannelPruner):
    """Adapter-backed width-channel pruner."""


__all__ = [
    "WidthPruner",
    "WidthGroupPruner",
    "WidthChannelPruner",
    "ElementPruner",
    "Llama3ElementPruner",
    "Qwen2ElementPruner",
    "MistralElementPruner",
    "available_element_pruning_strategies",
]
