from __future__ import annotations

from soict_llm_pruner_core import PRUNER_REGISTRY

from .._shared import _BaseLayerPruner


@PRUNER_REGISTRY.register("layer")
class LayerPruner(_BaseLayerPruner):
    """Adapter-backed layer pruner."""


class Llama3LayerPruner(LayerPruner):
    """Backward-compatible alias for legacy code."""


class Qwen2LayerPruner(LayerPruner):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "LayerPruner",
    "Llama3LayerPruner",
    "Qwen2LayerPruner",
]
