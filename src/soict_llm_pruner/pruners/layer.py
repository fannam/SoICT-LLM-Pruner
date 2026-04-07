from __future__ import annotations

from ..core import PRUNER_REGISTRY
from ._compat import warn_pruner_alias
from ._shared import _BaseLayerPruner


@PRUNER_REGISTRY.register("component", aliases=("layer",))
class ComponentPruner(_BaseLayerPruner):
    """Adapter-backed component pruner."""


class LayerPruner(ComponentPruner):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_pruner_alias("LayerPruner", "ComponentPruner", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3LayerPruner(LayerPruner):
    """Backward-compatible alias for legacy code."""


class Qwen2LayerPruner(LayerPruner):
    """Backward-compatible alias for legacy code."""


class MistralLayerPruner(LayerPruner):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "ComponentPruner",
    "LayerPruner",
    "Llama3LayerPruner",
    "Qwen2LayerPruner",
    "MistralLayerPruner",
]
