"""
Level-2 (layer-level) component pruner.

Replaces individual attention or MLP sublayers with identity modules,
effectively removing their contribution without changing the number of
decoder blocks.
"""
from __future__ import annotations

from ...core import PRUNER_REGISTRY
from .._base import _BaseLayerPruner
from .._compat import warn_pruner_alias


@PRUNER_REGISTRY.register("component", aliases=("layer",))
class ComponentPruner(_BaseLayerPruner):
    """
    Adapter-backed component pruner.

    Replaces the lowest-importance attention or MLP sublayers with
    identity pass-throughs, scored by any layer-level estimator
    (e.g. ``similarity.layer``).
    """


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

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
