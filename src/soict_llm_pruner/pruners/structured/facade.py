from __future__ import annotations

from .._compat import warn_pruner_alias
from ..depth import DepthLayerPruner
from ..width import WidthChannelPruner, WidthGroupPruner


class StructuredBlockPruner(WidthGroupPruner):
    config_cls = WidthGroupPruner.config_cls

    def __init__(self, *args, **kwargs):
        warn_pruner_alias("StructuredBlockPruner", "WidthGroupPruner", stacklevel=3)
        super().__init__(*args, **kwargs)


class StructuredChannelPruner(WidthChannelPruner):
    config_cls = WidthChannelPruner.config_cls

    def __init__(self, *args, **kwargs):
        warn_pruner_alias("StructuredChannelPruner", "WidthChannelPruner", stacklevel=3)
        super().__init__(*args, **kwargs)


class StructuredLayerPruner(DepthLayerPruner):
    config_cls = DepthLayerPruner.config_cls

    def __init__(self, *args, **kwargs):
        warn_pruner_alias("StructuredLayerPruner", "DepthLayerPruner", stacklevel=3)
        super().__init__(*args, **kwargs)


__all__ = [
    "StructuredBlockPruner",
    "StructuredChannelPruner",
    "StructuredLayerPruner",
]
