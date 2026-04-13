from __future__ import annotations

from .._engine.config import DepthLayerConfig, EstimatorSpec, ImportanceConfig, WidthChannelConfig, WidthGroupConfig

BlockWiseConfig = WidthGroupConfig
ChannelWiseConfig = WidthChannelConfig
LayerWiseConfig = DepthLayerConfig

__all__ = [
    "BlockWiseConfig",
    "ChannelWiseConfig",
    "LayerWiseConfig",
    "EstimatorSpec",
    "ImportanceConfig",
]
