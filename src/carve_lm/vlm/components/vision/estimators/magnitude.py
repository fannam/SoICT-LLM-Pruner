from __future__ import annotations

from ..core import ESTIMATOR_REGISTRY
from ._base import _BaseVisionMagnitudeEstimator


@ESTIMATOR_REGISTRY.register("magnitude.element")
class MagnitudeEstimator(_BaseVisionMagnitudeEstimator):
    """Magnitude estimator for Qwen2.5-VL vision transformer blocks."""


__all__ = ["MagnitudeEstimator"]
