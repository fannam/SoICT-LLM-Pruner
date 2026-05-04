from __future__ import annotations

from ..core import ESTIMATOR_REGISTRY
from ._base import _BaseMergerMagnitudeEstimator


@ESTIMATOR_REGISTRY.register("magnitude.element")
class MagnitudeEstimator(_BaseMergerMagnitudeEstimator):
    """Magnitude estimator for Qwen2.5-VL patch merger channels."""


__all__ = ["MagnitudeEstimator"]
