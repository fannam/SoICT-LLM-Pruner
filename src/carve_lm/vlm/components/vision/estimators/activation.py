from __future__ import annotations

from ..core import ESTIMATOR_REGISTRY
from ._base import _BaseVisionActivationEstimator


@ESTIMATOR_REGISTRY.register("activation.element")
class ActivationEstimator(_BaseVisionActivationEstimator):
    """Activation estimator for Qwen2.5-VL vision transformer blocks."""


__all__ = ["ActivationEstimator"]
