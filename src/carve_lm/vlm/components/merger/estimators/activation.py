from __future__ import annotations

from ..core import ESTIMATOR_REGISTRY
from ._base import _BaseMergerActivationEstimator


@ESTIMATOR_REGISTRY.register("activation.element")
class ActivationEstimator(_BaseMergerActivationEstimator):
    """Activation estimator for Qwen2.5-VL patch merger channels."""


__all__ = ["ActivationEstimator"]
