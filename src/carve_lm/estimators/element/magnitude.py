from __future__ import annotations

import torch

from ...core import ESTIMATOR_REGISTRY
from ...pruners._engine.utils import resolve_slice_tensor
from .._compat import warn_estimator_alias
from .._base import _BaseWeightMagnitudeEstimator


@ESTIMATOR_REGISTRY.register("magnitude.element", aliases=("element.weight_magnitude",))
class MagnitudeEstimator(_BaseWeightMagnitudeEstimator):
    """Adapter-backed magnitude estimator for element-level scores."""


class _BaseContextMagnitudeEstimator:
    def __init__(self, model, device: str = "cpu", model_adapter=None):
        del model_adapter
        self.model = model
        self.device = device


@ESTIMATOR_REGISTRY.register("magnitude.group")
class MagnitudeGroupEstimator(_BaseContextMagnitudeEstimator):
    """Magnitude estimator over discovered pruning groups."""

    def estimate(
        self,
        context,
        dataloader=None,
        *,
        norm: str = "l1",
    ) -> dict[str, float]:
        del dataloader
        return _context_magnitude_scores(self.model, context, norm=norm)


@ESTIMATOR_REGISTRY.register("magnitude.channel")
class MagnitudeChannelEstimator(MagnitudeGroupEstimator):
    """Magnitude estimator specialized for channel bundles."""

    def estimate(
        self,
        context,
        dataloader=None,
        *,
        norm: str = "l1",
    ) -> dict[str, float]:
        if any(group.family != "channel" for group in context.groups):
            raise ValueError("magnitude.channel requires a channel discovery context.")
        return super().estimate(context, dataloader=dataloader, norm=norm)


class WeightMagnitudeElementEstimator(MagnitudeEstimator):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_estimator_alias(
            "WeightMagnitudeElementEstimator",
            "MagnitudeEstimator",
            stacklevel=3,
        )
        super().__init__(*args, **kwargs)


class Llama3WeightMagnitudeEstimator(WeightMagnitudeElementEstimator):
    """Backward-compatible alias for legacy code."""


class Qwen2WeightMagnitudeEstimator(WeightMagnitudeElementEstimator):
    """Backward-compatible alias for legacy code."""


class MistralWeightMagnitudeEstimator(WeightMagnitudeElementEstimator):
    """Backward-compatible alias for legacy code."""


def _context_magnitude_scores(model, context, *, norm: str) -> dict[str, float]:
    resolved_norm = norm.lower()
    if resolved_norm not in {"l1", "l2"}:
        raise ValueError("norm must be one of ['l1', 'l2'].")

    scores: dict[str, float] = {}
    for group in context.groups:
        total = 0.0
        for slice_spec in group.dependent_slices:
            tensor = resolve_slice_tensor(model, slice_spec).float().reshape(-1)
            if resolved_norm == "l1":
                total += float(tensor.abs().sum().item())
            else:
                total += float(torch.linalg.vector_norm(tensor, ord=2).item())
        scores[group.group_id] = total
    return scores


__all__ = [
    "MagnitudeEstimator",
    "MagnitudeGroupEstimator",
    "MagnitudeChannelEstimator",
    "WeightMagnitudeElementEstimator",
    "Llama3WeightMagnitudeEstimator",
    "Qwen2WeightMagnitudeEstimator",
    "MistralWeightMagnitudeEstimator",
]
