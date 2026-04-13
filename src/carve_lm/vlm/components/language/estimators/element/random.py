from __future__ import annotations

import torch

from ...core import ESTIMATOR_REGISTRY


@ESTIMATOR_REGISTRY.register("random.group")
class RandomGroupEstimator:
    """Random baseline estimator over discovered pruning groups."""

    def __init__(self, model, device: str = "cpu", model_adapter=None):
        del model, model_adapter
        self.device = device

    def estimate(
        self,
        context,
        dataloader=None,
        *,
        seed: int = 0,
    ) -> dict[str, float]:
        del dataloader
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        values = torch.rand(len(context.groups), generator=generator)
        return {
            group.group_id: float(values[idx].item())
            for idx, group in enumerate(context.groups)
        }


__all__ = ["RandomGroupEstimator"]
