from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ...language.estimators._base import _ActivationAccumulator, _move_batch_to_device, _remove_hooks
from ..adapters import BaseMergerAdapter, resolve_model_adapter


class _BaseMergerEstimator:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        model_adapter: BaseMergerAdapter | str | None = None,
    ):
        self.adapter = resolve_model_adapter(model, model_adapter)
        self.model = model
        self.device = device


class _BaseMergerActivationEstimator(_BaseMergerEstimator):
    """Activation-based importance for Qwen2.5-VL patch merger channels."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        model_adapter: BaseMergerAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        self.model = model.to(device)

    def estimate_input_channels(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        mergers = self.adapter.get_mergers(self.model)
        input_hidden_size = self.adapter.input_hidden_size(self.model, mergers[0])
        accumulators = []
        hooks = []
        for merger in mergers:
            width = self.adapter.input_norm_width(merger) or input_hidden_size
            accumulator = _ActivationAccumulator(width, agg, self.device)
            accumulators.append((merger, accumulator))
            hooks.append(
                self.adapter.get_ln_q(merger).register_forward_hook(
                    lambda module, inputs, output, acc=accumulator: acc.update(output.detach())
                )
            )
        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating merger input channels importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)

        importance = torch.zeros(input_hidden_size, device=self.device)
        for merger, accumulator in accumulators:
            importance.add_(
                _reduce_input_channel_scores(
                    self.adapter,
                    self.model,
                    merger,
                    accumulator.finalize().to(self.device),
                )
            )
        return {"merger_input_channels": importance.cpu()}

    def estimate_intermediate_channels(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        mergers = self.adapter.get_mergers(self.model)
        accumulators = []
        hooks = []
        for merger in mergers:
            fc2 = self.adapter.get_projections(merger).fc2
            accumulator = _ActivationAccumulator(fc2.in_features, agg, self.device)
            accumulators.append(accumulator)
            hooks.append(
                fc2.register_forward_hook(
                    lambda module, inputs, output, acc=accumulator: acc.update(inputs[0].detach())
                )
            )
        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating merger intermediate channels importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)
        importance = torch.stack([accumulator.finalize().to(self.device) for accumulator in accumulators]).sum(dim=0)
        return {"merger_intermediate_channels": importance.cpu()}

    def estimate_output_channels(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        mergers = self.adapter.get_mergers(self.model)
        output_hidden_size = self.adapter.output_hidden_size(self.model, mergers[0])
        accumulators = []
        hooks = []
        for merger in mergers:
            accumulator = _ActivationAccumulator(output_hidden_size, agg, self.device)
            accumulators.append(accumulator)
            hooks.append(
                merger.register_forward_hook(
                    lambda module, inputs, output, acc=accumulator: acc.update(output.detach())
                )
            )
        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating merger output channels importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)
        importance = torch.stack([accumulator.finalize().to(self.device) for accumulator in accumulators]).sum(dim=0)
        return {"merger_output_channels": importance.cpu()}


class _BaseMergerMagnitudeEstimator(_BaseMergerEstimator):
    """Weight-magnitude importance for Qwen2.5-VL patch merger channels."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        model_adapter: BaseMergerAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)

    @staticmethod
    def _calculate_norm(
        tensor: torch.Tensor,
        agg: str,
        dim: int | Sequence[int],
    ) -> torch.Tensor:
        agg = agg.lower()
        if agg == "l1":
            return torch.norm(tensor, p=1, dim=dim)
        if agg == "l2":
            return torch.norm(tensor, p=2, dim=dim)
        raise ValueError("Unknown aggregation (norm type): {}. Choose 'l1' or 'l2'.".format(agg))

    @staticmethod
    def _scalar_slice_importance(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.abs()

    def estimate_input_channels(self, agg: str = "l2") -> Dict[str, torch.Tensor]:
        self.model.eval()
        mergers = self.adapter.get_mergers(self.model)
        input_hidden_size = self.adapter.input_hidden_size(self.model, mergers[0])
        importance = torch.zeros(input_hidden_size, device=self.device)

        for merger in mergers:
            projections = self.adapter.get_projections(merger)
            merge_factor = self.adapter.merge_factor(self.model, merger)

            ln_weight = getattr(self.adapter.get_ln_q(merger), "weight", None)
            if ln_weight is not None:
                importance.add_(
                    _reduce_input_channel_scores(
                        self.adapter,
                        self.model,
                        merger,
                        self._scalar_slice_importance(ln_weight.detach().to(self.device)),
                    )
                )

            fc1_weight = projections.fc1.weight.detach().to(self.device)
            for channel_idx in range(input_hidden_size):
                indices = self.adapter.input_channel_indices(self.model, merger, channel_idx)
                if len(indices) != merge_factor:
                    raise ValueError("Merger input channel index mapping must match the merge factor.")
                importance[channel_idx] += self._calculate_norm(
                    fc1_weight[:, indices],
                    agg,
                    dim=(0, 1),
                )
        return {"merger_input_channels": importance.cpu()}

    def estimate_intermediate_channels(self, agg: str = "l2") -> Dict[str, torch.Tensor]:
        self.model.eval()
        importance = None
        for merger in self.adapter.get_mergers(self.model):
            projections = self.adapter.get_projections(merger)
            fc1_weight = projections.fc1.weight.detach().to(self.device)
            fc2_weight = projections.fc2.weight.detach().to(self.device)
            merger_importance = self._calculate_norm(fc1_weight, agg, dim=1)
            merger_importance = merger_importance + self._calculate_norm(fc2_weight, agg, dim=0)
            if projections.fc1.bias is not None:
                merger_importance = merger_importance + self._scalar_slice_importance(
                    projections.fc1.bias.detach().to(self.device)
                )
            importance = merger_importance if importance is None else importance + merger_importance
        return {"merger_intermediate_channels": importance.cpu()}

    def estimate_output_channels(self, agg: str = "l2") -> Dict[str, torch.Tensor]:
        self.model.eval()
        importance = None
        for merger in self.adapter.get_mergers(self.model):
            fc2 = self.adapter.get_projections(merger).fc2
            merger_importance = self._calculate_norm(fc2.weight.detach().to(self.device), agg, dim=1)
            if fc2.bias is not None:
                merger_importance = merger_importance + self._scalar_slice_importance(
                    fc2.bias.detach().to(self.device)
                )
            importance = merger_importance if importance is None else importance + merger_importance
        return {"merger_output_channels": importance.cpu()}


def _reduce_input_channel_scores(
    adapter: BaseMergerAdapter,
    model: nn.Module,
    merger: nn.Module,
    scores: torch.Tensor,
) -> torch.Tensor:
    input_hidden_size = adapter.input_hidden_size(model, merger)
    if scores.numel() == input_hidden_size:
        return scores.reshape(input_hidden_size)

    merge_factor = adapter.merge_factor(model, merger)
    expected_width = input_hidden_size * merge_factor
    if scores.numel() != expected_width:
        raise ValueError(
            "Expected merger input scores with {} or {} elements, received {}.".format(
                input_hidden_size,
                expected_width,
                scores.numel(),
            )
        )
    return scores.reshape(merge_factor, input_hidden_size).sum(dim=0)
