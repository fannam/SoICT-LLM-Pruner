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
        merger = self.adapter.get_merger(self.model)
        input_hidden_size = self.adapter.input_hidden_size(self.model, merger)
        accumulator = _ActivationAccumulator(input_hidden_size, agg, self.device)
        hook = self.adapter.get_ln_q(merger).register_forward_hook(
            lambda module, inputs, output: accumulator.update(output.detach())
        )
        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating merger input channels importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks((hook,))
        return {"merger_input_channels": accumulator.finalize().cpu()}

    def estimate_intermediate_channels(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        merger = self.adapter.get_merger(self.model)
        fc2 = self.adapter.get_projections(merger).fc2
        accumulator = _ActivationAccumulator(fc2.in_features, agg, self.device)
        hook = fc2.register_forward_hook(
            lambda module, inputs, output: accumulator.update(inputs[0].detach())
        )
        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating merger intermediate channels importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks((hook,))
        return {"merger_intermediate_channels": accumulator.finalize().cpu()}

    def estimate_output_channels(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        merger = self.adapter.get_merger(self.model)
        output_hidden_size = self.adapter.output_hidden_size(self.model, merger)
        accumulator = _ActivationAccumulator(output_hidden_size, agg, self.device)
        hook = merger.register_forward_hook(
            lambda module, inputs, output: accumulator.update(output.detach())
        )
        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating merger output channels importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks((hook,))
        return {"merger_output_channels": accumulator.finalize().cpu()}


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
        merger = self.adapter.get_merger(self.model)
        projections = self.adapter.get_projections(merger)
        input_hidden_size = self.adapter.input_hidden_size(self.model, merger)
        merge_factor = self.adapter.merge_factor(self.model, merger)
        importance = torch.zeros(input_hidden_size, device=self.device)

        ln_weight = getattr(self.adapter.get_ln_q(merger), "weight", None)
        if ln_weight is not None:
            importance.add_(self._scalar_slice_importance(ln_weight.detach().to(self.device)))

        fc1_weight = projections.fc1.weight.detach().to(self.device)
        for channel_idx in range(input_hidden_size):
            indices = [
                channel_idx + merge_idx * input_hidden_size
                for merge_idx in range(merge_factor)
            ]
            importance[channel_idx] += self._calculate_norm(
                fc1_weight[:, indices],
                agg,
                dim=(0, 1),
            )
        return {"merger_input_channels": importance.cpu()}

    def estimate_intermediate_channels(self, agg: str = "l2") -> Dict[str, torch.Tensor]:
        self.model.eval()
        merger = self.adapter.get_merger(self.model)
        projections = self.adapter.get_projections(merger)
        fc1_weight = projections.fc1.weight.detach().to(self.device)
        fc2_weight = projections.fc2.weight.detach().to(self.device)
        importance = self._calculate_norm(fc1_weight, agg, dim=1)
        importance = importance + self._calculate_norm(fc2_weight, agg, dim=0)
        if projections.fc1.bias is not None:
            importance = importance + self._scalar_slice_importance(
                projections.fc1.bias.detach().to(self.device)
            )
        return {"merger_intermediate_channels": importance.cpu()}

    def estimate_output_channels(self, agg: str = "l2") -> Dict[str, torch.Tensor]:
        self.model.eval()
        merger = self.adapter.get_merger(self.model)
        fc2 = self.adapter.get_projections(merger).fc2
        importance = self._calculate_norm(fc2.weight.detach().to(self.device), agg, dim=1)
        if fc2.bias is not None:
            importance = importance + self._scalar_slice_importance(fc2.bias.detach().to(self.device))
        return {"merger_output_channels": importance.cpu()}
