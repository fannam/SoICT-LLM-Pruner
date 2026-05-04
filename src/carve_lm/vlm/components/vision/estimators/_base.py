from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ...language.core.scoring import calculate_importance
from ...language.estimators._base import (
    _ActivationAccumulator,
    _move_batch_to_device,
    _remove_hooks,
    _sequence_mean,
    _sum_cosine_distances,
)
from ..adapters import BaseVisionAdapter, resolve_model_adapter


def _as_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


class _BaseVisionEstimator:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        model_adapter: BaseVisionAdapter | str | None = None,
    ):
        self.adapter = resolve_model_adapter(model, model_adapter)
        self.model = model
        self.device = device


class _BaseVisionActivationEstimator(_BaseVisionEstimator):
    """Activation-based importance for Qwen2.5-VL vision transformer blocks."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        model_adapter: BaseVisionAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        self.model = model.to(device)

    def estimate_attention_heads(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[int, torch.Tensor]:
        self.model.eval()
        blocks = self.adapter.get_blocks(self.model)
        importance_by_block = {}
        hooks = []

        def make_hook(block_idx: int, num_heads: int, head_dim: int):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                del module, output
                context = inputs[0].detach()
                heads = context.reshape(*context.shape[:-1], num_heads, head_dim)
                norms = torch.linalg.vector_norm(heads, ord=2, dim=-1)
                importance_by_block[block_idx].update(norms)

            return hook

        for block_idx, block in enumerate(blocks):
            num_heads = self.adapter.num_attention_heads(self.model, block)
            head_dim = self.adapter.head_dim(self.model, block)
            importance_by_block[block_idx] = _ActivationAccumulator(num_heads, agg, self.device)
            hooks.append(
                self.adapter.get_attention_projections(block).proj.register_forward_hook(
                    make_hook(block_idx, num_heads, head_dim)
                )
            )

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating vision attention heads importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)

        return {
            idx: accumulator.finalize().cpu()
            for idx, accumulator in importance_by_block.items()
        }

    def estimate_mlp_neurons(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_block = {}
        hooks = []

        def make_hook(block_idx: int):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                del module, output
                importance_by_block[block_idx].update(inputs[0].detach())

            return hook

        for block_idx, block in enumerate(self.adapter.get_blocks(self.model)):
            _, down_proj = self.adapter.get_mlp_output_projection(block)
            importance_by_block[block_idx] = _ActivationAccumulator(
                down_proj.in_features,
                agg,
                self.device,
            )
            hooks.append(down_proj.register_forward_hook(make_hook(block_idx)))

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating vision MLP neuron importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)

        return {
            idx: accumulator.finalize().cpu()
            for idx, accumulator in importance_by_block.items()
        }

    def estimate_hidden_channels(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        importance_by_key: Dict[str, _ActivationAccumulator] = {}
        hooks = []

        def make_hook(key: str):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                del module, inputs
                importance_by_key[key].update(_as_tensor(output).detach())

            return hook

        for block_idx, block in enumerate(self.adapter.get_blocks(self.model)):
            hidden_size = self.adapter.hidden_size(self.model, block)
            norm1_key = "vision_block{}_norm1".format(block_idx)
            norm2_key = "vision_block{}_norm2".format(block_idx)
            importance_by_key[norm1_key] = _ActivationAccumulator(hidden_size, agg, self.device)
            importance_by_key[norm2_key] = _ActivationAccumulator(hidden_size, agg, self.device)
            hooks.append(self.adapter.get_norm1(block).register_forward_hook(make_hook(norm1_key)))
            hooks.append(self.adapter.get_norm2(block).register_forward_hook(make_hook(norm2_key)))

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating vision hidden channels importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)

        return {
            key: accumulator.finalize().cpu()
            for key, accumulator in importance_by_key.items()
        }


class _BaseVisionMagnitudeEstimator(_BaseVisionEstimator):
    """Weight-magnitude importance for fused-qkv Qwen2.5-VL vision blocks."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        model_adapter: BaseVisionAdapter | str | None = None,
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

    def _qkv_head_rows(
        self,
        *,
        hidden_size: int,
        head_idx: int,
        head_dim: int,
    ) -> list[int]:
        base_rows = range(head_idx * head_dim, (head_idx + 1) * head_dim)
        return [
            offset + row
            for offset in (0, hidden_size, 2 * hidden_size)
            for row in base_rows
        ]

    def estimate_attention_heads(self, agg: str = "l2") -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_block = {}
        with torch.no_grad():
            for block_idx, block in enumerate(self.adapter.get_blocks(self.model)):
                projections = self.adapter.get_attention_projections(block)
                hidden_size = self.adapter.hidden_size(self.model, block)
                num_heads = self.adapter.num_attention_heads(self.model, block)
                head_dim = self.adapter.head_dim(self.model, block)
                scores = []
                for head_idx in range(num_heads):
                    qkv_rows = self._qkv_head_rows(
                        hidden_size=hidden_size,
                        head_idx=head_idx,
                        head_dim=head_dim,
                    )
                    qkv_score = self._calculate_norm(
                        projections.qkv.weight.detach().to(self.device)[qkv_rows, :],
                        agg,
                        dim=(0, 1),
                    )
                    proj_score = self._calculate_norm(
                        projections.proj.weight.detach().to(self.device)[
                            :,
                            head_idx * head_dim : (head_idx + 1) * head_dim,
                        ],
                        agg,
                        dim=(0, 1),
                    )
                    if projections.qkv.bias is not None:
                        qkv_score = qkv_score + self._calculate_norm(
                            projections.qkv.bias.detach().to(self.device)[qkv_rows],
                            agg,
                            dim=0,
                        )
                    scores.append(qkv_score + proj_score)
                importance_by_block[block_idx] = torch.stack(scores).cpu()
        return importance_by_block

    def estimate_mlp_neurons(self, agg: str = "l2") -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_block = {}
        with torch.no_grad():
            for block_idx, block in enumerate(self.adapter.get_blocks(self.model)):
                mlp = self.adapter.get_mlp_projections(block)
                input_norm = None
                for _, projection in mlp.input_projections():
                    projection_norm = self._calculate_norm(
                        projection.weight.detach().to(self.device),
                        agg,
                        dim=1,
                    )
                    if projection.bias is not None:
                        projection_norm = projection_norm + self._scalar_slice_importance(
                            projection.bias.detach().to(self.device)
                        )
                    input_norm = projection_norm if input_norm is None else input_norm + projection_norm
                _, output_projection = mlp.output_projection()
                output_norm = self._calculate_norm(
                    output_projection.weight.detach().to(self.device),
                    agg,
                    dim=0,
                )
                importance_by_block[block_idx] = (input_norm + output_norm).cpu()
        return importance_by_block

    def estimate_hidden_channels(self, agg: str = "l2") -> Dict[str, torch.Tensor]:
        self.model.eval()
        importance_by_key = {}
        with torch.no_grad():
            for block_idx, block in enumerate(self.adapter.get_blocks(self.model)):
                hidden_size = self.adapter.hidden_size(self.model, block)
                importance = torch.zeros(hidden_size, device=self.device)

                for norm in (self.adapter.get_norm1(block), self.adapter.get_norm2(block)):
                    weight = getattr(norm, "weight", None)
                    if weight is not None:
                        importance.add_(self._scalar_slice_importance(weight.detach().to(self.device)))

                attention = self.adapter.get_attention_projections(block)
                importance.add_(self._calculate_norm(attention.qkv.weight.detach().to(self.device), agg, dim=0))
                importance.add_(self._calculate_norm(attention.proj.weight.detach().to(self.device), agg, dim=1))
                if attention.proj.bias is not None:
                    importance.add_(self._scalar_slice_importance(attention.proj.bias.detach().to(self.device)))

                mlp = self.adapter.get_mlp_projections(block)
                for _, projection in mlp.input_projections():
                    importance.add_(self._calculate_norm(projection.weight.detach().to(self.device), agg, dim=0))
                _, output_projection = mlp.output_projection()
                importance.add_(
                    self._calculate_norm(output_projection.weight.detach().to(self.device), agg, dim=1)
                )
                if output_projection.bias is not None:
                    importance.add_(self._scalar_slice_importance(output_projection.bias.detach().to(self.device)))

                importance_by_key["vision_block{}_hidden_channels".format(block_idx)] = importance.cpu()
        return importance_by_key


class _BaseVisionSimilarityLayerEstimator(_BaseVisionEstimator):
    """Cosine-distance importance for vision attention and MLP sublayers."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        model_adapter: BaseVisionAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        self.model = model.eval().to(device)
        self._inputs: Dict[str, torch.Tensor] = {}
        self._outputs: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _capture_input(self, key: str):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            del module, output
            self._inputs[key] = inputs[0].detach().clone()

        return hook

    def _capture_output(self, key: str):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            del module, inputs
            self._outputs[key] = _as_tensor(output).detach().clone()

        return hook

    def _register_hooks(self) -> None:
        self._inputs.clear()
        self._outputs.clear()
        self._hooks = []
        for block_idx, block in enumerate(self.adapter.get_blocks(self.model)):
            self._hooks.append(
                self.adapter.get_norm1(block).register_forward_hook(
                    self._capture_input("attn_in_{}".format(block_idx))
                )
            )
            self._hooks.append(
                self.adapter.get_attention_module(block).register_forward_hook(
                    self._capture_output("attn_out_{}".format(block_idx))
                )
            )
            self._hooks.append(
                self.adapter.get_norm2(block).register_forward_hook(
                    self._capture_input("mlp_in_{}".format(block_idx))
                )
            )
            self._hooks.append(
                self.adapter.get_mlp_module(block).register_forward_hook(
                    self._capture_output("mlp_out_{}".format(block_idx))
                )
            )

    def _remove_hooks(self) -> None:
        _remove_hooks(self._hooks)
        self._hooks = []

    @torch.no_grad()
    def estimate(self, dataloader: DataLoader) -> Dict[str, List[float]]:
        num_blocks = len(self.adapter.get_blocks(self.model))
        attention_scores: Dict[int, List[float]] = defaultdict(list)
        mlp_scores: Dict[int, List[float]] = defaultdict(list)

        self._register_hooks()
        try:
            for batch in tqdm(dataloader, desc="Estimating vision layer importance"):
                self._inputs.clear()
                self._outputs.clear()
                self.model(**_move_batch_to_device(batch, self.device))

                for block_idx in range(num_blocks):
                    attn_input = self._inputs.get("attn_in_{}".format(block_idx))
                    attn_output = self._outputs.get("attn_out_{}".format(block_idx))
                    if attn_input is not None and attn_output is not None:
                        attention_scores[block_idx].append(
                            calculate_importance(attn_input, attn_input + attn_output)
                        )

                    mlp_input = self._inputs.get("mlp_in_{}".format(block_idx))
                    mlp_output = self._outputs.get("mlp_out_{}".format(block_idx))
                    if mlp_input is not None and mlp_output is not None:
                        mlp_scores[block_idx].append(
                            calculate_importance(mlp_input, mlp_input + mlp_output)
                        )
        finally:
            self._remove_hooks()

        return {
            "attention": [_sequence_mean(attention_scores[idx]) for idx in range(num_blocks)],
            "mlp": [_sequence_mean(mlp_scores[idx]) for idx in range(num_blocks)],
        }


class _BaseVisionSimilarityBlockEstimator(_BaseVisionEstimator):
    """Cosine-distance importance for contiguous vision blocks."""

    def __init__(
        self,
        model: nn.Module,
        block_size: int,
        device: str = "cuda",
        model_adapter: BaseVisionAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.model = model.eval().to(device)
        self.block_size = block_size
        num_blocks = len(self.adapter.get_blocks(self.model))
        if block_size > num_blocks:
            raise ValueError(
                "block_size ({}) cannot be greater than num_vision_blocks ({})".format(
                    block_size,
                    num_blocks,
                )
            )
        self._attn_inputs: Dict[int, torch.Tensor] = {}
        self._mlp_inputs: Dict[int, torch.Tensor] = {}
        self._mlp_outputs: Dict[int, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _register_hooks(self) -> None:
        self._attn_inputs.clear()
        self._mlp_inputs.clear()
        self._mlp_outputs.clear()
        self._hooks = []
        for block_idx, block in enumerate(self.adapter.get_blocks(self.model)):
            self._hooks.append(
                self.adapter.get_norm1(block).register_forward_hook(
                    lambda module, inputs, output, key=block_idx: self._attn_inputs.update(
                        {key: inputs[0].detach().clone()}
                    )
                )
            )
            self._hooks.append(
                self.adapter.get_norm2(block).register_forward_hook(
                    lambda module, inputs, output, key=block_idx: self._mlp_inputs.update(
                        {key: inputs[0].detach().clone()}
                    )
                )
            )
            self._hooks.append(
                self.adapter.get_mlp_module(block).register_forward_hook(
                    lambda module, inputs, output, key=block_idx: self._mlp_outputs.update(
                        {key: _as_tensor(output).detach().clone()}
                    )
                )
            )

    def _remove_hooks(self) -> None:
        _remove_hooks(self._hooks)
        self._hooks = []

    @torch.no_grad()
    def estimate(self, dataloader: DataLoader) -> List[float]:
        num_blocks = len(self.adapter.get_blocks(self.model))
        max_start = num_blocks - self.block_size + 1
        importance_sums = [0.0 for _ in range(max_start)]
        valid_counts = [0 for _ in range(max_start)]

        self._register_hooks()
        try:
            for batch in tqdm(
                dataloader,
                desc="Estimating vision block size={}".format(self.block_size),
            ):
                self._attn_inputs.clear()
                self._mlp_inputs.clear()
                self._mlp_outputs.clear()
                self.model(**_move_batch_to_device(batch, self.device))

                for start_idx in range(max_start):
                    end_idx = start_idx + self.block_size - 1
                    block_input = self._attn_inputs.get(start_idx)
                    residual_input = self._mlp_inputs.get(end_idx)
                    residual_output = self._mlp_outputs.get(end_idx)
                    if block_input is None or residual_input is None or residual_output is None:
                        continue

                    next_hidden = residual_input + residual_output
                    distance_sum, valid_count = _sum_cosine_distances(block_input, next_hidden)
                    if valid_count == 0:
                        continue
                    importance_sums[start_idx] += distance_sum
                    valid_counts[start_idx] += valid_count
        finally:
            self._remove_hooks()

        return [
            importance_sums[start_idx] / float(valid_counts[start_idx])
            if valid_counts[start_idx]
            else 0.0
            for start_idx in range(max_start)
        ]
