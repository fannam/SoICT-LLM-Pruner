from __future__ import annotations

import gc
import math
from collections import defaultdict
from typing import Any, Dict, List, MutableMapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..adapters import BaseModelAdapter, resolve_model_adapter
from ..core.scoring import calculate_importance


def _move_batch_to_device(batch: MutableMapping[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _prepare_causal_lm_inputs(batch: MutableMapping[str, Any], device: str) -> Dict[str, Any]:
    inputs = _move_batch_to_device(batch, device)
    if "input_ids" not in inputs:
        raise ValueError("Batch must contain `input_ids` for causal language modeling.")

    labels = inputs.get("labels")
    if labels is None:
        labels = inputs["input_ids"].clone()
    else:
        labels = labels.clone()

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)

    inputs["labels"] = labels
    return inputs


def _count_valid_causal_lm_targets(batch: MutableMapping[str, Any]) -> int:
    labels = batch["labels"]
    valid_targets = labels[..., 1:].ne(-100)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        valid_targets = valid_targets & attention_mask[..., 1:].bool()
    return int(valid_targets.sum().item())


def _flatten_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states.reshape(-1, hidden_states.size(-1)).float()


def _flatten_hidden_state_mask(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None

    expected_shape = tuple(hidden_states.shape[:-1])
    if tuple(attention_mask.shape) != expected_shape:
        raise ValueError(
            "attention_mask shape {} must match hidden state shape {} without the hidden dimension.".format(
                tuple(attention_mask.shape),
                expected_shape,
            )
        )
    return attention_mask.reshape(-1).bool()


def _sum_cosine_distances(
    x: torch.Tensor,
    y: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape to compute cosine distance.")

    cosine_similarity = F.cosine_similarity(
        _flatten_hidden_states(x),
        _flatten_hidden_states(y),
        dim=-1,
    )
    valid_mask = _flatten_hidden_state_mask(x, attention_mask)
    if valid_mask is not None:
        cosine_similarity = cosine_similarity[valid_mask]
    if cosine_similarity.numel() == 0:
        return 0.0, 0

    cosine_distance = 1.0 - torch.nan_to_num(cosine_similarity, nan=1.0)
    return float(cosine_distance.sum().item()), int(cosine_distance.numel())


class _ActivationAccumulator:
    """Streaming reducer for activation statistics across all observed tokens."""

    def __init__(self, width: int, agg: str, device: str):
        self.agg = agg.lower()
        self.count = 0

        if self.agg not in {"sum", "mean", "l2", "var"}:
            raise ValueError("Unknown agg: {}.".format(agg))

        self.total = (
            torch.zeros(width, device=device, dtype=torch.float32)
            if self.agg in {"sum", "mean", "var"}
            else None
        )
        self.total_sq = (
            torch.zeros(width, device=device, dtype=torch.float32)
            if self.agg in {"l2", "var"}
            else None
        )

    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.size(-1)).float()
        if flat.numel() == 0:
            return

        self.count += flat.size(0)
        if self.total is not None:
            self.total.add_(flat.sum(dim=0))
        if self.total_sq is not None:
            self.total_sq.add_((flat ** 2).sum(dim=0))

    def finalize(self) -> torch.Tensor:
        if self.agg == "sum":
            return self.total
        if self.agg == "mean":
            return self.total / float(self.count or 1)
        if self.agg == "l2":
            return torch.sqrt(self.total_sq)
        if self.count <= 1:
            return torch.zeros_like(self.total)

        numerator = self.total_sq - (self.total ** 2) / float(self.count)
        return (numerator / float(self.count - 1)).clamp_min(0.0)


def _remove_hooks(hooks: Sequence[Any]) -> None:
    for hook in hooks:
        hook.remove()


def _device_is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _sequence_mean(values: Sequence[float]) -> float:
    return float(torch.tensor(values).mean()) if values else 0.0


def _dataloader_total(dataloader: DataLoader, max_batches: int) -> int | None:
    try:
        return min(max_batches, len(dataloader))
    except TypeError:
        return None


def _first_decoder_layer(adapter: BaseModelAdapter, model: nn.Module) -> nn.Module:
    layers = adapter.get_layers(model)
    if len(layers) == 0:
        raise ValueError("Model does not contain any decoder layers.")
    return layers[0]


def _attention_head_dim(adapter: BaseModelAdapter, model: nn.Module) -> int:
    config_head_dim = getattr(model.config, "head_dim", None)
    if config_head_dim is not None:
        return int(config_head_dim)

    layer = _first_decoder_layer(adapter, model)
    attention = adapter.get_attention_projections(layer)
    num_heads = model.config.num_attention_heads
    if attention.q_proj.out_features % num_heads != 0:
        raise ValueError(
            "q_proj.out_features ({}) must be divisible by num_attention_heads ({}).".format(
                attention.q_proj.out_features,
                num_heads,
            )
        )
    return attention.q_proj.out_features // num_heads


class _BaseEstimator:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        model_adapter: BaseModelAdapter | str | None = None,
    ):
        self.adapter = resolve_model_adapter(model, model_adapter)
        self.model = model
        self.device = device


class _BaseActivationElementEstimator(_BaseEstimator):
    """
    Estimate activation-based importance for attention heads, MLP neurons,
    and embedding channels.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        model_adapter: BaseModelAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        self.model = model.to(device)
        self.num_heads = self.model.config.num_attention_heads
        self.num_key_value_heads = getattr(self.model.config, "num_key_value_heads", self.num_heads)
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = _attention_head_dim(self.adapter, self.model)
        first_layer = _first_decoder_layer(self.adapter, self.model)
        self.intermediate_size = self.adapter.get_mlp_projections(first_layer).down_proj.in_features

    def _query_heads_per_group(self) -> int:
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        return self.num_heads // self.num_key_value_heads

    def estimate_attention_heads(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[int, torch.Tensor]:
        self.model.eval()
        layers = self.adapter.get_layers(self.model)
        importance_by_layer = {
            idx: _ActivationAccumulator(self.num_heads, agg, self.device)
            for idx in range(len(layers))
        }
        hooks = []

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                context = inputs[0].detach()
                batch_size, seq_len, _ = context.shape
                heads = context.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
                norms = torch.norm(heads, dim=-1)
                importance_by_layer[layer_idx].update(norms)

            return hook

        for layer_idx, layer in enumerate(layers):
            hooks.append(
                self.adapter.get_attention_projections(layer).o_proj.register_forward_hook(
                    make_hook(layer_idx)
                )
            )

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating attention heads importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)

        return {
            idx: accumulator.finalize().cpu()
            for idx, accumulator in importance_by_layer.items()
        }

    def estimate_attention_groups(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[int, torch.Tensor]:
        self.model.eval()
        layers = self.adapter.get_layers(self.model)
        query_heads_per_group = self._query_heads_per_group()
        query_group_width = query_heads_per_group * self.head_dim
        importance_by_layer = {
            idx: _ActivationAccumulator(self.num_key_value_heads, agg, self.device)
            for idx in range(len(layers))
        }
        hooks = []

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                context = inputs[0].detach()
                batch_size, seq_len, _ = context.shape
                grouped = context.reshape(
                    batch_size,
                    seq_len,
                    self.num_key_value_heads,
                    query_group_width,
                )
                norms = torch.norm(grouped, dim=-1)
                importance_by_layer[layer_idx].update(norms)

            return hook

        for layer_idx, layer in enumerate(layers):
            hooks.append(
                self.adapter.get_attention_projections(layer).o_proj.register_forward_hook(
                    make_hook(layer_idx)
                )
            )

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating attention groups importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)

        return {
            idx: accumulator.finalize().cpu()
            for idx, accumulator in importance_by_layer.items()
        }

    def estimate_mlp_neurons(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_layer: Dict[int, _ActivationAccumulator] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                hidden_states = inputs[0].detach()
                importance_by_layer[layer_idx].update(hidden_states)

            return hook

        for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
            down_proj = self.adapter.get_mlp_projections(layer).down_proj
            importance_by_layer[layer_idx] = _ActivationAccumulator(
                down_proj.in_features,
                agg,
                self.device,
            )
            hooks.append(down_proj.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating MLP neuron importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)

        return {
            idx: accumulator.finalize().cpu()
            for idx, accumulator in importance_by_layer.items()
        }

    def estimate_embedding_channels(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        importance_by_key: Dict[str, _ActivationAccumulator] = {}
        hooks = []

        for layer_idx, _ in enumerate(self.adapter.get_layers(self.model)):
            importance_by_key["input_layernorm_{}".format(layer_idx)] = _ActivationAccumulator(
                self.hidden_size,
                agg,
                self.device,
            )
            importance_by_key["post_attention_layernorm_{}".format(layer_idx)] = _ActivationAccumulator(
                self.hidden_size,
                agg,
                self.device,
            )
        importance_by_key["final_norm"] = _ActivationAccumulator(
            self.hidden_size,
            agg,
            self.device,
        )

        def make_hook(key: str):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                hidden_states = output.detach()
                importance_by_key[key].update(hidden_states)

            return hook

        for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
            hooks.append(
                self.adapter.get_input_layernorm(layer).register_forward_hook(
                    make_hook("input_layernorm_{}".format(layer_idx))
                )
            )
            hooks.append(
                self.adapter.get_post_attention_layernorm(layer).register_forward_hook(
                    make_hook("post_attention_layernorm_{}".format(layer_idx))
                )
            )
        hooks.append(self.adapter.get_final_norm(self.model).register_forward_hook(make_hook("final_norm")))

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating embedding channels importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)

        return {
            key: accumulator.finalize().cpu()
            for key, accumulator in importance_by_key.items()
        }


class _BaseWeightMagnitudeEstimator(_BaseEstimator):
    """
    Estimate weight magnitude-based importance for embedding channels,
    MLP neurons, attention query heads, and attention groups.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        model_adapter: BaseModelAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        if not hasattr(model, "config"):
            raise ValueError(
                "Model does not have a 'config' attribute. Please pass a Hugging Face model."
            )

        self.hidden_size = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers
        self.num_attention_heads = model.config.num_attention_heads
        self.num_key_value_heads = getattr(model.config, "num_key_value_heads", self.num_attention_heads)
        self.head_dim = _attention_head_dim(self.adapter, self.model)
        self._validate_model_structure()

    def _validate_model_structure(self) -> None:
        embed_tokens = self.adapter.get_embed_tokens(self.model)
        if not hasattr(embed_tokens, "weight"):
            raise ValueError("Missing model embedding weights.")

        if self.num_layers == 0:
            return

        first_layer = _first_decoder_layer(self.adapter, self.model)
        mlp = self.adapter.get_mlp_projections(first_layer)
        for projection_name, projection in (
            ("gate_proj", mlp.gate_proj),
            ("up_proj", mlp.up_proj),
            ("down_proj", mlp.down_proj),
        ):
            if not hasattr(projection, "weight"):
                raise ValueError("Missing MLP {} weight.".format(projection_name))

        attention = self.adapter.get_attention_projections(first_layer)
        for projection_name, projection in (
            ("q_proj", attention.q_proj),
            ("k_proj", attention.k_proj),
            ("v_proj", attention.v_proj),
            ("o_proj", attention.o_proj),
        ):
            if not hasattr(projection, "weight"):
                raise ValueError("Missing attention {} weight.".format(projection_name))

        final_norm = self.adapter.get_final_norm(self.model)
        if not hasattr(final_norm, "weight"):
            raise ValueError("Missing final norm weight.")

        for layer in self.adapter.get_layers(self.model):
            for norm_name, norm in (
                ("input_layernorm", self.adapter.get_input_layernorm(layer)),
                ("post_attention_layernorm", self.adapter.get_post_attention_layernorm(layer)),
            ):
                if not hasattr(norm, "weight"):
                    raise ValueError("Missing {} weight.".format(norm_name))

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
        raise ValueError(
            "Unknown aggregation (norm type): {}. Choose 'l1' or 'l2'.".format(agg)
        )

    @staticmethod
    def _scalar_slice_importance(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.abs()

    @staticmethod
    def _tensor_storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
        detached = tensor.detach()
        return (
            detached.device.type,
            detached.device.index,
            detached.untyped_storage().data_ptr(),
            detached.storage_offset(),
            tuple(detached.shape),
            tuple(detached.stride()),
        )

    def _query_heads_per_group(self) -> int:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        return self.num_attention_heads // self.num_key_value_heads

    def _row_block_norm(
        self,
        weight: torch.Tensor,
        num_blocks: int,
        block_size: int,
        agg: str,
    ) -> torch.Tensor:
        blocks = weight.detach().to(self.device).reshape(num_blocks, block_size, -1)
        return self._calculate_norm(blocks, agg, dim=(1, 2))

    def _column_block_norm(
        self,
        weight: torch.Tensor,
        num_blocks: int,
        block_size: int,
        agg: str,
    ) -> torch.Tensor:
        blocks = weight.detach().to(self.device).reshape(weight.shape[0], num_blocks, block_size)
        return self._calculate_norm(blocks.permute(1, 0, 2), agg, dim=(1, 2))

    def estimate_embedding_channels(self, agg: str = "l2") -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            importance = torch.zeros(self.hidden_size, device=self.device)
            seen_weight_keys: set[tuple[Any, ...]] = set()

            def add_matrix_channel_scores(
                weight: torch.Tensor,
                dim: int,
                *,
                deduplicate: bool = False,
            ) -> None:
                key = self._tensor_storage_key(weight)
                if deduplicate and key in seen_weight_keys:
                    return
                if deduplicate:
                    seen_weight_keys.add(key)
                importance.add_(self._calculate_norm(weight.detach().to(self.device), agg, dim=dim))

            def add_norm_channel_scores(module: nn.Module) -> None:
                weight = getattr(module, "weight", None)
                if weight is None:
                    raise ValueError(
                        "Module {} is missing a weight parameter.".format(type(module).__name__)
                    )
                importance.add_(self._scalar_slice_importance(weight.detach().to(self.device)))

            add_matrix_channel_scores(
                self.adapter.get_embed_tokens(self.model).weight,
                dim=0,
                deduplicate=True,
            )

            lm_head = self.adapter.get_lm_head(self.model)
            if lm_head is not None and hasattr(lm_head, "weight"):
                add_matrix_channel_scores(lm_head.weight, dim=0, deduplicate=True)

            add_norm_channel_scores(self.adapter.get_final_norm(self.model))

            for layer in self.adapter.get_layers(self.model):
                add_norm_channel_scores(self.adapter.get_input_layernorm(layer))
                add_norm_channel_scores(self.adapter.get_post_attention_layernorm(layer))

                attention = self.adapter.get_attention_projections(layer)
                for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
                    add_matrix_channel_scores(projection.weight, dim=0)
                add_matrix_channel_scores(attention.o_proj.weight, dim=1)

                mlp = self.adapter.get_mlp_projections(layer)
                for projection in (mlp.gate_proj, mlp.up_proj):
                    add_matrix_channel_scores(projection.weight, dim=0)
                add_matrix_channel_scores(mlp.down_proj.weight, dim=1)

        return {"embedding_channels": importance.cpu()}

    def estimate_mlp_neurons(self, agg: str = "l2") -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_layer = {}
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
                mlp = self.adapter.get_mlp_projections(layer)
                gate_norm = self._calculate_norm(
                    mlp.gate_proj.weight.detach().to(self.device),
                    agg,
                    dim=1,
                )
                up_norm = self._calculate_norm(
                    mlp.up_proj.weight.detach().to(self.device),
                    agg,
                    dim=1,
                )
                down_norm = self._calculate_norm(
                    mlp.down_proj.weight.detach().to(self.device),
                    agg,
                    dim=0,
                )
                importance_by_layer[layer_idx] = (gate_norm + up_norm + down_norm).cpu()
        return importance_by_layer

    def estimate_attention_heads(self, agg: str = "l2") -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_layer = {}
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
                attention = self.adapter.get_attention_projections(layer)
                q_norm = self._row_block_norm(
                    attention.q_proj.weight,
                    num_blocks=self.num_attention_heads,
                    block_size=self.head_dim,
                    agg=agg,
                )
                o_norm = self._column_block_norm(
                    attention.o_proj.weight,
                    num_blocks=self.num_attention_heads,
                    block_size=self.head_dim,
                    agg=agg,
                )
                importance_by_layer[layer_idx] = (q_norm + o_norm).cpu()

        return importance_by_layer

    def estimate_attention_groups(self, agg: str = "l2") -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_layer = {}
        query_heads_per_group = self._query_heads_per_group()
        query_group_width = query_heads_per_group * self.head_dim
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
                attention = self.adapter.get_attention_projections(layer)
                q_norm = self._row_block_norm(
                    attention.q_proj.weight,
                    num_blocks=self.num_key_value_heads,
                    block_size=query_group_width,
                    agg=agg,
                )
                k_norm = self._row_block_norm(
                    attention.k_proj.weight,
                    num_blocks=self.num_key_value_heads,
                    block_size=self.head_dim,
                    agg=agg,
                )
                v_norm = self._row_block_norm(
                    attention.v_proj.weight,
                    num_blocks=self.num_key_value_heads,
                    block_size=self.head_dim,
                    agg=agg,
                )
                o_norm = self._column_block_norm(
                    attention.o_proj.weight,
                    num_blocks=self.num_key_value_heads,
                    block_size=query_group_width,
                    agg=agg,
                )
                importance_by_layer[layer_idx] = (q_norm + k_norm + v_norm + o_norm).cpu()
        return importance_by_layer


class _BaseSimilarityLayerEstimator(_BaseEstimator):
    """
    Estimate similarity-based importance for attention and MLP sublayers.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        model_adapter: BaseModelAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        self.model = model.eval().to(device)
        self._inputs: Dict[str, torch.Tensor] = {}
        self._outputs: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _capture_input(self, key: str):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            self._inputs[key] = inputs[0].detach().clone()

        return hook

    def _capture_output(self, key: str):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            self._outputs[key] = value.detach().clone()

        return hook

    def _register_hooks(self) -> None:
        self._inputs.clear()
        self._outputs.clear()
        self._hooks = []
        for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
            self._hooks.append(
                self.adapter.get_input_layernorm(layer).register_forward_hook(
                    self._capture_input("attn_in_{}".format(layer_idx))
                )
            )
            self._hooks.append(
                self.adapter.get_attention_module(layer).register_forward_hook(
                    self._capture_output("attn_out_{}".format(layer_idx))
                )
            )
            self._hooks.append(
                self.adapter.get_post_attention_layernorm(layer).register_forward_hook(
                    self._capture_input("mlp_in_{}".format(layer_idx))
                )
            )
            self._hooks.append(
                self.adapter.get_mlp_module(layer).register_forward_hook(
                    self._capture_output("mlp_out_{}".format(layer_idx))
                )
            )

    def _remove_hooks(self) -> None:
        _remove_hooks(self._hooks)
        self._hooks = []

    @torch.no_grad()
    def estimate(self, dataloader: DataLoader) -> Dict[str, List[float]]:
        num_layers = self.model.config.num_hidden_layers
        attention_scores: Dict[int, List[float]] = defaultdict(list)
        mlp_scores: Dict[int, List[float]] = defaultdict(list)

        self._register_hooks()
        try:
            for batch in tqdm(dataloader, desc="Estimating layer importance"):
                if _device_is_cuda(self.device) and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self._inputs.clear()
                self._outputs.clear()
                self.model(**_move_batch_to_device(batch, self.device))

                for layer_idx in range(num_layers):
                    attn_input = self._inputs.get("attn_in_{}".format(layer_idx))
                    attn_output = self._outputs.get("attn_out_{}".format(layer_idx))
                    if attn_input is not None and attn_output is not None:
                        attention_scores[layer_idx].append(
                            calculate_importance(attn_input, attn_input + attn_output)
                        )

                    mlp_input = self._inputs.get("mlp_in_{}".format(layer_idx))
                    mlp_output = self._outputs.get("mlp_out_{}".format(layer_idx))
                    if mlp_input is not None and mlp_output is not None:
                        mlp_scores[layer_idx].append(
                            calculate_importance(mlp_input, mlp_input + mlp_output)
                        )

                self._inputs.clear()
                self._outputs.clear()
                gc.collect()
        finally:
            self._remove_hooks()

        return {
            "attention": [_sequence_mean(attention_scores[idx]) for idx in range(num_layers)],
            "mlp": [_sequence_mean(mlp_scores[idx]) for idx in range(num_layers)],
        }


class _BaseSimilarityBlockEstimator(_BaseEstimator):
    """
    Estimate similarity-based importance for contiguous decoder blocks.
    """

    def __init__(
        self,
        model: nn.Module,
        block_size: int,
        device: str = "cuda",
        model_adapter: BaseModelAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.model = model.eval().to(device)
        self.device = device
        self.block_size = block_size
        num_layers = self.model.config.num_hidden_layers
        if block_size > num_layers:
            raise ValueError(
                "block_size ({}) cannot be greater than num_layers ({})".format(
                    block_size,
                    num_layers,
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
        for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
            self._hooks.append(
                self.adapter.get_input_layernorm(layer).register_forward_hook(
                    lambda module, inputs, output, key=layer_idx: self._attn_inputs.update(
                        {key: inputs[0].detach().clone()}
                    )
                )
            )
            self._hooks.append(
                self.adapter.get_post_attention_layernorm(layer).register_forward_hook(
                    lambda module, inputs, output, key=layer_idx: self._mlp_inputs.update(
                        {key: inputs[0].detach().clone()}
                    )
                )
            )
            self._hooks.append(
                self.adapter.get_mlp_module(layer).register_forward_hook(
                    lambda module, inputs, output, key=layer_idx: self._mlp_outputs.update(
                        {
                            key: (
                                output[0] if isinstance(output, tuple) else output
                            ).detach().clone()
                        }
                    )
                )
            )

    def _remove_hooks(self) -> None:
        _remove_hooks(self._hooks)
        self._hooks = []

    @torch.no_grad()
    def estimate(self, dataloader: DataLoader) -> List[float]:
        num_layers = self.model.config.num_hidden_layers
        max_start = num_layers - self.block_size + 1
        importance_sums = [0.0 for _ in range(max_start)]
        valid_counts = [0 for _ in range(max_start)]

        self._register_hooks()
        try:
            for batch in tqdm(dataloader, desc="Estimating block size={}".format(self.block_size)):
                self._attn_inputs.clear()
                self._mlp_inputs.clear()
                self._mlp_outputs.clear()
                inputs = _move_batch_to_device(batch, self.device)
                self.model(**inputs)
                attention_mask = inputs.get("attention_mask")

                for start_idx in range(max_start):
                    end_idx = start_idx + self.block_size - 1
                    block_input = self._attn_inputs.get(start_idx)
                    residual_input = self._mlp_inputs.get(end_idx)
                    residual_output = self._mlp_outputs.get(end_idx)
                    if block_input is None or residual_input is None or residual_output is None:
                        continue

                    next_hidden = residual_input + residual_output
                    distance_sum, valid_count = _sum_cosine_distances(
                        block_input,
                        next_hidden,
                        attention_mask=attention_mask,
                    )
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


class _BaseBlockPerplexityEstimator(_BaseEstimator):
    """
    Estimate block importance from perplexity deltas after replacing blocks with identities.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        block_size: int,
        device: str = "cpu",
        model_adapter: BaseModelAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers
        if block_size > self.num_layers:
            raise ValueError(
                "block_size ({}) cannot be greater than num_layers ({})".format(
                    block_size,
                    self.num_layers,
                )
            )

    def _calculate_perplexity(
        self,
        current_model: nn.Module,
        dataloader: DataLoader,
        n_samples: int,
    ) -> float:
        current_model.eval()
        total_nll = 0.0
        total_tokens = 0
        batch_size = getattr(dataloader, "batch_size", None)
        batches_to_process = math.ceil(float(n_samples) / batch_size) if batch_size else n_samples

        with torch.no_grad():
            progress_bar = tqdm(
                dataloader,
                total=_dataloader_total(dataloader, batches_to_process),
                desc="Calculating PPL for block",
                leave=False,
            )
            for batch_idx, batch in enumerate(progress_bar):
                if batch_idx >= batches_to_process:
                    break

                inputs = _prepare_causal_lm_inputs(batch, self.device)
                valid_tokens = _count_valid_causal_lm_targets(inputs)
                if valid_tokens == 0:
                    continue

                forward_inputs = dict(inputs)
                forward_inputs["use_cache"] = False
                outputs = current_model(**forward_inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                total_nll += loss.item() * valid_tokens
                total_tokens += valid_tokens
                progress_bar.set_postfix({"loss": loss.item(), "tokens": valid_tokens})

        average_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
        if average_loss <= 0 or math.isinf(average_loss) or math.isnan(average_loss):
            return float("inf")
        return math.exp(average_loss)

    def estimate(
        self,
        dataloader: DataLoader,
        n_samples: int = 1024,
        importance_metric: str = "perplexity_increase",
    ) -> List[float]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for perplexity-based estimation.")

        self.model.eval()
        self.model.to(self.device)

        print("Calculating baseline perplexity (block removal) using {} samples...".format(n_samples))
        baseline_perplexity = self._calculate_perplexity(self.model, dataloader, n_samples)
        print("Baseline Perplexity for block estimation: {:.4f}".format(baseline_perplexity))

        max_start_idx = self.num_layers - self.block_size + 1
        block_importances = []
        layers = self.adapter.get_layers(self.model)

        for start_idx in tqdm(
            range(max_start_idx),
            desc="Estimating Block Importance (Size {}, PPL)".format(self.block_size),
        ):
            layer_indices = range(start_idx, start_idx + self.block_size)
            original_layers = {}
            try:
                for layer_idx in layer_indices:
                    original_layers[layer_idx] = layers[layer_idx]
                    layers[layer_idx] = self.adapter.make_identity_decoder_layer().to(self.device)
                current_perplexity = self._calculate_perplexity(self.model, dataloader, n_samples)
            finally:
                for layer_idx, layer in original_layers.items():
                    layers[layer_idx] = layer

            if importance_metric == "perplexity_increase":
                score = current_perplexity - baseline_perplexity
            elif importance_metric == "perplexity_ratio":
                if baseline_perplexity > 0 and not math.isinf(baseline_perplexity):
                    score = current_perplexity / baseline_perplexity
                else:
                    score = float("inf")
            else:
                raise ValueError("Unknown importance_metric: {}".format(importance_metric))

            block_importances.append(score)
            tqdm.write(
                "Block starting at {} (Size {}): PPL={:.2f}, Score={:.2f}".format(
                    start_idx,
                    self.block_size,
                    current_perplexity,
                    score,
                )
            )

        return block_importances
