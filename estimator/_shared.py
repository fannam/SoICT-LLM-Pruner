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

from soict_llm_pruner_core import BaseModelAdapter, resolve_model_adapter
from utils import calculate_importance


def _move_batch_to_device(batch: MutableMapping[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _activation_aggregate(x: torch.Tensor, agg: str) -> torch.Tensor:
    flat = x.reshape(-1, x.size(-1))
    if agg == "sum":
        return flat.sum(dim=0)
    if agg == "mean":
        return flat.mean(dim=0)
    if agg == "l2":
        return torch.sqrt((flat ** 2).sum(dim=0))
    if agg == "var":
        return flat.var(dim=0)
    raise ValueError("Unknown agg: {}.".format(agg))


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
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = _attention_head_dim(self.adapter, self.model)
        first_layer = _first_decoder_layer(self.adapter, self.model)
        self.intermediate_size = self.adapter.get_mlp_projections(first_layer).down_proj.in_features

    def estimate_attention_heads(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[int, torch.Tensor]:
        self.model.eval()
        layers = self.adapter.get_layers(self.model)
        importance_by_layer = {
            idx: torch.zeros(self.num_heads, device=self.device)
            for idx in range(len(layers))
        }
        token_counts: Dict[int, int] = defaultdict(int)
        hooks = []

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                context = inputs[0].detach()
                batch_size, seq_len, _ = context.shape
                heads = context.view(batch_size, seq_len, self.num_heads, self.head_dim)
                norms = torch.norm(heads, dim=-1)
                importance_by_layer[layer_idx] += _activation_aggregate(norms, agg)
                token_counts[layer_idx] += batch_size * seq_len

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
            idx: (values / float(token_counts[idx] or 1)).cpu()
            for idx, values in importance_by_layer.items()
        }

    def estimate_mlp_neurons(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_layer: Dict[int, torch.Tensor] = {}
        token_counts: Dict[int, int] = defaultdict(int)
        hooks = []

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                hidden_states = inputs[0].detach()
                importance_by_layer[layer_idx] += _activation_aggregate(hidden_states, agg)
                batch_size, seq_len, _ = hidden_states.shape
                token_counts[layer_idx] += batch_size * seq_len

            return hook

        for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
            down_proj = self.adapter.get_mlp_projections(layer).down_proj
            importance_by_layer[layer_idx] = torch.zeros(down_proj.in_features, device=self.device)
            hooks.append(down_proj.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Estimating MLP neuron importance"):
                    self.model(**_move_batch_to_device(batch, self.device))
        finally:
            _remove_hooks(hooks)

        return {
            idx: (values / float(token_counts[idx] or 1)).cpu()
            for idx, values in importance_by_layer.items()
        }

    def estimate_embedding_channels(
        self,
        dataloader: DataLoader,
        agg: str = "l2",
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        importance_by_key: Dict[str, torch.Tensor] = {}
        token_counts: Dict[str, int] = defaultdict(int)
        hooks = []

        for layer_idx, _ in enumerate(self.adapter.get_layers(self.model)):
            importance_by_key["input_layernorm_{}".format(layer_idx)] = torch.zeros(
                self.hidden_size,
                device=self.device,
            )
            importance_by_key["post_attention_layernorm_{}".format(layer_idx)] = torch.zeros(
                self.hidden_size,
                device=self.device,
            )
        importance_by_key["final_norm"] = torch.zeros(self.hidden_size, device=self.device)

        def make_hook(key: str):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                hidden_states = output.detach()
                batch_size, seq_len, _ = hidden_states.shape
                importance_by_key[key] += _activation_aggregate(hidden_states, agg)
                token_counts[key] += batch_size * seq_len

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
            key: (values / float(token_counts[key] or 1)).cpu()
            for key, values in importance_by_key.items()
        }


class _BaseWeightMagnitudeEstimator(_BaseEstimator):
    """
    Estimate weight magnitude-based importance for embedding channels,
    MLP neurons, and attention heads.
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
        if not hasattr(mlp.gate_proj, "weight"):
            raise ValueError("Missing MLP gate projection weight.")

        attention = self.adapter.get_attention_projections(first_layer)
        for projection_name, projection in (
            ("q_proj", attention.q_proj),
            ("k_proj", attention.k_proj),
            ("v_proj", attention.v_proj),
            ("o_proj", attention.o_proj),
        ):
            if not hasattr(projection, "weight"):
                raise ValueError("Missing attention {} weight.".format(projection_name))

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

    def estimate_embedding_channels(self, agg: str = "l2") -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            embedding_weights = self.adapter.get_embed_tokens(self.model).weight.detach().to(self.device)
            importance = self._calculate_norm(embedding_weights, agg, dim=0)
        return {"embedding_channels": importance.cpu()}

    def estimate_mlp_neurons(self, agg: str = "l2") -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_layer = {}
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
                gate_projection_weights = (
                    self.adapter.get_mlp_projections(layer).gate_proj.weight.detach().to(self.device)
                )
                importance_by_layer[layer_idx] = self._calculate_norm(
                    gate_projection_weights,
                    agg,
                    dim=1,
                ).cpu()
        return importance_by_layer

    def estimate_attention_heads(self, agg: str = "l2") -> Dict[int, torch.Tensor]:
        self.model.eval()
        importance_by_layer = {}
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.adapter.get_layers(self.model)):
                attention = self.adapter.get_attention_projections(layer)

                q_weights = attention.q_proj.weight.detach().to(self.device).view(
                    self.num_attention_heads,
                    self.head_dim,
                    -1,
                )
                q_norm = self._calculate_norm(q_weights, agg, dim=(1, 2))

                k_weights = attention.k_proj.weight.detach().to(self.device).view(
                    self.num_key_value_heads,
                    self.head_dim,
                    -1,
                )
                k_norm = self._calculate_norm(k_weights, agg, dim=(1, 2))

                v_weights = attention.v_proj.weight.detach().to(self.device).view(
                    self.num_key_value_heads,
                    self.head_dim,
                    -1,
                )
                v_norm = self._calculate_norm(v_weights, agg, dim=(1, 2))

                if self.num_attention_heads != self.num_key_value_heads:
                    repetition_factor = self.num_attention_heads // self.num_key_value_heads
                    k_norm = k_norm.repeat_interleave(repetition_factor)
                    v_norm = v_norm.repeat_interleave(repetition_factor)

                o_weights = attention.o_proj.weight.detach().to(self.device).view(
                    attention.o_proj.out_features,
                    self.num_attention_heads,
                    self.head_dim,
                )
                o_norm = self._calculate_norm(o_weights.permute(1, 0, 2), agg, dim=(1, 2))

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
        results: List[List[float]] = [[] for _ in range(max_start)]

        self._register_hooks()
        try:
            for batch in tqdm(dataloader, desc="Estimating block size={}".format(self.block_size)):
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
                    flat_block_input = block_input.view(-1, block_input.size(-1)).float()
                    flat_next_hidden = next_hidden.view(-1, next_hidden.size(-1)).float()
                    cosine_similarity = F.cosine_similarity(
                        flat_block_input,
                        flat_next_hidden,
                        dim=1,
                    ).nan_to_num(1.0).mean().item()
                    results[start_idx].append(1.0 - cosine_similarity)
        finally:
            self._remove_hooks()

        return [sum(values) / len(values) if values else 0.0 for values in results]


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
        total_loss = 0.0
        samples_done = 0
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

                inputs = _move_batch_to_device(batch, self.device)
                outputs = current_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs["labels"],
                    use_cache=False,
                )
                loss = outputs.loss
                current_batch_size = inputs["input_ids"].size(0)
                total_loss += loss.item() * current_batch_size
                samples_done += current_batch_size
                progress_bar.set_postfix({"loss": loss.item()})

        average_loss = total_loss / samples_done if samples_done > 0 else 0.0
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
