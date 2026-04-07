from __future__ import annotations

import copy
import warnings
from typing import Mapping

import torch
import torch.nn as nn

from ..core import (
    PRUNER_REGISTRY,
    PRUNING_STRATEGY_REGISTRY,
    calculate_embedding_channels_global_score,
)
from ._compat import warn_pruner_alias
from ._shared import _BasePruner


class BaseElementPruningStrategy:
    """Extension point for element-level pruning strategies."""

    def prune(self, pruner: "ElementPruner", **kwargs):
        raise NotImplementedError


@PRUNER_REGISTRY.register("width", aliases=("element",))
class WidthPruner(_BasePruner):
    """
    Generic width-oriented pruner backed by model adapters and pruning strategies.
    """

    def __init__(self, original_model, device: str = "cuda", model_adapter=None):
        super().__init__(model=original_model, device=device, model_adapter=model_adapter)
        self.model = original_model
        self.original_config = original_model.config
        self.original_num_key_value_heads = getattr(
            self.original_config,
            "num_key_value_heads",
            self.original_config.num_attention_heads,
        )
        self.original_num_attention_heads = self.original_config.num_attention_heads
        self.original_num_layers = self.original_config.num_hidden_layers
        self.original_hidden_size = self.original_config.hidden_size
        self.dtype = getattr(self.model, "dtype", None)

        layers = self.adapter.get_layers(self.model)
        if len(layers) == 0:
            raise ValueError("Model does not contain any decoder layers.")

        first_layer = layers[0]
        self.original_intermediate_size = self.adapter.get_mlp_projections(first_layer).gate_proj.out_features
        self.head_dim = getattr(self.original_config, "head_dim", None)
        if self.head_dim is None:
            q_proj = self.adapter.get_attention_projections(first_layer).q_proj
            if q_proj.out_features % self.original_num_attention_heads != 0:
                raise ValueError(
                    "q_proj.out_features ({}) must be divisible by num_attention_heads ({}).".format(
                        q_proj.out_features,
                        self.original_num_attention_heads,
                    )
                )
            self.head_dim = q_proj.out_features // self.original_num_attention_heads

    @staticmethod
    def _warn_and_abort(message: str):
        warnings.warn(message, stacklevel=2)
        return None

    def _clone_config(self):
        return copy.deepcopy(self.model.config)

    def _instantiate_model(self, config):
        return self.adapter.instantiate_model(config, device=self.device, dtype=self.dtype)

    def _load_matching_state(self, target_model):
        source_state = self.model.state_dict()
        target_state = target_model.state_dict()
        for key, target_value in target_state.items():
            source_value = source_state.get(key)
            if source_value is not None and tuple(source_value.shape) == tuple(target_value.shape):
                target_value.copy_(source_value)
        target_model.load_state_dict(target_state, strict=True)
        return target_model

    def _annotate_pruned_model(self, pruned_model, strategy_name: str, **metadata):
        history = list(getattr(pruned_model.config, "pruning_history", []))
        history.append({"strategy": strategy_name, **metadata})
        setattr(pruned_model.config, "pruning_history", history)
        setattr(pruned_model.config, "last_pruning_strategy", strategy_name)
        return pruned_model

    def _normalize_strategy_name(self, strategy_name: str) -> str:
        if strategy_name in PRUNING_STRATEGY_REGISTRY:
            return strategy_name
        prefixed_name = "element.{}".format(strategy_name)
        if prefixed_name in PRUNING_STRATEGY_REGISTRY:
            return prefixed_name
        available = ", ".join(self.available_strategies()) or "<empty>"
        raise KeyError(
            "Unknown element pruning strategy '{}'. Available: {}.".format(
                strategy_name,
                available,
            )
        )

    def available_strategies(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in PRUNING_STRATEGY_REGISTRY.names()
            if name.startswith("element.")
        )

    def apply(self, strategy_name: str, **kwargs):
        normalized_name = self._normalize_strategy_name(strategy_name)
        strategy_cls = PRUNING_STRATEGY_REGISTRY.get(normalized_name)
        strategy = strategy_cls()
        return strategy.prune(self, **kwargs)

    def prune_attention_query(self, head_importance, target_num_attention_heads):
        return self.apply(
            "element.attention_query",
            head_importance=head_importance,
            target_num_attention_heads=target_num_attention_heads,
        )

    def prune_attention_group(
        self,
        head_importance=None,
        target_group: int | None = None,
        *,
        group_importance=None,
    ):
        if target_group is None:
            raise TypeError("target_group must be provided.")
        return self.apply(
            "element.attention_group",
            head_importance=head_importance,
            group_importance=group_importance,
            target_group=target_group,
        )

    def prune_mlp(self, neuron_importance, target_num_neurons):
        return self.apply(
            "element.mlp",
            neuron_importance=neuron_importance,
            target_num_neurons=target_num_neurons,
        )

    def prune_embeddings(self, embedding_importance, target_embedding_size):
        return self.apply(
            "element.embedding_channels",
            embedding_importance=embedding_importance,
            target_embedding_size=target_embedding_size,
        )

    def _validate_layer_score_map(
        self,
        score_map: Mapping[int, torch.Tensor],
        score_name: str,
        expected_numel: int | None = None,
        min_numel: int | None = None,
    ) -> None:
        missing_layers = [layer_idx for layer_idx in range(self.original_num_layers) if layer_idx not in score_map]
        if missing_layers:
            raise ValueError(
                "{} is missing scores for layers {}.".format(
                    score_name,
                    missing_layers,
                )
            )

        for layer_idx in range(self.original_num_layers):
            scores = score_map[layer_idx]
            tensor = scores if torch.is_tensor(scores) else torch.as_tensor(scores)
            if tensor.ndim != 1:
                raise ValueError(
                    "{}[{}] must be a 1D tensor.".format(score_name, layer_idx)
                )
            if expected_numel is not None and tensor.numel() != expected_numel:
                raise ValueError(
                    "{}[{}] must contain {} scores, received {}.".format(
                        score_name,
                        layer_idx,
                        expected_numel,
                        tensor.numel(),
                    )
                )
            if min_numel is not None and tensor.numel() < min_numel:
                raise ValueError(
                    "{}[{}] must contain at least {} scores, received {}.".format(
                        score_name,
                        layer_idx,
                        min_numel,
                        tensor.numel(),
                    )
                )

    @staticmethod
    def _to_device_tensor(values, device: str) -> torch.Tensor:
        tensor = values if torch.is_tensor(values) else torch.as_tensor(values)
        return tensor.to(device)

    @staticmethod
    def _sorted_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
        return torch.sort(torch.topk(scores, k=k, largest=True).indices).values

    def _build_head_mask(
        self,
        keep_indices: torch.Tensor,
        total_heads: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        mask = torch.zeros(total_heads * self.head_dim, dtype=torch.bool, device=device)
        for head_idx in keep_indices.tolist():
            start = head_idx * self.head_dim
            mask[start : start + self.head_dim] = True
        return mask

    def _query_heads_per_group(self) -> int:
        if self.original_num_attention_heads % self.original_num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        return self.original_num_attention_heads // self.original_num_key_value_heads

    def _resolve_attention_group_scores(
        self,
        *,
        head_importance,
        group_importance,
    ) -> Mapping[int, torch.Tensor]:
        has_head_importance = head_importance is not None
        has_group_importance = group_importance is not None

        if has_head_importance == has_group_importance:
            raise ValueError(
                "Exactly one of group_importance or head_importance must be provided."
            )

        if has_group_importance:
            self._validate_layer_score_map(
                score_map=group_importance,
                score_name="group_importance",
                expected_numel=self.original_num_key_value_heads,
            )
            return group_importance

        warnings.warn(
            "head_importance for attention-group pruning is deprecated. "
            "Pass group_importance from estimate_attention_groups() instead.",
            stacklevel=3,
        )
        self._validate_layer_score_map(
            score_map=head_importance,
            score_name="head_importance",
            expected_numel=self.original_num_attention_heads,
        )

        query_heads_per_group = self._query_heads_per_group()
        resolved_scores = {}
        for layer_idx in range(self.original_num_layers):
            scores = self._to_device_tensor(head_importance[layer_idx], self.device)
            resolved_scores[layer_idx] = scores.reshape(
                self.original_num_key_value_heads,
                query_heads_per_group,
            ).mean(dim=1).cpu()
        return resolved_scores

    @staticmethod
    def _copy_tensor(target_tensor: torch.Tensor, source_tensor: torch.Tensor) -> None:
        if tuple(target_tensor.shape) != tuple(source_tensor.shape):
            raise ValueError(
                "Shape mismatch. Expected {}, got {}.".format(
                    tuple(target_tensor.shape),
                    tuple(source_tensor.shape),
                )
            )
        target_tensor.data.copy_(
            source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
        )

    def _copy_linear(
        self,
        linear: nn.Linear,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        self._copy_tensor(linear.weight, weight)
        if linear.bias is None:
            if bias is not None:
                raise ValueError("Tried to copy bias into a bias-free linear layer.")
            return
        if bias is None:
            linear.bias.data.zero_()
            return
        self._copy_tensor(linear.bias, bias)

    @staticmethod
    def _linear_with_reference(
        reference: nn.Linear,
        in_features: int,
        out_features: int,
        bias: bool,
    ) -> nn.Linear:
        new_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        ).to(reference.weight.device)
        return new_linear.to(reference.weight.dtype)

    @staticmethod
    def _global_embedding_scores(embedding_importance) -> torch.Tensor:
        if isinstance(embedding_importance, Mapping):
            return calculate_embedding_channels_global_score(embedding_importance)
        return embedding_importance if torch.is_tensor(embedding_importance) else torch.as_tensor(embedding_importance)


@PRUNING_STRATEGY_REGISTRY.register("element.attention_query")
class AttentionQueryPruningStrategy(BaseElementPruningStrategy):
    def prune(
        self,
        pruner: ElementPruner,
        head_importance,
        target_num_attention_heads: int,
    ):
        if target_num_attention_heads % pruner.original_num_key_value_heads != 0:
            return pruner._warn_and_abort("Number of query heads is invalid.")
        if target_num_attention_heads == pruner.original_num_attention_heads:
            return pruner._warn_and_abort("No attention heads need to be removed.")
        if target_num_attention_heads > pruner.original_num_attention_heads:
            return pruner._warn_and_abort(
                "New number of attention heads must not exceed the original one."
            )
        if target_num_attention_heads < pruner.original_num_key_value_heads:
            return pruner._warn_and_abort(
                "Number of attention heads must be at least {}.".format(
                    pruner.original_num_key_value_heads
                )
            )

        pruner._validate_layer_score_map(
            score_map=head_importance,
            score_name="head_importance",
            expected_numel=pruner.original_num_attention_heads,
        )

        new_config = pruner._clone_config()
        pruner.adapter.set_num_attention_heads(new_config, target_num_attention_heads)
        pruner.adapter.set_head_dim(new_config, pruner.head_dim)
        new_model = pruner._load_matching_state(pruner._instantiate_model(new_config))

        source_layers = pruner.adapter.get_layers(pruner.model)
        target_layers = pruner.adapter.get_layers(new_model)
        for layer_idx, (source_layer, target_layer) in enumerate(zip(source_layers, target_layers)):
            source_scores = pruner._to_device_tensor(head_importance[layer_idx], pruner.device)
            keep_indices = pruner._sorted_topk_indices(
                source_scores,
                target_num_attention_heads,
            )
            attention_src = pruner.adapter.get_attention_projections(source_layer)
            attention_dst = pruner.adapter.get_attention_projections(target_layer)
            head_mask = pruner._build_head_mask(
                keep_indices=keep_indices,
                total_heads=pruner.original_num_attention_heads,
                device=attention_src.q_proj.weight.device,
            )

            pruner._copy_linear(
                attention_dst.q_proj,
                attention_src.q_proj.weight.data[head_mask, :],
                attention_src.q_proj.bias.data[head_mask] if attention_src.q_proj.bias is not None else None,
            )
            pruner._copy_linear(
                attention_dst.o_proj,
                attention_src.o_proj.weight.data[:, head_mask],
                attention_src.o_proj.bias.data if attention_src.o_proj.bias is not None else None,
            )

        return pruner._annotate_pruned_model(
            new_model,
            "element.attention_query",
            target_num_attention_heads=target_num_attention_heads,
        )


@PRUNING_STRATEGY_REGISTRY.register("element.attention_group")
class AttentionGroupPruningStrategy(BaseElementPruningStrategy):
    def prune(
        self,
        pruner: ElementPruner,
        head_importance=None,
        target_group: int | None = None,
        group_importance=None,
    ):
        if target_group is None:
            raise TypeError("target_group must be provided.")

        query_heads_per_group = pruner._query_heads_per_group()
        if target_group >= pruner.original_num_key_value_heads:
            return pruner._warn_and_abort("target_group does not remove any attention group.")
        if target_group <= 0:
            return pruner._warn_and_abort("target_group must be a positive integer.")

        resolved_group_scores = pruner._resolve_attention_group_scores(
            head_importance=head_importance,
            group_importance=group_importance,
        )

        new_config = pruner._clone_config()
        pruner.adapter.set_num_attention_heads(new_config, target_group * query_heads_per_group)
        pruner.adapter.set_num_key_value_heads(new_config, target_group)
        pruner.adapter.set_head_dim(new_config, pruner.head_dim)
        new_model = pruner._load_matching_state(pruner._instantiate_model(new_config))

        source_layers = pruner.adapter.get_layers(pruner.model)
        target_layers = pruner.adapter.get_layers(new_model)

        for layer_idx, (source_layer, target_layer) in enumerate(zip(source_layers, target_layers)):
            group_scores = pruner._to_device_tensor(resolved_group_scores[layer_idx], pruner.device)
            keep_groups = pruner._sorted_topk_indices(group_scores, target_group)

            keep_query_heads = []
            for group_idx in keep_groups.tolist():
                start = group_idx * query_heads_per_group
                keep_query_heads.extend(range(start, start + query_heads_per_group))
            keep_query_heads = torch.tensor(keep_query_heads, device=pruner.device, dtype=torch.long)

            attention_src = pruner.adapter.get_attention_projections(source_layer)
            attention_dst = pruner.adapter.get_attention_projections(target_layer)
            query_mask = pruner._build_head_mask(
                keep_indices=keep_query_heads,
                total_heads=pruner.original_num_attention_heads,
                device=attention_src.q_proj.weight.device,
            )
            kv_mask = pruner._build_head_mask(
                keep_indices=keep_groups,
                total_heads=pruner.original_num_key_value_heads,
                device=attention_src.k_proj.weight.device,
            )

            pruner._copy_linear(
                attention_dst.q_proj,
                attention_src.q_proj.weight.data[query_mask, :],
                attention_src.q_proj.bias.data[query_mask] if attention_src.q_proj.bias is not None else None,
            )
            pruner._copy_linear(
                attention_dst.k_proj,
                attention_src.k_proj.weight.data[kv_mask, :],
                attention_src.k_proj.bias.data[kv_mask] if attention_src.k_proj.bias is not None else None,
            )
            pruner._copy_linear(
                attention_dst.v_proj,
                attention_src.v_proj.weight.data[kv_mask, :],
                attention_src.v_proj.bias.data[kv_mask] if attention_src.v_proj.bias is not None else None,
            )
            pruner._copy_linear(
                attention_dst.o_proj,
                attention_src.o_proj.weight.data[:, query_mask],
                attention_src.o_proj.bias.data if attention_src.o_proj.bias is not None else None,
            )

        return pruner._annotate_pruned_model(
            new_model,
            "element.attention_group",
            target_num_key_value_heads=target_group,
            target_num_attention_heads=target_group * query_heads_per_group,
        )


@PRUNING_STRATEGY_REGISTRY.register("element.mlp")
class MLPPruningStrategy(BaseElementPruningStrategy):
    def prune(
        self,
        pruner: ElementPruner,
        neuron_importance,
        target_num_neurons: int,
    ):
        if target_num_neurons >= pruner.original_intermediate_size:
            return pruner._warn_and_abort(
                "target_num_neurons is larger than or equal to the original intermediate size."
            )
        if target_num_neurons <= 0:
            return pruner._warn_and_abort("target_num_neurons must be a positive integer.")

        pruner._validate_layer_score_map(
            score_map=neuron_importance,
            score_name="neuron_importance",
            min_numel=target_num_neurons,
        )

        pruned_model = copy.deepcopy(pruner.model).to(pruner.device)
        pruner.adapter.set_intermediate_size(pruned_model.config, target_num_neurons)

        for layer_idx, layer in enumerate(pruner.adapter.get_layers(pruned_model)):
            scores = pruner._to_device_tensor(neuron_importance[layer_idx], pruner.device)
            keep_indices = pruner._sorted_topk_indices(scores, target_num_neurons).tolist()
            mlp = pruner.adapter.get_mlp_projections(layer)

            for projection_name, projection in (
                ("gate_proj", mlp.gate_proj),
                ("up_proj", mlp.up_proj),
            ):
                new_projection = pruner._linear_with_reference(
                    reference=projection,
                    in_features=projection.in_features,
                    out_features=target_num_neurons,
                    bias=projection.bias is not None,
                )
                pruner._copy_linear(
                    new_projection,
                    projection.weight.data[keep_indices, :],
                    projection.bias.data[keep_indices] if projection.bias is not None else None,
                )
                pruner.adapter.set_mlp_projection(layer, projection_name, new_projection)

            down_proj = mlp.down_proj
            new_down_proj = pruner._linear_with_reference(
                reference=down_proj,
                in_features=target_num_neurons,
                out_features=down_proj.out_features,
                bias=down_proj.bias is not None,
            )
            pruner._copy_linear(
                new_down_proj,
                down_proj.weight.data[:, keep_indices],
                down_proj.bias.data if down_proj.bias is not None else None,
            )
            pruner.adapter.set_mlp_projection(layer, "down_proj", new_down_proj)

        return pruner._annotate_pruned_model(
            pruned_model,
            "element.mlp",
            target_num_neurons=target_num_neurons,
        )


@PRUNING_STRATEGY_REGISTRY.register("element.embedding_channels")
class EmbeddingChannelPruningStrategy(BaseElementPruningStrategy):
    def prune(
        self,
        pruner: ElementPruner,
        embedding_importance,
        target_embedding_size: int,
    ):
        if target_embedding_size >= pruner.original_hidden_size:
            return pruner._warn_and_abort(
                "target_embedding_size is larger than or equal to the original hidden size."
            )
        if target_embedding_size <= 0:
            return pruner._warn_and_abort("target_embedding_size must be a positive integer.")

        global_scores = pruner._global_embedding_scores(embedding_importance)
        if global_scores.ndim != 1:
            raise ValueError("embedding_importance must resolve to a 1D tensor.")
        if target_embedding_size > global_scores.numel():
            raise ValueError(
                "target_embedding_size={} exceeds available channels {}.".format(
                    target_embedding_size,
                    global_scores.numel(),
                )
            )

        new_config = pruner._clone_config()
        pruner.adapter.set_hidden_size(new_config, target_embedding_size)
        pruner.adapter.set_head_dim(new_config, pruner.head_dim)
        new_model = pruner._load_matching_state(pruner._instantiate_model(new_config))

        keep_indices = pruner._sorted_topk_indices(
            pruner._to_device_tensor(global_scores, pruner.device),
            target_embedding_size,
        ).tolist()

        source_embed = pruner.adapter.get_embed_tokens(pruner.model)
        target_embed = pruner.adapter.get_embed_tokens(new_model)
        pruner._copy_tensor(target_embed.weight, source_embed.weight.data[:, keep_indices])

        source_lm_head = pruner.adapter.get_lm_head(pruner.model)
        target_lm_head = pruner.adapter.get_lm_head(new_model)
        if source_lm_head is not None and target_lm_head is not None and hasattr(source_lm_head, "weight"):
            pruner._copy_tensor(target_lm_head.weight, source_lm_head.weight.data[:, keep_indices])
            source_bias = getattr(source_lm_head, "bias", None)
            target_bias = getattr(target_lm_head, "bias", None)
            if source_bias is not None and target_bias is not None:
                pruner._copy_tensor(target_bias, source_bias.data)

        pruner._copy_tensor(
            pruner.adapter.get_final_norm(new_model).weight,
            pruner.adapter.get_final_norm(pruner.model).weight.data[keep_indices],
        )

        source_layers = pruner.adapter.get_layers(pruner.model)
        target_layers = pruner.adapter.get_layers(new_model)
        for source_layer, target_layer in zip(source_layers, target_layers):
            pruner._copy_tensor(
                pruner.adapter.get_input_layernorm(target_layer).weight,
                pruner.adapter.get_input_layernorm(source_layer).weight.data[keep_indices],
            )
            pruner._copy_tensor(
                pruner.adapter.get_post_attention_layernorm(target_layer).weight,
                pruner.adapter.get_post_attention_layernorm(source_layer).weight.data[keep_indices],
            )

            attention_src = pruner.adapter.get_attention_projections(source_layer)
            attention_dst = pruner.adapter.get_attention_projections(target_layer)
            for projection_src, projection_dst in (
                (attention_src.q_proj, attention_dst.q_proj),
                (attention_src.k_proj, attention_dst.k_proj),
                (attention_src.v_proj, attention_dst.v_proj),
            ):
                pruner._copy_linear(
                    projection_dst,
                    projection_src.weight.data[:, keep_indices],
                    projection_src.bias.data if projection_src.bias is not None else None,
                )
            pruner._copy_linear(
                attention_dst.o_proj,
                attention_src.o_proj.weight.data[keep_indices, :],
                attention_src.o_proj.bias.data[keep_indices] if attention_src.o_proj.bias is not None else None,
            )

            mlp_src = pruner.adapter.get_mlp_projections(source_layer)
            mlp_dst = pruner.adapter.get_mlp_projections(target_layer)
            for projection_src, projection_dst in (
                (mlp_src.gate_proj, mlp_dst.gate_proj),
                (mlp_src.up_proj, mlp_dst.up_proj),
            ):
                pruner._copy_linear(
                    projection_dst,
                    projection_src.weight.data[:, keep_indices],
                    projection_src.bias.data if projection_src.bias is not None else None,
                )
            pruner._copy_linear(
                mlp_dst.down_proj,
                mlp_src.down_proj.weight.data[keep_indices, :],
                mlp_src.down_proj.bias.data[keep_indices] if mlp_src.down_proj.bias is not None else None,
            )

        return pruner._annotate_pruned_model(
            new_model,
            "element.embedding_channels",
            target_hidden_size=target_embedding_size,
        )


def available_element_pruning_strategies() -> tuple[str, ...]:
    warnings.warn(
        "available_element_pruning_strategies() is deprecated; use WidthPruner.available_strategies().",
        DeprecationWarning,
        stacklevel=2,
    )
    return tuple(
        name
        for name in PRUNING_STRATEGY_REGISTRY.names()
        if name.startswith("element.")
    )


class ElementPruner(WidthPruner):
    """Backward-compatible alias for legacy code."""

    def __init__(self, *args, **kwargs):
        warn_pruner_alias("ElementPruner", "WidthPruner", stacklevel=3)
        super().__init__(*args, **kwargs)


class Llama3ElementPruner(ElementPruner):
    """Backward-compatible alias for legacy code."""


class Qwen2ElementPruner(ElementPruner):
    """Backward-compatible alias for legacy code."""


class MistralElementPruner(ElementPruner):
    """Backward-compatible alias for legacy code."""


__all__ = [
    "BaseElementPruningStrategy",
    "WidthPruner",
    "ElementPruner",
    "Llama3ElementPruner",
    "Qwen2ElementPruner",
    "MistralElementPruner",
    "available_element_pruning_strategies",
]
