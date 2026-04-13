from __future__ import annotations

import copy
import math
from collections import defaultdict

import torch.nn as nn

from ...adapters import BaseModelAdapter
from .config import DepthLayerConfig, WidthChannelConfig, WidthGroupConfig
from .discovery import filter_groups_by_layers
from .types import DiscoveryContext, PruningPlan
from .utils import clone_or_share, set_module_by_path


def select_blockwise_plan(
    context: DiscoveryContext,
    scores: dict[str, float],
    config: WidthGroupConfig,
) -> PruningPlan:
    attention_groups = filter_groups_by_layers(
        context.groups,
        family="attention",
        layer_indices=config.attention_layers,
    )
    mlp_groups = filter_groups_by_layers(
        context.groups,
        family="mlp",
        layer_indices=config.mlp_layers,
    )

    if config.global_pruning:
        candidate_groups = tuple(attention_groups) + tuple(mlp_groups)
        target_prune_count = min(
            int(math.floor(len(candidate_groups) * config.pruning_ratio)),
            _max_prunable(candidate_groups, min_keep_per_layer=config.min_keep_per_layer),
        )
        pruned_group_ids = _greedy_prune(
            candidate_groups,
            scores=scores,
            target_prune_count=target_prune_count,
            min_keep_per_layer=config.min_keep_per_layer,
        )
    else:
        pruned_group_ids = []
        for candidate_groups in (attention_groups, mlp_groups):
            target_prune_count = min(
                int(math.floor(len(candidate_groups) * config.pruning_ratio)),
                _max_prunable(candidate_groups, min_keep_per_layer=config.min_keep_per_layer),
            )
            pruned_group_ids.extend(
                _greedy_prune(
                    candidate_groups,
                    scores=scores,
                    target_prune_count=target_prune_count,
                    min_keep_per_layer=config.min_keep_per_layer,
                )
            )

    pruned_group_id_set = set(pruned_group_ids)
    selected_group_ids = tuple(
        group.group_id for group in context.groups if group.group_id not in pruned_group_id_set
    )
    selected_attention_groups_by_layer = defaultdict(list)
    selected_mlp_neurons_by_layer = defaultdict(list)
    for group in context.groups:
        if group.group_id in pruned_group_id_set:
            continue
        if group.family == "attention" and group.layer_idx is not None:
            selected_attention_groups_by_layer[group.layer_idx].append(group.local_idx)
        if group.family == "mlp" and group.layer_idx is not None:
            selected_mlp_neurons_by_layer[group.layer_idx].append(group.local_idx)
    return PruningPlan(
        mode="block",
        selected_group_ids=selected_group_ids,
        pruned_group_ids=tuple(sorted(pruned_group_id_set)),
        scores=_plan_scores(context, scores),
        metadata={
            "global_pruning": config.global_pruning,
            "attention_layers": list(config.attention_layers or []),
            "mlp_layers": list(config.mlp_layers or []),
            "min_keep_per_layer": config.min_keep_per_layer,
            "selected_attention_groups_by_layer": {
                str(layer_idx): sorted(indices)
                for layer_idx, indices in selected_attention_groups_by_layer.items()
            },
            "selected_mlp_neurons_by_layer": {
                str(layer_idx): sorted(indices)
                for layer_idx, indices in selected_mlp_neurons_by_layer.items()
            },
        },
    )


def select_channelwise_plan(
    context: DiscoveryContext,
    scores: dict[str, float],
    config: WidthChannelConfig,
) -> PruningPlan:
    if any(group.family != "channel" for group in context.groups):
        raise ValueError("Channel-wise selection requires a channel discovery context.")

    group_width = context.groups[0].width if context.groups else context.num_attention_heads
    raw_keep_hidden = max(1, context.hidden_size - int(math.floor(context.hidden_size * config.pruning_ratio)))
    round_to = config.round_to or context.num_attention_heads
    if round_to % group_width != 0:
        raise ValueError(
            "round_to={} must be a multiple of the channel bundle width {}.".format(
                round_to,
                group_width,
            )
        )

    raw_keep_hidden -= raw_keep_hidden % round_to
    if raw_keep_hidden <= 0:
        raw_keep_hidden = round_to
    keep_group_count = max(1, raw_keep_hidden // group_width)

    ranked = sorted(
        context.groups,
        key=lambda group: (-scores.get(group.group_id, 0.0), group.group_id),
    )
    selected_group_ids = tuple(group.group_id for group in ranked[:keep_group_count])
    selected_group_id_set = set(selected_group_ids)
    pruned_group_ids = tuple(
        group.group_id for group in context.groups if group.group_id not in selected_group_id_set
    )
    selected_groups = [group for group in context.groups if group.group_id in selected_group_id_set]
    target_hidden_size = sum(group.width for group in selected_groups)
    selected_residual_indices = sorted(
        {
            index
            for group in selected_groups
            for index in group.metadata["residual_indices"]
        }
    )
    selected_kv_indices = sorted(
        {
            index
            for group in selected_groups
            for index in group.metadata["kv_indices"]
        }
    )
    return PruningPlan(
        mode="channel",
        selected_group_ids=tuple(
            group.group_id for group in sorted(selected_groups, key=lambda group: group.local_idx)
        ),
        pruned_group_ids=pruned_group_ids,
        scores=_plan_scores(context, scores),
        metadata={
            "target_hidden_size": target_hidden_size,
            "round_to": round_to,
            "selected_channel_count": len(selected_groups),
            "selected_channel_indices": sorted(group.local_idx for group in selected_groups),
            "selected_residual_indices": selected_residual_indices,
            "selected_kv_indices": selected_kv_indices,
        },
    )


def build_layerwise_plan(model, adapter: BaseModelAdapter, config: DepthLayerConfig) -> PruningPlan:
    num_layers = len(adapter.get_layers(model))
    if config.target_num_layers > num_layers:
        raise ValueError(
            "target_num_layers={} exceeds the model depth {}.".format(
                config.target_num_layers,
                num_layers,
            )
        )

    selected_layer_indices = tuple(range(config.target_num_layers))
    pruned_layer_indices = tuple(range(config.target_num_layers, num_layers))
    return PruningPlan(
        mode="layer",
        selected_group_ids=tuple("layer.{}".format(layer_idx) for layer_idx in selected_layer_indices),
        pruned_group_ids=tuple("layer.{}".format(layer_idx) for layer_idx in pruned_layer_indices),
        scores={},
        metadata={
            "target_num_layers": config.target_num_layers,
            "selected_layer_indices": list(selected_layer_indices),
            "pruned_layer_indices": list(pruned_layer_indices),
        },
    )


def apply_blockwise_plan(
    model,
    adapter: BaseModelAdapter,
    context: DiscoveryContext,
    plan: PruningPlan,
    config: WidthGroupConfig,
):
    pruned_model = clone_or_share(model, config.clone_model)
    group_map = context.group_map()
    selected_groups = [group_map[group_id] for group_id in plan.selected_group_ids]
    selected_attention = defaultdict(list)
    selected_mlp = defaultdict(list)
    for group in selected_groups:
        if group.family == "attention":
            selected_attention[group.layer_idx].append(group)
        elif group.family == "mlp":
            selected_mlp[group.layer_idx].append(group)

    for layer_metadata in context.layer_metadata:
        handles = adapter.get_layer_handles(pruned_model, layer_metadata.layer_idx)
        if layer_metadata.layer_idx in selected_attention:
            keep_groups = sorted(
                group.local_idx for group in selected_attention[layer_metadata.layer_idx]
            )
            if len(keep_groups) != layer_metadata.num_key_value_heads:
                _rewrite_attention_layer(
                    adapter,
                    handles.attention,
                    num_attention_heads=layer_metadata.num_attention_heads,
                    num_key_value_heads=layer_metadata.num_key_value_heads,
                    head_dim=layer_metadata.head_dim,
                    keep_groups=keep_groups,
                )

        if layer_metadata.layer_idx in selected_mlp:
            keep_neurons = sorted(group.local_idx for group in selected_mlp[layer_metadata.layer_idx])
            if len(keep_neurons) != layer_metadata.intermediate_size:
                _rewrite_mlp_layer(handles.mlp, keep_neurons)

    _append_pruning_history(pruned_model, plan)
    return pruned_model


def apply_channelwise_plan(
    model,
    adapter: BaseModelAdapter,
    context: DiscoveryContext,
    plan: PruningPlan,
    config: WidthChannelConfig,
):
    pruned_model = clone_or_share(model, config.clone_model)
    group_map = context.group_map()
    selected_groups = [group_map[group_id] for group_id in plan.selected_group_ids]
    residual_indices = sorted(
        {
            index
            for group in selected_groups
            for index in group.metadata["residual_indices"]
        }
    )
    kv_indices = sorted(
        {
            index
            for group in selected_groups
            for index in group.metadata["kv_indices"]
        }
    )

    if not residual_indices or not kv_indices:
        raise ValueError("Channel-wise pruning must keep at least one channel bundle.")

    new_hidden_size = len(residual_indices)
    new_head_dim = len(selected_groups)
    tied_embeddings = _is_tied_embedding(model, adapter)

    source_embed = adapter.get_embed_tokens(pruned_model)
    new_embed = nn.Embedding(source_embed.num_embeddings, new_hidden_size).to(
        source_embed.weight.device,
        dtype=source_embed.weight.dtype,
    )
    new_embed.weight.data.copy_(source_embed.weight.data[:, residual_indices])
    set_module_by_path(
        pruned_model,
        adapter.module_path(pruned_model, source_embed),
        new_embed,
    )

    final_norm = adapter.get_final_norm(pruned_model)
    set_module_by_path(
        pruned_model,
        adapter.module_path(pruned_model, final_norm),
        _slice_norm(final_norm, residual_indices),
    )

    lm_head = adapter.get_lm_head(pruned_model)
    if lm_head is not None and hasattr(lm_head, "weight"):
        new_lm_head = _slice_linear(lm_head, out_indices=None, in_indices=residual_indices)
        if tied_embeddings:
            new_lm_head.weight = adapter.get_embed_tokens(pruned_model).weight
        set_module_by_path(
            pruned_model,
            adapter.module_path(pruned_model, lm_head),
            new_lm_head,
        )

    for layer_idx in range(len(adapter.get_layers(pruned_model))):
        handles = adapter.get_layer_handles(pruned_model, layer_idx)
        layer = handles.layer
        layer.input_layernorm = _slice_norm(handles.input_layernorm, residual_indices)
        layer.post_attention_layernorm = _slice_norm(
            handles.post_attention_layernorm,
            residual_indices,
        )
        adapter.set_attention_projection(
            layer,
            "q_proj",
            _slice_linear(
                handles.attention.q_proj,
                out_indices=residual_indices,
                in_indices=residual_indices,
            ),
        )
        adapter.set_attention_projection(
            layer,
            "k_proj",
            _slice_linear(
                handles.attention.k_proj,
                out_indices=kv_indices,
                in_indices=residual_indices,
            ),
        )
        adapter.set_attention_projection(
            layer,
            "v_proj",
            _slice_linear(
                handles.attention.v_proj,
                out_indices=kv_indices,
                in_indices=residual_indices,
            ),
        )
        adapter.set_attention_projection(
            layer,
            "o_proj",
            _slice_linear(
                handles.attention.o_proj,
                out_indices=residual_indices,
                in_indices=residual_indices,
            ),
        )
        adapter.patch_attention_metadata(
            adapter.get_attention_module(layer),
            num_heads=context.num_attention_heads,
            num_key_value_heads=context.num_key_value_heads,
            head_dim=new_head_dim,
            hidden_size=new_hidden_size,
        )

        adapter.set_mlp_projection(
            layer,
            "gate_proj",
            _slice_linear(handles.mlp.gate_proj, out_indices=None, in_indices=residual_indices),
        )
        adapter.set_mlp_projection(
            layer,
            "up_proj",
            _slice_linear(handles.mlp.up_proj, out_indices=None, in_indices=residual_indices),
        )
        adapter.set_mlp_projection(
            layer,
            "down_proj",
            _slice_linear(handles.mlp.down_proj, out_indices=residual_indices, in_indices=None),
        )

    adapter.patch_model_hidden_size(pruned_model, new_hidden_size)
    setattr(pruned_model.config, "head_dim", new_head_dim)
    _append_pruning_history(pruned_model, plan)
    return pruned_model


def apply_layerwise_plan(
    model,
    adapter: BaseModelAdapter,
    plan: PruningPlan,
    config: DepthLayerConfig,
):
    pruned_model = clone_or_share(model, config.clone_model)
    layers = adapter.get_layers(pruned_model)
    keep_count = int(plan.metadata["target_num_layers"])
    pruned_model.model.layers = nn.ModuleList(list(layers[:keep_count]))
    adapter.patch_num_hidden_layers(pruned_model, keep_count)
    _append_pruning_history(pruned_model, plan)
    return pruned_model


def _plan_scores(context: DiscoveryContext, scores: dict[str, float]) -> dict[str, float]:
    return {group.group_id: float(scores.get(group.group_id, 0.0)) for group in context.groups}


def _max_prunable(groups, *, min_keep_per_layer: int) -> int:
    counts = defaultdict(int)
    for group in groups:
        counts[(group.family, group.layer_idx)] += 1
    return sum(max(0, count - min_keep_per_layer) for count in counts.values())


def _greedy_prune(groups, *, scores: dict[str, float], target_prune_count: int, min_keep_per_layer: int):
    keep_counts = defaultdict(int)
    for group in groups:
        keep_counts[(group.family, group.layer_idx)] += 1

    pruned_group_ids = []
    for group in sorted(groups, key=lambda candidate: (scores.get(candidate.group_id, 0.0), candidate.group_id)):
        if len(pruned_group_ids) >= target_prune_count:
            break
        key = (group.family, group.layer_idx)
        if keep_counts[key] <= min_keep_per_layer:
            continue
        keep_counts[key] -= 1
        pruned_group_ids.append(group.group_id)
    return pruned_group_ids


def _rewrite_attention_layer(
    adapter: BaseModelAdapter,
    attention_handles,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    keep_groups: list[int],
) -> None:
    if not keep_groups:
        raise ValueError("Cannot prune all attention groups from a layer.")

    query_heads_per_group = num_attention_heads // num_key_value_heads
    query_rows = []
    kv_rows = []
    for group_idx in keep_groups:
        kv_rows.extend(range(group_idx * head_dim, (group_idx + 1) * head_dim))
        for head_idx in range(
            group_idx * query_heads_per_group,
            (group_idx + 1) * query_heads_per_group,
        ):
            query_rows.extend(range(head_idx * head_dim, (head_idx + 1) * head_dim))

    attention_module = attention_handles.module
    attention_module.q_proj = _slice_linear(attention_handles.q_proj, out_indices=query_rows, in_indices=None)
    attention_module.k_proj = _slice_linear(attention_handles.k_proj, out_indices=kv_rows, in_indices=None)
    attention_module.v_proj = _slice_linear(attention_handles.v_proj, out_indices=kv_rows, in_indices=None)
    attention_module.o_proj = _slice_linear(attention_handles.o_proj, out_indices=None, in_indices=query_rows)
    adapter.patch_attention_metadata(
        attention_module,
        num_heads=len(keep_groups) * query_heads_per_group,
        num_key_value_heads=len(keep_groups),
        head_dim=head_dim,
    )


def _rewrite_mlp_layer(mlp_handles, keep_neurons: list[int]) -> None:
    if not keep_neurons:
        raise ValueError("Cannot prune all MLP neurons from a layer.")
    mlp_module = mlp_handles.module
    mlp_module.gate_proj = _slice_linear(mlp_handles.gate_proj, out_indices=keep_neurons, in_indices=None)
    mlp_module.up_proj = _slice_linear(mlp_handles.up_proj, out_indices=keep_neurons, in_indices=None)
    mlp_module.down_proj = _slice_linear(mlp_handles.down_proj, out_indices=None, in_indices=keep_neurons)


def _slice_linear(
    linear: nn.Linear,
    *,
    out_indices: list[int] | None,
    in_indices: list[int] | None,
) -> nn.Linear:
    weight = linear.weight.data
    if out_indices is not None:
        weight = weight[out_indices, :]
    if in_indices is not None:
        weight = weight[:, in_indices]

    new_linear = nn.Linear(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=linear.bias is not None,
    ).to(linear.weight.device, dtype=linear.weight.dtype)
    new_linear.weight.data.copy_(weight)
    if linear.bias is not None:
        bias = linear.bias.data
        if out_indices is not None:
            bias = bias[out_indices]
        new_linear.bias.data.copy_(bias)
    return new_linear


def _slice_norm(norm: nn.Module, keep_indices: list[int]) -> nn.Module:
    new_norm = copy.deepcopy(norm)
    if hasattr(new_norm, "weight") and getattr(new_norm, "weight") is not None:
        new_norm.weight = nn.Parameter(norm.weight.data[keep_indices].clone())
    if hasattr(new_norm, "bias") and getattr(new_norm, "bias") is not None:
        new_norm.bias = nn.Parameter(norm.bias.data[keep_indices].clone())
    if hasattr(new_norm, "normalized_shape"):
        new_norm.normalized_shape = (len(keep_indices),)
    return new_norm


def _append_pruning_history(model, plan: PruningPlan) -> None:
    history = list(getattr(model.config, "pruning_history", []))
    history.append(plan.to_dict())
    setattr(model.config, "pruning_history", history)
    setattr(model.config, "last_pruning_mode", plan.mode)


def _is_tied_embedding(model, adapter: BaseModelAdapter) -> bool:
    embed = adapter.get_embed_tokens(model)
    lm_head = adapter.get_lm_head(model)
    if lm_head is None or not hasattr(lm_head, "weight"):
        return False
    return lm_head.weight.data_ptr() == embed.weight.data_ptr()
