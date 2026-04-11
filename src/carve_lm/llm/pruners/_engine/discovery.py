from __future__ import annotations

from typing import Iterable

from ...adapters import BaseModelAdapter
from .types import DiscoveryContext, LayerMetadata, PruningGroup, SliceSpec


def discover_blockwise(model, adapter: BaseModelAdapter, example_batch=None) -> DiscoveryContext:
    del example_batch
    layers = adapter.get_layers(model)
    layer_metadata = tuple(
        _build_layer_metadata(model, adapter, layer_idx)
        for layer_idx in range(len(layers))
    )
    groups: list[PruningGroup] = []

    for metadata in layer_metadata:
        handles = adapter.get_layer_handles(model, metadata.layer_idx)
        query_heads_per_group = metadata.num_attention_heads // metadata.num_key_value_heads

        for group_idx in range(metadata.num_key_value_heads):
            query_head_indices = tuple(
                range(
                    group_idx * query_heads_per_group,
                    (group_idx + 1) * query_heads_per_group,
                )
            )
            query_row_indices = tuple(
                offset
                for head_idx in query_head_indices
                for offset in range(
                    head_idx * metadata.head_dim,
                    (head_idx + 1) * metadata.head_dim,
                )
            )
            kv_row_indices = tuple(
                range(
                    group_idx * metadata.head_dim,
                    (group_idx + 1) * metadata.head_dim,
                )
            )
            groups.append(
                PruningGroup(
                    group_id="attention.layer{}.group{}".format(metadata.layer_idx, group_idx),
                    family="attention",
                    layer_idx=metadata.layer_idx,
                    local_idx=group_idx,
                    members=("q_proj", "k_proj", "v_proj", "o_proj"),
                    dependent_slices=(
                        SliceSpec(
                            module_path=adapter.module_path(model, handles.attention.q_proj),
                            param_name="weight",
                            axis=0,
                            indices=query_row_indices,
                            role="q_proj_out",
                        ),
                        SliceSpec(
                            module_path=adapter.module_path(model, handles.attention.k_proj),
                            param_name="weight",
                            axis=0,
                            indices=kv_row_indices,
                            role="k_proj_out",
                        ),
                        SliceSpec(
                            module_path=adapter.module_path(model, handles.attention.v_proj),
                            param_name="weight",
                            axis=0,
                            indices=kv_row_indices,
                            role="v_proj_out",
                        ),
                        SliceSpec(
                            module_path=adapter.module_path(model, handles.attention.o_proj),
                            param_name="weight",
                            axis=1,
                            indices=query_row_indices,
                            role="o_proj_in",
                        ),
                    ),
                    width=len(query_head_indices),
                    metadata={
                        "query_head_indices": query_head_indices,
                        "query_row_indices": query_row_indices,
                        "kv_row_indices": kv_row_indices,
                    },
                )
            )

        for neuron_idx in range(metadata.intermediate_size):
            groups.append(
                PruningGroup(
                    group_id="mlp.layer{}.neuron{}".format(metadata.layer_idx, neuron_idx),
                    family="mlp",
                    layer_idx=metadata.layer_idx,
                    local_idx=neuron_idx,
                    members=("gate_proj", "up_proj", "down_proj"),
                    dependent_slices=(
                        SliceSpec(
                            module_path=adapter.module_path(model, handles.mlp.gate_proj),
                            param_name="weight",
                            axis=0,
                            indices=(neuron_idx,),
                            role="gate_proj_out",
                        ),
                        SliceSpec(
                            module_path=adapter.module_path(model, handles.mlp.up_proj),
                            param_name="weight",
                            axis=0,
                            indices=(neuron_idx,),
                            role="up_proj_out",
                        ),
                        SliceSpec(
                            module_path=adapter.module_path(model, handles.mlp.down_proj),
                            param_name="weight",
                            axis=1,
                            indices=(neuron_idx,),
                            role="down_proj_in",
                        ),
                    ),
                    metadata={"neuron_idx": neuron_idx},
                )
            )

    first = layer_metadata[0]
    return DiscoveryContext(
        mode="block",
        family_key=adapter.family_for_model(model),
        model_class_path=adapter.model_class_path(model),
        config_class_path=adapter.config_class_path(model),
        base_config=adapter.config_to_dict(model),
        groups=tuple(groups),
        layer_metadata=layer_metadata,
        hidden_size=first.hidden_size,
        num_attention_heads=first.num_attention_heads,
        num_key_value_heads=first.num_key_value_heads,
        head_dim=first.head_dim,
        metadata={"discovery_kind": "adapter_rules"},
    )


def discover_channelwise(model, adapter: BaseModelAdapter, example_batch=None) -> DiscoveryContext:
    del example_batch
    layers = adapter.get_layers(model)
    if len(layers) == 0:
        raise ValueError("Model does not contain any decoder layers.")

    layer_metadata = tuple(
        _build_layer_metadata(model, adapter, layer_idx)
        for layer_idx in range(len(layers))
    )
    first = layer_metadata[0]
    for metadata in layer_metadata[1:]:
        if metadata.hidden_size != first.hidden_size:
            raise ValueError("Channel-wise pruning requires a uniform hidden size across layers.")
        if metadata.head_dim != first.head_dim:
            raise ValueError("Channel-wise pruning requires a uniform head_dim across layers.")
        if metadata.num_attention_heads != first.num_attention_heads:
            raise ValueError(
                "Channel-wise pruning requires a uniform num_attention_heads across layers."
            )
        if metadata.num_key_value_heads != first.num_key_value_heads:
            raise ValueError(
                "Channel-wise pruning requires a uniform num_key_value_heads across layers."
            )

    embed_tokens = adapter.get_embed_tokens(model)
    final_norm = adapter.get_final_norm(model)
    lm_head = adapter.get_lm_head(model)
    head_dim = first.head_dim
    groups: list[PruningGroup] = []
    for channel_idx in range(head_dim):
        residual_indices = tuple(
            head_idx * first.head_dim + channel_idx
            for head_idx in range(first.num_attention_heads)
        )
        kv_indices = tuple(
            group_idx * first.head_dim + channel_idx
            for group_idx in range(first.num_key_value_heads)
        )

        dependent_slices = [
            SliceSpec(
                module_path=adapter.module_path(model, embed_tokens),
                param_name="weight",
                axis=1,
                indices=residual_indices,
                role="embed_tokens_in",
            ),
            SliceSpec(
                module_path=adapter.module_path(model, final_norm),
                param_name="weight",
                axis=0,
                indices=residual_indices,
                role="final_norm",
            ),
        ]

        if lm_head is not None and hasattr(lm_head, "weight"):
            dependent_slices.append(
                SliceSpec(
                    module_path=adapter.module_path(model, lm_head),
                    param_name="weight",
                    axis=1,
                    indices=residual_indices,
                    role="lm_head_in",
                )
            )

        for metadata in layer_metadata:
            handles = adapter.get_layer_handles(model, metadata.layer_idx)
            dependent_slices.extend(
                (
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.input_layernorm),
                        param_name="weight",
                        axis=0,
                        indices=residual_indices,
                        role="input_norm",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.post_attention_layernorm),
                        param_name="weight",
                        axis=0,
                        indices=residual_indices,
                        role="post_attention_norm",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.attention.q_proj),
                        param_name="weight",
                        axis=0,
                        indices=residual_indices,
                        role="q_proj_out",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.attention.q_proj),
                        param_name="weight",
                        axis=1,
                        indices=residual_indices,
                        role="q_proj_in",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.attention.k_proj),
                        param_name="weight",
                        axis=0,
                        indices=kv_indices,
                        role="k_proj_out",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.attention.k_proj),
                        param_name="weight",
                        axis=1,
                        indices=residual_indices,
                        role="k_proj_in",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.attention.v_proj),
                        param_name="weight",
                        axis=0,
                        indices=kv_indices,
                        role="v_proj_out",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.attention.v_proj),
                        param_name="weight",
                        axis=1,
                        indices=residual_indices,
                        role="v_proj_in",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.attention.o_proj),
                        param_name="weight",
                        axis=0,
                        indices=residual_indices,
                        role="o_proj_out",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.attention.o_proj),
                        param_name="weight",
                        axis=1,
                        indices=residual_indices,
                        role="o_proj_in",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.mlp.gate_proj),
                        param_name="weight",
                        axis=1,
                        indices=residual_indices,
                        role="gate_proj_in",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.mlp.up_proj),
                        param_name="weight",
                        axis=1,
                        indices=residual_indices,
                        role="up_proj_in",
                    ),
                    SliceSpec(
                        module_path=adapter.module_path(model, handles.mlp.down_proj),
                        param_name="weight",
                        axis=0,
                        indices=residual_indices,
                        role="down_proj_out",
                    ),
                )
            )

        groups.append(
            PruningGroup(
                group_id="channel.bundle{}".format(channel_idx),
                family="channel",
                layer_idx=None,
                local_idx=channel_idx,
                members=("hidden_stream", "attention", "mlp", "lm_head"),
                dependent_slices=tuple(dependent_slices),
                width=len(residual_indices),
                metadata={
                    "residual_indices": residual_indices,
                    "kv_indices": kv_indices,
                    "channel_idx": channel_idx,
                },
            )
        )

    return DiscoveryContext(
        mode="channel",
        family_key=adapter.family_for_model(model),
        model_class_path=adapter.model_class_path(model),
        config_class_path=adapter.config_class_path(model),
        base_config=adapter.config_to_dict(model),
        groups=tuple(groups),
        layer_metadata=layer_metadata,
        hidden_size=first.hidden_size,
        num_attention_heads=first.num_attention_heads,
        num_key_value_heads=first.num_key_value_heads,
        head_dim=first.head_dim,
        metadata={
            "discovery_kind": "adapter_rules",
            "channel_group_kind": "per_head_channel_bundle",
        },
    )


def _build_layer_metadata(model, adapter: BaseModelAdapter, layer_idx: int) -> LayerMetadata:
    handles = adapter.get_layer_handles(model, layer_idx)
    attention = handles.attention
    mlp = handles.mlp
    return LayerMetadata(
        layer_idx=layer_idx,
        num_attention_heads=attention.num_heads,
        num_key_value_heads=attention.num_key_value_heads,
        head_dim=attention.head_dim,
        intermediate_size=mlp.down_proj.in_features,
        hidden_size=attention.o_proj.out_features,
    )


def filter_groups_by_layers(
    groups: Iterable[PruningGroup],
    *,
    family: str,
    layer_indices: tuple[int, ...] | None,
) -> tuple[PruningGroup, ...]:
    if layer_indices is None:
        return tuple(group for group in groups if group.family == family)
    return tuple(
        group
        for group in groups
        if group.family == family and group.layer_idx in layer_indices
    )
