from __future__ import annotations

from collections.abc import Iterable

from transformers import AutoModelForCausalLM, PreTrainedModel

from .adapters import BaseModelAdapter, resolve_model_adapter


def _normalize_pruned_indices(
    indices: Iterable[int] | None,
    *,
    num_layers: int,
    field_name: str,
) -> list[int]:
    normalized = sorted({int(index) for index in (indices or ())})
    for index in normalized:
        if not 0 <= index < num_layers:
            raise ValueError(
                "{} contains invalid layer index {} for a model with {} layers.".format(
                    field_name,
                    index,
                    num_layers,
                )
            )
    return normalized


def apply_component_pruning_from_config(
    model: PreTrainedModel,
    model_adapter: BaseModelAdapter | str | None = None,
) -> PreTrainedModel:
    """
    Replay component pruning encoded on ``model.config``.

    CarveLM component pruning keeps the original config class and stores the
    removed attention/MLP layer indices on the config. This helper reconstructs
    the matching identity-module architecture after a regular HF load.
    """

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model does not expose a config, so component pruning cannot be replayed.")

    attention_layers = getattr(config, "attention_layer_to_prune", None)
    mlp_layers = getattr(config, "mlp_layer_to_prune", None)
    if not attention_layers and not mlp_layers:
        return model

    adapter = resolve_model_adapter(model, model_adapter)
    layers = adapter.get_layers(model)
    num_layers = len(layers)

    normalized_attention_layers = _normalize_pruned_indices(
        attention_layers,
        num_layers=num_layers,
        field_name="attention_layer_to_prune",
    )
    normalized_mlp_layers = _normalize_pruned_indices(
        mlp_layers,
        num_layers=num_layers,
        field_name="mlp_layer_to_prune",
    )

    for layer_idx in normalized_attention_layers:
        adapter.set_attention_module(layers[layer_idx], adapter.make_identity_attention())
    for layer_idx in normalized_mlp_layers:
        adapter.set_mlp_module(layers[layer_idx], adapter.make_identity_mlp())

    config.attention_layer_to_prune = normalized_attention_layers
    config.mlp_layer_to_prune = normalized_mlp_layers
    return model


class PrunedAutoModelForCausalLM(AutoModelForCausalLM):
    """
    Auto-model loader that restores CarveLM component-pruned decoder layouts.

    Save the pruned checkpoint with the normal model ``save_pretrained`` flow,
    then reload it with this class so any attention/MLP layers recorded on the
    config are rebuilt as identity pass-through modules.
    """

    @classmethod
    def from_config(cls, config, **kwargs):
        model = super().from_config(config, **kwargs)
        return apply_component_pruning_from_config(model)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return apply_component_pruning_from_config(model)


__all__ = [
    "PrunedAutoModelForCausalLM",
    "apply_component_pruning_from_config",
]
