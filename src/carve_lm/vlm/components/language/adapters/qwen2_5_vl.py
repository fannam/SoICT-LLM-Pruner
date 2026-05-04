from __future__ import annotations

import torch.nn as nn

from .decoder import DecoderModelAdapter

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None


def _patch_decoder_config(config, attr: str, value: int) -> None:
    if hasattr(config, "text_config") and config.text_config is not None:
        setattr(config.text_config, attr, int(value))
    if hasattr(config, attr):
        setattr(config, attr, int(value))


class Qwen2_5_VLModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(
            name="qwen2_5_vl",
            model_cls=Qwen2_5_VLForConditionalGeneration or nn.Module,
        )

    def matches(self, model: nn.Module) -> bool:
        if Qwen2_5_VLForConditionalGeneration is not None and isinstance(
            model,
            Qwen2_5_VLForConditionalGeneration,
        ):
            return True
        try:
            language_model = model.model.language_model
            layers = language_model.layers
            if len(layers) == 0:
                return False
            first_layer = layers[0]
            getattr(first_layer, "self_attn")
            getattr(first_layer, "mlp")
            getattr(first_layer, "input_layernorm")
            getattr(first_layer, "post_attention_layernorm")
            getattr(language_model, "embed_tokens")
            getattr(language_model, "norm")
        except Exception:
            return False
        return True

    def get_layers(self, model):
        self.ensure_supported(model)
        return model.model.language_model.layers

    def set_layers(self, model, layers) -> None:
        self.ensure_supported(model)
        model.model.language_model.layers = nn.ModuleList(list(layers))

    def get_embed_tokens(self, model) -> nn.Embedding:
        self.ensure_supported(model)
        return model.model.language_model.embed_tokens

    def get_final_norm(self, model) -> nn.Module:
        self.ensure_supported(model)
        return model.model.language_model.norm

    def hidden_size(self, model) -> int:
        self.ensure_supported(model)
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            return int(text_config.hidden_size)
        return int(model.model.language_model.embed_tokens.embedding_dim)

    def patch_model_hidden_size(self, model, hidden_size: int) -> None:
        self.ensure_supported(model)
        _patch_decoder_config(model.config, "hidden_size", hidden_size)

    def patch_num_hidden_layers(self, model, num_hidden_layers: int) -> None:
        self.ensure_supported(model)
        _patch_decoder_config(model.config, "num_hidden_layers", num_hidden_layers)

    @staticmethod
    def set_num_attention_heads(config, value: int) -> None:
        _patch_decoder_config(config, "num_attention_heads", value)

    @staticmethod
    def set_num_key_value_heads(config, value: int) -> None:
        _patch_decoder_config(config, "num_key_value_heads", value)

    @staticmethod
    def set_hidden_size(config, value: int) -> None:
        _patch_decoder_config(config, "hidden_size", value)

    @staticmethod
    def set_intermediate_size(config, value: int) -> None:
        _patch_decoder_config(config, "intermediate_size", value)

    @staticmethod
    def set_num_hidden_layers(config, value: int) -> None:
        _patch_decoder_config(config, "num_hidden_layers", value)

    @staticmethod
    def set_head_dim(config, value: int) -> None:
        _patch_decoder_config(config, "head_dim", value)
