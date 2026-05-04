from __future__ import annotations

import torch.nn as nn

from ..decoder import DecoderModelAdapter

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


class Qwen3VLModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(
            name="qwen3_vl",
            model_cls=Qwen3VLForConditionalGeneration or nn.Module,
        )

    def matches(self, model: nn.Module) -> bool:
        if Qwen3VLForConditionalGeneration is not None and isinstance(
            model,
            Qwen3VLForConditionalGeneration,
        ):
            return True
        try:
            language_model = model.model.language_model
            layers = language_model.layers
            if len(layers) == 0:
                return False
            first_layer = layers[0]
            attention = first_layer.self_attn
            getattr(attention, "q_norm")
            getattr(attention, "k_norm")
            getattr(first_layer, "mlp")
            getattr(first_layer, "input_layernorm")
            getattr(first_layer, "post_attention_layernorm")
            getattr(language_model, "embed_tokens")
            getattr(language_model, "norm")
            getattr(model.model, "visual")
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
        self.patch_decoder_config(model.config, "hidden_size", hidden_size)

    def patch_num_hidden_layers(self, model, num_hidden_layers: int) -> None:
        self.ensure_supported(model)
        self.patch_decoder_config(model.config, "num_hidden_layers", num_hidden_layers)

    def uses_hidden_stream_channel_pruning(self, model: nn.Module | None = None) -> bool:
        del model
        return True
