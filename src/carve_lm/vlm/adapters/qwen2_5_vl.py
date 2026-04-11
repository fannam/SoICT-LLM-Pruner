from __future__ import annotations

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


if Qwen2_5_VLForConditionalGeneration is not None:
    class Qwen2_5_VLModelAdapter(DecoderModelAdapter):
        def __init__(self):
            super().__init__(name="qwen2_5_vl", model_cls=Qwen2_5_VLForConditionalGeneration)

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
