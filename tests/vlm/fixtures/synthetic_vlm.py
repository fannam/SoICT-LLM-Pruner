from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from carve_lm.vlm.adapters import DecoderModelAdapter, get_model_adapter, register_model_adapter
from tests.fixtures.synthetic_models import SyntheticBackbone, SyntheticConfig

_TEXT_FIELDS = (
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "vocab_size",
)


@dataclass
class SyntheticVLMTextConfig:
    hidden_size: int = 8
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 6
    vocab_size: int = 16
    head_dim: int = 2

    @classmethod
    def from_dict(cls, payload: dict) -> "SyntheticVLMTextConfig":
        return cls(**payload)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SyntheticVLMConfig:
    text_config: SyntheticVLMTextConfig = field(default_factory=SyntheticVLMTextConfig)
    tie_word_embeddings: bool = False
    vision_feature_size: int = 4

    def __post_init__(self) -> None:
        self.sync_from_text_config()

    def sync_from_text_config(self) -> None:
        for field_name in _TEXT_FIELDS:
            setattr(self, field_name, getattr(self.text_config, field_name))

    @classmethod
    def from_dict(cls, payload: dict) -> "SyntheticVLMConfig":
        text_config = payload.get("text_config", {})
        if not isinstance(text_config, SyntheticVLMTextConfig):
            text_config = SyntheticVLMTextConfig.from_dict(text_config)
        config = cls(
            text_config=text_config,
            tie_word_embeddings=payload.get("tie_word_embeddings", False),
            vision_feature_size=payload.get("vision_feature_size", 4),
        )
        for field_name in _TEXT_FIELDS:
            if field_name in payload:
                setattr(config.text_config, field_name, payload[field_name])
        config.sync_from_text_config()
        return config

    def to_dict(self) -> dict:
        payload = {
            "text_config": self.text_config.to_dict(),
            "tie_word_embeddings": self.tie_word_embeddings,
            "vision_feature_size": self.vision_feature_size,
        }
        for field_name in _TEXT_FIELDS:
            payload[field_name] = getattr(self, field_name)
        return payload


class SyntheticVisualEncoder(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.proj = nn.Linear(feature_size, 1, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        flat = pixel_values.reshape(pixel_values.size(0), -1).float()
        return self.proj(flat[:, : self.proj.in_features])


class SyntheticVLMModel(nn.Module):
    def __init__(self, config: SyntheticVLMConfig):
        super().__init__()
        self.config = config
        self.visual = SyntheticVisualEncoder(config.vision_feature_size)
        self.model = SyntheticBackbone(SyntheticConfig(**config.text_config.to_dict()))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        output_hidden_states=False,
        **kwargs,
    ):
        del kwargs
        if pixel_values is None:
            raise ValueError("pixel_values is required for SyntheticVLMModel.")

        hidden_states = self.model.embed_tokens(input_ids)
        visual_bias = self.visual(pixel_values).to(hidden_states.dtype).view(hidden_states.size(0), 1, 1)
        hidden_states = hidden_states + visual_bias
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()

        all_hidden_states = [hidden_states]
        for layer in self.model.layers:
            layer_output = layer(hidden_states)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()
            all_hidden_states.append(hidden_states)
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        outputs = {"logits": logits}
        if output_hidden_states:
            outputs["hidden_states"] = tuple(all_hidden_states)
        if labels is not None:
            shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[..., 1:].reshape(-1)
            valid_mask = shift_labels.ne(-100)
            if attention_mask is not None:
                valid_mask = valid_mask & attention_mask[..., 1:].reshape(-1).bool()
            if torch.any(valid_mask):
                outputs["loss"] = F.cross_entropy(shift_logits[valid_mask], shift_labels[valid_mask])
            else:
                outputs["loss"] = shift_logits.sum() * 0
        return outputs

    def generate(self, input_ids=None, pixel_values=None, max_new_tokens: int = 1, **kwargs):
        del kwargs
        if pixel_values is None:
            raise ValueError("pixel_values is required for SyntheticVLMModel.generate.")
        if input_ids is None:
            raise ValueError("input_ids is required for SyntheticVLMModel.generate.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0.")

        repeated = input_ids[:, -1:].repeat(1, max_new_tokens)
        return torch.cat([input_ids, repeated], dim=1)


def _patch_decoder_config(config, attr: str, value: int) -> None:
    setattr(config.text_config, attr, int(value))
    setattr(config, attr, int(value))


class SyntheticVLMModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(name="synthetic_vlm", model_cls=SyntheticVLMModel)

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


def ensure_registered_synthetic_vlm_adapter() -> SyntheticVLMModelAdapter:
    try:
        return register_model_adapter(SyntheticVLMModelAdapter())
    except KeyError:
        return get_model_adapter("synthetic_vlm")


def make_synthetic_vlm(**overrides) -> SyntheticVLMModel:
    text_payload = {
        field_name: overrides.pop(field_name)
        for field_name in tuple(overrides)
        if field_name in _TEXT_FIELDS
    }
    config = SyntheticVLMConfig(
        text_config=SyntheticVLMTextConfig(**text_payload),
        tie_word_embeddings=overrides.pop("tie_word_embeddings", False),
        vision_feature_size=overrides.pop("vision_feature_size", 4),
    )
    if overrides:
        raise TypeError("Unknown SyntheticVLMConfig overrides: {}".format(sorted(overrides)))
    return SyntheticVLMModel(config)
