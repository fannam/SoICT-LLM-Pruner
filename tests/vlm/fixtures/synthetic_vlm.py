from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from carve_lm.vlm.components.language.adapters import DecoderModelAdapter, get_model_adapter, register_model_adapter
from tests.fixtures.synthetic_models import (
    SyntheticAttention,
    SyntheticBackbone,
    SyntheticConfig,
    SyntheticMLP,
    SyntheticNorm,
)

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


@dataclass
class SyntheticQwenVisionConfig:
    hidden_size: int = 8
    depth: int = 2
    num_heads: int = 4
    intermediate_size: int = 6
    spatial_merge_size: int = 2
    out_hidden_size: int = 8

    @classmethod
    def from_dict(cls, payload: dict) -> "SyntheticQwenVisionConfig":
        return cls(**payload)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SyntheticQwen25VLConfig:
    text_config: SyntheticVLMTextConfig = field(default_factory=SyntheticVLMTextConfig)
    vision_config: SyntheticQwenVisionConfig = field(default_factory=SyntheticQwenVisionConfig)
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "SyntheticQwen25VLConfig":
        text_config = payload.get("text_config", {})
        vision_config = payload.get("vision_config", {})
        if not isinstance(text_config, SyntheticVLMTextConfig):
            text_config = SyntheticVLMTextConfig.from_dict(text_config)
        if not isinstance(vision_config, SyntheticQwenVisionConfig):
            vision_config = SyntheticQwenVisionConfig.from_dict(vision_config)
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            tie_word_embeddings=payload.get("tie_word_embeddings", False),
        )

    def to_dict(self) -> dict:
        return {
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "tie_word_embeddings": self.tie_word_embeddings,
        }


class SyntheticQwenVisionAttention(nn.Module):
    def __init__(self, config: SyntheticQwenVisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        return self.proj(torch.tanh(q + k + v))


class SyntheticQwenVisionBlock(nn.Module):
    def __init__(self, config: SyntheticQwenVisionConfig):
        super().__init__()
        self.norm1 = SyntheticNorm(config.hidden_size)
        self.norm2 = SyntheticNorm(config.hidden_size)
        self.attn = SyntheticQwenVisionAttention(config)
        self.mlp = SyntheticMLP(
            SyntheticConfig(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                head_dim=config.hidden_size // config.num_heads,
            )
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class SyntheticQwenPatchMerger(nn.Module):
    def __init__(self, config: SyntheticQwenVisionConfig):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.merge_factor = config.spatial_merge_size**2
        self.ln_q = SyntheticNorm(config.hidden_size)
        merged_hidden_size = config.hidden_size * self.merge_factor
        self.mlp = nn.Sequential(
            nn.Linear(merged_hidden_size, merged_hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(merged_hidden_size, config.out_hidden_size, bias=True),
        )

    def forward(self, hidden_states):
        hidden_states = self.ln_q(hidden_states)
        batch_size, seq_len, hidden_size = hidden_states.shape
        if seq_len % self.merge_factor != 0:
            raise ValueError("SyntheticQwenPatchMerger requires seq_len divisible by merge_factor.")
        hidden_states = hidden_states.reshape(
            batch_size,
            seq_len // self.merge_factor,
            hidden_size * self.merge_factor,
        )
        return self.mlp(hidden_states)


class SyntheticQwenPatchEmbed(nn.Module):
    def __init__(self, config: SyntheticQwenVisionConfig):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, pixel_values):
        return self.proj(pixel_values.float())


class SyntheticQwenVisualModel(nn.Module):
    def __init__(self, config: SyntheticQwenVisionConfig):
        super().__init__()
        self.patch_embed = SyntheticQwenPatchEmbed(config)
        self.blocks = nn.ModuleList([SyntheticQwenVisionBlock(config) for _ in range(config.depth)])
        self.merger = SyntheticQwenPatchMerger(config)

    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.merger(hidden_states)


class SyntheticQwen25VLBackbone(nn.Module):
    def __init__(self, config: SyntheticQwen25VLConfig):
        super().__init__()
        self.visual = SyntheticQwenVisualModel(config.vision_config)
        self.language_model = SyntheticBackbone(SyntheticConfig(**config.text_config.to_dict()))


class SyntheticQwen25VLModel(nn.Module):
    def __init__(self, config: SyntheticQwen25VLConfig):
        super().__init__()
        self.config = config
        self.model = SyntheticQwen25VLBackbone(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

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
            raise ValueError("pixel_values is required for SyntheticQwen25VLModel.")

        visual_features = self.model.visual(pixel_values)
        visual_bias = visual_features.mean(dim=1, keepdim=True)
        hidden_states = self.model.language_model.embed_tokens(input_ids) + visual_bias
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()

        all_hidden_states = [hidden_states]
        for layer in self.model.language_model.layers:
            layer_output = layer(hidden_states)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()
            all_hidden_states.append(hidden_states)
        hidden_states = self.model.language_model.norm(hidden_states)
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


def make_synthetic_qwen2_5_vl(**overrides) -> SyntheticQwen25VLModel:
    text_payload = {
        field_name: overrides.pop(field_name)
        for field_name in tuple(overrides)
        if field_name in _TEXT_FIELDS
    }
    vision_payload = {
        field_name: overrides.pop(field_name)
        for field_name in tuple(overrides)
        if field_name
        in {
            "vision_hidden_size",
            "vision_depth",
            "vision_num_heads",
            "vision_intermediate_size",
            "spatial_merge_size",
        }
    }
    text_config = SyntheticVLMTextConfig(**text_payload)
    vision_config = SyntheticQwenVisionConfig(
        hidden_size=vision_payload.pop("vision_hidden_size", 8),
        depth=vision_payload.pop("vision_depth", 2),
        num_heads=vision_payload.pop("vision_num_heads", 4),
        intermediate_size=vision_payload.pop("vision_intermediate_size", 6),
        spatial_merge_size=vision_payload.pop("spatial_merge_size", 2),
        out_hidden_size=text_config.hidden_size,
    )
    config = SyntheticQwen25VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        tie_word_embeddings=overrides.pop("tie_word_embeddings", False),
    )
    if overrides or vision_payload:
        raise TypeError(
            "Unknown SyntheticQwen25VLConfig overrides: {}".format(
                sorted((*overrides.keys(), *vision_payload.keys()))
            )
        )
    return SyntheticQwen25VLModel(config)


@dataclass
class SyntheticQwen3VisionConfig:
    hidden_size: int = 8
    depth: int = 2
    num_heads: int = 4
    intermediate_size: int = 6
    spatial_merge_size: int = 2
    out_hidden_size: int = 8
    deepstack_visual_indexes: tuple[int, ...] = (0,)

    @classmethod
    def from_dict(cls, payload: dict) -> "SyntheticQwen3VisionConfig":
        payload = dict(payload)
        if "deepstack_visual_indexes" in payload:
            payload["deepstack_visual_indexes"] = tuple(payload["deepstack_visual_indexes"])
        return cls(**payload)

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["deepstack_visual_indexes"] = list(self.deepstack_visual_indexes)
        return payload


@dataclass
class SyntheticQwen3VLConfig:
    text_config: SyntheticVLMTextConfig = field(default_factory=SyntheticVLMTextConfig)
    vision_config: SyntheticQwen3VisionConfig = field(default_factory=SyntheticQwen3VisionConfig)
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "SyntheticQwen3VLConfig":
        text_config = payload.get("text_config", {})
        vision_config = payload.get("vision_config", {})
        if not isinstance(text_config, SyntheticVLMTextConfig):
            text_config = SyntheticVLMTextConfig.from_dict(text_config)
        if not isinstance(vision_config, SyntheticQwen3VisionConfig):
            vision_config = SyntheticQwen3VisionConfig.from_dict(vision_config)
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            tie_word_embeddings=payload.get("tie_word_embeddings", False),
        )

    def to_dict(self) -> dict:
        return {
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "tie_word_embeddings": self.tie_word_embeddings,
        }


class SyntheticQwen3Attention(SyntheticAttention):
    def __init__(self, config: SyntheticConfig):
        super().__init__(config)
        self.q_norm = SyntheticNorm(config.head_dim)
        self.k_norm = SyntheticNorm(config.head_dim)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.num_heads != self.num_key_value_heads:
            k = k.repeat_interleave(self.num_key_value_groups, dim=2)
            v = v.repeat_interleave(self.num_key_value_groups, dim=2)
        context = torch.tanh(q + k + v).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(context)


class SyntheticQwen3Layer(nn.Module):
    def __init__(self, config: SyntheticConfig):
        super().__init__()
        self.input_layernorm = SyntheticNorm(config.hidden_size)
        self.post_attention_layernorm = SyntheticNorm(config.hidden_size)
        self.self_attn = SyntheticQwen3Attention(config)
        self.mlp = SyntheticMLP(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class SyntheticQwen3TextModel(nn.Module):
    def __init__(self, config: SyntheticVLMTextConfig):
        super().__init__()
        synthetic_config = SyntheticConfig(**config.to_dict())
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SyntheticQwen3Layer(synthetic_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = SyntheticNorm(config.hidden_size)


class SyntheticQwen3VisionMLP(nn.Module):
    def __init__(self, config: SyntheticQwen3VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states):
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states)))


class SyntheticQwen3VisionBlock(nn.Module):
    def __init__(self, config: SyntheticQwen3VisionConfig):
        super().__init__()
        self.norm1 = SyntheticNorm(config.hidden_size)
        self.norm2 = SyntheticNorm(config.hidden_size)
        self.attn = SyntheticQwenVisionAttention(
            SyntheticQwenVisionConfig(
                hidden_size=config.hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                spatial_merge_size=config.spatial_merge_size,
                out_hidden_size=config.out_hidden_size,
            )
        )
        self.mlp = SyntheticQwen3VisionMLP(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class SyntheticQwen3PatchMerger(nn.Module):
    def __init__(self, config: SyntheticQwen3VisionConfig, use_postshuffle_norm: bool = False):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.merge_factor = config.spatial_merge_size**2
        self.hidden_size = config.hidden_size * self.merge_factor
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = SyntheticNorm(norm_size)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=True)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        if seq_len % self.merge_factor != 0:
            raise ValueError("SyntheticQwen3PatchMerger requires seq_len divisible by merge_factor.")
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.reshape(
                batch_size,
                seq_len // self.merge_factor,
                hidden_size * self.merge_factor,
            )
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.reshape(
                batch_size,
                seq_len // self.merge_factor,
                hidden_size * self.merge_factor,
            )
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states)))


class SyntheticQwen3VisualModel(nn.Module):
    def __init__(self, config: SyntheticQwen3VisionConfig):
        super().__init__()
        self.patch_embed = SyntheticQwenPatchEmbed(
            SyntheticQwenVisionConfig(
                hidden_size=config.hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                spatial_merge_size=config.spatial_merge_size,
                out_hidden_size=config.out_hidden_size,
            )
        )
        self.blocks = nn.ModuleList([SyntheticQwen3VisionBlock(config) for _ in range(config.depth)])
        self.merger = SyntheticQwen3PatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [
                SyntheticQwen3PatchMerger(config, use_postshuffle_norm=True)
                for _ in self.deepstack_visual_indexes
            ]
        )

    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        deepstack_features = []
        for block_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if block_idx in self.deepstack_visual_indexes:
                merger_idx = self.deepstack_visual_indexes.index(block_idx)
                deepstack_features.append(self.deepstack_merger_list[merger_idx](hidden_states))
        merged = self.merger(hidden_states)
        if deepstack_features:
            merged = merged + sum(deepstack_features) * 0.0
        return merged


class SyntheticQwen3VLBackbone(nn.Module):
    def __init__(self, config: SyntheticQwen3VLConfig):
        super().__init__()
        self.visual = SyntheticQwen3VisualModel(config.vision_config)
        self.language_model = SyntheticQwen3TextModel(config.text_config)


class SyntheticQwen3VLModel(nn.Module):
    def __init__(self, config: SyntheticQwen3VLConfig):
        super().__init__()
        self.config = config
        self.model = SyntheticQwen3VLBackbone(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        output_hidden_states=False,
        use_cache=None,
        **kwargs,
    ):
        del use_cache
        if kwargs:
            raise TypeError("Unexpected SyntheticQwen3VLModel inputs: {}".format(sorted(kwargs)))
        if pixel_values is None:
            raise ValueError("pixel_values is required for SyntheticQwen3VLModel.")

        visual_features = self.model.visual(pixel_values)
        visual_bias = visual_features.mean(dim=1, keepdim=True)
        hidden_states = self.model.language_model.embed_tokens(input_ids) + visual_bias
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()

        all_hidden_states = [hidden_states]
        for layer in self.model.language_model.layers:
            layer_output = layer(hidden_states)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()
            all_hidden_states.append(hidden_states)
        hidden_states = self.model.language_model.norm(hidden_states)
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


def make_synthetic_qwen3_vl(**overrides) -> SyntheticQwen3VLModel:
    text_payload = {
        field_name: overrides.pop(field_name)
        for field_name in tuple(overrides)
        if field_name in _TEXT_FIELDS
    }
    vision_payload = {
        field_name: overrides.pop(field_name)
        for field_name in tuple(overrides)
        if field_name
        in {
            "vision_hidden_size",
            "vision_depth",
            "vision_num_heads",
            "vision_intermediate_size",
            "spatial_merge_size",
            "deepstack_visual_indexes",
        }
    }
    text_config = SyntheticVLMTextConfig(**text_payload)
    vision_config = SyntheticQwen3VisionConfig(
        hidden_size=vision_payload.pop("vision_hidden_size", 8),
        depth=vision_payload.pop("vision_depth", 2),
        num_heads=vision_payload.pop("vision_num_heads", 4),
        intermediate_size=vision_payload.pop("vision_intermediate_size", 6),
        spatial_merge_size=vision_payload.pop("spatial_merge_size", 2),
        out_hidden_size=text_config.hidden_size,
        deepstack_visual_indexes=tuple(vision_payload.pop("deepstack_visual_indexes", (0,))),
    )
    config = SyntheticQwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        tie_word_embeddings=overrides.pop("tie_word_embeddings", False),
    )
    if overrides or vision_payload:
        raise TypeError(
            "Unknown SyntheticQwen3VLConfig overrides: {}".format(
                sorted((*overrides.keys(), *vision_payload.keys()))
            )
        )
    return SyntheticQwen3VLModel(config)
