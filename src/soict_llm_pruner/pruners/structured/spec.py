from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class AttentionHandles:
    module: nn.Module
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    num_heads: int
    num_key_value_heads: int
    head_dim: int


@dataclass(frozen=True)
class MLPHandles:
    module: nn.Module
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear


@dataclass(frozen=True)
class LayerHandles:
    layer: nn.Module
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module
    attention: AttentionHandles
    mlp: MLPHandles


class ArchitectureSpec:
    family_key = "llama_like"

    def matches(self, model: nn.Module) -> bool:
        try:
            self.get_layers(model)
            first_layer = self.get_layers(model)[0]
            self.get_attention(first_layer)
            self.get_mlp(first_layer)
            self.get_embed_tokens(model)
            self.get_final_norm(model)
        except Exception:
            return False
        return True

    def clone_model(self, model: nn.Module) -> nn.Module:
        return copy.deepcopy(model)

    def model_class_path(self, model: nn.Module) -> str:
        return object_path(model.__class__)

    def config_class_path(self, model: nn.Module) -> str:
        return object_path(model.config.__class__)

    def config_to_dict(self, model: nn.Module) -> dict[str, Any]:
        config = model.config
        if hasattr(config, "to_dict") and callable(config.to_dict):
            return dict(config.to_dict())
        if hasattr(config, "__dict__"):
            return {
                key: value
                for key, value in vars(config).items()
                if not key.startswith("_") and _is_json_like(value)
            }
        raise TypeError("Unsupported config type {}.".format(type(config).__name__))

    def config_from_dict(self, config_cls: type, payload: dict[str, Any]):
        if hasattr(config_cls, "from_dict") and inspect.ismethod(config_cls.from_dict):
            return config_cls.from_dict(payload)
        if hasattr(config_cls, "from_dict") and callable(getattr(config_cls, "from_dict")):
            return config_cls.from_dict(payload)
        try:
            return config_cls(**payload)
        except TypeError:
            config = config_cls()
            for key, value in payload.items():
                setattr(config, key, value)
            return config

    def instantiate_model(
        self,
        model_cls: type[nn.Module],
        config,
        device: str | None = None,
        dtype=None,
    ) -> nn.Module:
        model = model_cls(config)
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype=dtype)
        return model

    def get_layers(self, model: nn.Module):
        return model.model.layers

    def get_embed_tokens(self, model: nn.Module) -> nn.Embedding:
        return model.model.embed_tokens

    def get_lm_head(self, model: nn.Module):
        return getattr(model, "lm_head", None)

    def get_final_norm(self, model: nn.Module):
        return model.model.norm

    def get_attention(self, layer: nn.Module):
        return layer.self_attn

    def get_mlp(self, layer: nn.Module):
        return layer.mlp

    def get_input_layernorm(self, layer: nn.Module):
        return layer.input_layernorm

    def get_post_attention_layernorm(self, layer: nn.Module):
        return layer.post_attention_layernorm

    def get_attention_handles(self, layer: nn.Module) -> AttentionHandles:
        attention = self.get_attention(layer)
        num_heads = int(getattr(attention, "num_heads", layer.self_attn.q_proj.out_features // self.head_dim(layer)))
        num_key_value_heads = int(getattr(attention, "num_key_value_heads", num_heads))
        head_dim = int(getattr(attention, "head_dim", self.head_dim(layer)))
        return AttentionHandles(
            module=attention,
            q_proj=attention.q_proj,
            k_proj=attention.k_proj,
            v_proj=attention.v_proj,
            o_proj=attention.o_proj,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

    def get_mlp_handles(self, layer: nn.Module) -> MLPHandles:
        mlp = self.get_mlp(layer)
        return MLPHandles(
            module=mlp,
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
        )

    def get_layer_handles(self, model: nn.Module, layer_idx: int) -> LayerHandles:
        layer = self.get_layers(model)[layer_idx]
        return LayerHandles(
            layer=layer,
            input_layernorm=self.get_input_layernorm(layer),
            post_attention_layernorm=self.get_post_attention_layernorm(layer),
            attention=self.get_attention_handles(layer),
            mlp=self.get_mlp_handles(layer),
        )

    def hidden_size(self, model: nn.Module) -> int:
        return int(model.config.hidden_size)

    def head_dim(self, layer_or_model) -> int:
        if hasattr(layer_or_model, "config"):
            config = layer_or_model.config
            if getattr(config, "head_dim", None) is not None:
                return int(config.head_dim)
            q_proj = self.get_attention_handles(self.get_layers(layer_or_model)[0]).q_proj
            return q_proj.out_features // int(layer_or_model.config.num_attention_heads)
        q_proj = self.get_attention(layer_or_model).q_proj
        num_heads = int(getattr(self.get_attention(layer_or_model), "num_heads", 0))
        if num_heads <= 0:
            raise ValueError("Unable to infer attention head_dim from layer.")
        if q_proj.out_features % num_heads != 0:
            raise ValueError("q_proj.out_features must be divisible by num_heads.")
        return q_proj.out_features // num_heads

    def family_for_model(self, model: nn.Module) -> str:
        mapping = {
            "LlamaForCausalLM": "llama",
            "MistralForCausalLM": "mistral",
            "Qwen2ForCausalLM": "qwen2",
        }
        return mapping.get(model.__class__.__name__, self.family_key)

    def patch_attention_metadata(
        self,
        attention_module: nn.Module,
        *,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        hidden_size: int | None = None,
    ) -> None:
        if hasattr(attention_module, "num_heads"):
            attention_module.num_heads = int(num_heads)
        if hasattr(attention_module, "num_key_value_heads"):
            attention_module.num_key_value_heads = int(num_key_value_heads)
        if hasattr(attention_module, "num_key_value_groups"):
            attention_module.num_key_value_groups = int(num_heads // num_key_value_heads)
        if hasattr(attention_module, "head_dim"):
            attention_module.head_dim = int(head_dim)
        if hidden_size is not None and hasattr(attention_module, "hidden_size"):
            attention_module.hidden_size = int(hidden_size)

    def patch_model_hidden_size(self, model: nn.Module, hidden_size: int) -> None:
        setattr(model.config, "hidden_size", int(hidden_size))

    def patch_num_hidden_layers(self, model: nn.Module, num_hidden_layers: int) -> None:
        setattr(model.config, "num_hidden_layers", int(num_hidden_layers))


def _is_json_like(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str, type(None), list, tuple, dict))


def object_path(obj: Any) -> str:
    return "{}:{}".format(obj.__module__, obj.__qualname__)


def import_object(path: str) -> Any:
    module_name, _, qualname = path.partition(":")
    module = importlib.import_module(module_name)
    target = module
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def resolve_architecture_spec(model: nn.Module) -> ArchitectureSpec:
    spec = ArchitectureSpec()
    if not spec.matches(model):
        raise TypeError(
            "Unsupported model {}. v1 only supports decoder-only Llama-like layouts.".format(
                type(model).__name__
            )
        )
    return spec
