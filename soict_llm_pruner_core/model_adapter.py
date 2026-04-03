from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch.nn as nn
from transformers import LlamaForCausalLM, PreTrainedModel, Qwen2ForCausalLM

from utils import AttentionPasser, FeedForwardPasser, IdentityLayer


@dataclass(frozen=True)
class AttentionProjectionBundle:
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear


@dataclass(frozen=True)
class MLPProjectionBundle:
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear


class BaseModelAdapter(ABC):
    """
    Contract between shared pruning logic and a concrete model family.

    Each model family should provide its own adapter. If two families happen to share
    the same internal layout, they may inherit from a shared helper adapter, but the
    registration should still happen through explicit per-model adapter classes.
    """

    def __init__(
        self,
        name: str,
        model_cls: type[PreTrainedModel],
        supported_components: Sequence[str] | None = None,
    ):
        self.name = name
        self.model_cls = model_cls
        self.supported_components = tuple(
            supported_components
            or (
                "attention_heads",
                "attention_groups",
                "mlp_neurons",
                "embedding_channels",
                "attention_layers",
                "mlp_layers",
                "decoder_blocks",
            )
        )

    def matches(self, model: nn.Module) -> bool:
        return isinstance(model, self.model_cls)

    def ensure_supported(self, model: nn.Module) -> None:
        if not self.matches(model):
            raise TypeError(
                "Model must be an instance of {} for adapter '{}'.".format(
                    self.model_cls.__name__,
                    self.name,
                )
            )

    def clone_config(self, model: PreTrainedModel):
        self.ensure_supported(model)
        return copy.deepcopy(model.config)

    def instantiate_model(
        self,
        config,
        device: str | None = None,
        dtype=None,
    ) -> PreTrainedModel:
        model = self.model_cls(config)
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)
        return model

    @abstractmethod
    def get_layers(self, model: PreTrainedModel):
        raise NotImplementedError

    @abstractmethod
    def get_embed_tokens(self, model: PreTrainedModel) -> nn.Embedding:
        raise NotImplementedError

    @abstractmethod
    def get_lm_head(self, model: PreTrainedModel) -> nn.Module | None:
        raise NotImplementedError

    @abstractmethod
    def get_final_norm(self, model: PreTrainedModel) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_input_layernorm(self, layer: nn.Module) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_post_attention_layernorm(self, layer: nn.Module) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_attention_module(self, layer: nn.Module) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def set_attention_module(self, layer: nn.Module, module: nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_mlp_module(self, layer: nn.Module) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def set_mlp_module(self, layer: nn.Module, module: nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_attention_projections(self, layer: nn.Module) -> AttentionProjectionBundle:
        raise NotImplementedError

    @abstractmethod
    def get_mlp_projections(self, layer: nn.Module) -> MLPProjectionBundle:
        raise NotImplementedError

    @abstractmethod
    def set_attention_projection(
        self,
        layer: nn.Module,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_mlp_projection(
        self,
        layer: nn.Module,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        raise NotImplementedError

    def get_moe_experts(self, layer: nn.Module) -> tuple[nn.Module, ...]:
        return ()

    def available_components(self, model: PreTrainedModel | None = None) -> tuple[str, ...]:
        components = set(self.supported_components)
        if model is not None:
            self.ensure_supported(model)
            layers = self.get_layers(model)
            if any(self.get_moe_experts(layer) for layer in layers):
                components.add("experts")
        return tuple(sorted(components))

    @staticmethod
    def set_num_attention_heads(config, value: int) -> None:
        config.num_attention_heads = value

    @staticmethod
    def set_num_key_value_heads(config, value: int) -> None:
        setattr(config, "num_key_value_heads", value)

    @staticmethod
    def set_hidden_size(config, value: int) -> None:
        config.hidden_size = value

    @staticmethod
    def set_intermediate_size(config, value: int) -> None:
        config.intermediate_size = value

    @staticmethod
    def set_num_hidden_layers(config, value: int) -> None:
        config.num_hidden_layers = value

    @staticmethod
    def set_head_dim(config, value: int) -> None:
        setattr(config, "head_dim", value)

    @staticmethod
    def make_identity_attention() -> nn.Module:
        return AttentionPasser()

    @staticmethod
    def make_identity_mlp() -> nn.Module:
        return FeedForwardPasser()

    @staticmethod
    def make_identity_decoder_layer() -> nn.Module:
        return IdentityLayer()


class DecoderModelAdapter(BaseModelAdapter):
    """
    Optional helper for decoder-only architectures that follow the common Hugging Face
    layout `model.model.layers[*].{self_attn,mlp,...}`.

    Use this only when the target model genuinely follows this layout. Other model
    families should implement `BaseModelAdapter` directly.
    """

    def get_layers(self, model: PreTrainedModel):
        self.ensure_supported(model)
        return model.model.layers

    def get_embed_tokens(self, model: PreTrainedModel) -> nn.Embedding:
        self.ensure_supported(model)
        return model.model.embed_tokens

    def get_lm_head(self, model: PreTrainedModel) -> nn.Module | None:
        self.ensure_supported(model)
        return getattr(model, "lm_head", None)

    def get_final_norm(self, model: PreTrainedModel) -> nn.Module:
        self.ensure_supported(model)
        return model.model.norm

    def get_input_layernorm(self, layer: nn.Module) -> nn.Module:
        return layer.input_layernorm

    def get_post_attention_layernorm(self, layer: nn.Module) -> nn.Module:
        return layer.post_attention_layernorm

    def get_attention_module(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def set_attention_module(self, layer: nn.Module, module: nn.Module) -> None:
        layer.self_attn = module

    def get_mlp_module(self, layer: nn.Module) -> nn.Module:
        return layer.mlp

    def set_mlp_module(self, layer: nn.Module, module: nn.Module) -> None:
        layer.mlp = module

    def get_attention_projections(self, layer: nn.Module) -> AttentionProjectionBundle:
        attention = self.get_attention_module(layer)
        return AttentionProjectionBundle(
            q_proj=attention.q_proj,
            k_proj=attention.k_proj,
            v_proj=attention.v_proj,
            o_proj=attention.o_proj,
        )

    def get_mlp_projections(self, layer: nn.Module) -> MLPProjectionBundle:
        mlp = self.get_mlp_module(layer)
        return MLPProjectionBundle(
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
        )

    def set_attention_projection(
        self,
        layer: nn.Module,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        setattr(self.get_attention_module(layer), projection_name, projection)

    def set_mlp_projection(
        self,
        layer: nn.Module,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        setattr(self.get_mlp_module(layer), projection_name, projection)

    def get_moe_experts(self, layer: nn.Module) -> tuple[nn.Module, ...]:
        experts = getattr(self.get_mlp_module(layer), "experts", None)
        if experts is None:
            return ()
        if isinstance(experts, nn.ModuleList):
            return tuple(experts)
        if isinstance(experts, dict):
            return tuple(experts.values())
        if isinstance(experts, (list, tuple)):
            return tuple(experts)
        return ()


class LlamaModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(name="llama", model_cls=LlamaForCausalLM)


class Qwen2ModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(name="qwen2", model_cls=Qwen2ForCausalLM)


_MODEL_ADAPTERS: list[BaseModelAdapter] = []


def register_model_adapter(adapter: BaseModelAdapter) -> BaseModelAdapter:
    if any(existing.name == adapter.name for existing in _MODEL_ADAPTERS):
        raise KeyError("Model adapter '{}' is already registered.".format(adapter.name))
    _MODEL_ADAPTERS.append(adapter)
    return adapter


def get_model_adapter(name: str) -> BaseModelAdapter:
    for adapter in _MODEL_ADAPTERS:
        if adapter.name == name:
            return adapter
    available = ", ".join(adapter.name for adapter in _MODEL_ADAPTERS) or "<empty>"
    raise KeyError("Unknown model adapter '{}'. Available: {}.".format(name, available))


def registered_model_adapters() -> tuple[BaseModelAdapter, ...]:
    return tuple(_MODEL_ADAPTERS)


def resolve_model_adapter(
    model: PreTrainedModel,
    adapter: str | BaseModelAdapter | None = None,
) -> BaseModelAdapter:
    if adapter is None:
        for candidate in reversed(_MODEL_ADAPTERS):
            if candidate.matches(model):
                return candidate
        available = ", ".join(candidate.name for candidate in _MODEL_ADAPTERS) or "<empty>"
        raise TypeError(
            "No registered model adapter matches model type {}. Registered adapters: {}.".format(
                type(model).__name__,
                available,
            )
        )

    resolved = get_model_adapter(adapter) if isinstance(adapter, str) else adapter
    resolved.ensure_supported(model)
    return resolved


register_model_adapter(LlamaModelAdapter())
register_model_adapter(Qwen2ModelAdapter())
