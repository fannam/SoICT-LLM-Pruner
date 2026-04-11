from __future__ import annotations

import copy
import importlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import torch.nn as nn
from transformers import PreTrainedModel

from ..core.identity import AttentionPasser, FeedForwardPasser, IdentityLayer


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

    def clone_model(self, model: nn.Module) -> nn.Module:
        self.ensure_supported(model)
        return copy.deepcopy(model)

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

    def module_path(self, model: nn.Module, module: nn.Module) -> str:
        if module is model:
            return ""
        for name, candidate in model.named_modules():
            if candidate is module:
                return name
        raise KeyError("Unable to resolve module path for {}.".format(type(module).__name__))

    def model_class_path(self, model: nn.Module) -> str:
        self.ensure_supported(model)
        return object_path(model.__class__)

    def config_class_path(self, model: nn.Module) -> str:
        self.ensure_supported(model)
        return object_path(model.config.__class__)

    def config_to_dict(self, model: nn.Module) -> dict[str, Any]:
        self.ensure_supported(model)
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

    def _infer_head_dim(self, model: nn.Module, layer: nn.Module) -> int:
        config_head_dim = getattr(getattr(model, "config", None), "head_dim", None)
        if config_head_dim is not None:
            return int(config_head_dim)

        attention = self.get_attention_projections(layer)
        num_heads = int(
            getattr(
                self.get_attention_module(layer),
                "num_heads",
                getattr(model.config, "num_attention_heads", 0),
            )
        )
        if num_heads <= 0:
            raise ValueError("Unable to infer attention head_dim.")
        if attention.q_proj.out_features % num_heads != 0:
            raise ValueError("q_proj.out_features must be divisible by num_heads.")
        return attention.q_proj.out_features // num_heads

    def get_attention_handles(
        self,
        model: nn.Module,
        layer: nn.Module,
    ) -> AttentionHandles:
        attention_module = self.get_attention_module(layer)
        projections = self.get_attention_projections(layer)
        num_heads = int(
            getattr(
                attention_module,
                "num_heads",
                getattr(model.config, "num_attention_heads", 0),
            )
        )
        if num_heads <= 0:
            raise ValueError("Unable to infer num_heads for {}.".format(type(attention_module).__name__))
        num_key_value_heads = int(
            getattr(
                attention_module,
                "num_key_value_heads",
                getattr(model.config, "num_key_value_heads", num_heads),
            )
        )
        head_dim = int(
            getattr(
                attention_module,
                "head_dim",
                self._infer_head_dim(model, layer),
            )
        )
        return AttentionHandles(
            module=attention_module,
            q_proj=projections.q_proj,
            k_proj=projections.k_proj,
            v_proj=projections.v_proj,
            o_proj=projections.o_proj,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

    def get_mlp_handles(self, layer: nn.Module) -> MLPHandles:
        mlp_module = self.get_mlp_module(layer)
        projections = self.get_mlp_projections(layer)
        return MLPHandles(
            module=mlp_module,
            gate_proj=projections.gate_proj,
            up_proj=projections.up_proj,
            down_proj=projections.down_proj,
        )

    def get_layer_handles(self, model: nn.Module, layer_idx: int) -> LayerHandles:
        self.ensure_supported(model)
        layer = self.get_layers(model)[layer_idx]
        return LayerHandles(
            layer=layer,
            input_layernorm=self.get_input_layernorm(layer),
            post_attention_layernorm=self.get_post_attention_layernorm(layer),
            attention=self.get_attention_handles(model, layer),
            mlp=self.get_mlp_handles(layer),
        )

    def hidden_size(self, model: nn.Module) -> int:
        self.ensure_supported(model)
        return int(model.config.hidden_size)

    def head_dim(self, model: nn.Module) -> int:
        self.ensure_supported(model)
        layers = self.get_layers(model)
        if len(layers) == 0:
            raise ValueError("Model does not contain any decoder layers.")
        return self.get_attention_handles(model, layers[0]).head_dim

    def family_for_model(self, model: nn.Module) -> str:
        self.ensure_supported(model)
        return self.name

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
        self.ensure_supported(model)
        setattr(model.config, "hidden_size", int(hidden_size))

    def patch_num_hidden_layers(self, model: nn.Module, num_hidden_layers: int) -> None:
        self.ensure_supported(model)
        setattr(model.config, "num_hidden_layers", int(num_hidden_layers))

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
