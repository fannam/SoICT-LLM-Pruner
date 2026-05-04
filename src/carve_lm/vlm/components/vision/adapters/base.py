from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from ...language.adapters import object_path


@dataclass(frozen=True)
class VisionAttentionProjectionBundle:
    qkv: nn.Linear
    proj: nn.Linear


@dataclass(frozen=True)
class VisionMLPProjectionBundle:
    gate_proj: nn.Linear | None = None
    up_proj: nn.Linear | None = None
    down_proj: nn.Linear | None = None
    linear_fc1: nn.Linear | None = None
    linear_fc2: nn.Linear | None = None

    def input_projections(self) -> tuple[tuple[str, nn.Linear], ...]:
        projections = []
        if self.gate_proj is not None:
            projections.append(("gate_proj", self.gate_proj))
        if self.up_proj is not None:
            projections.append(("up_proj", self.up_proj))
        if not projections and self.linear_fc1 is not None:
            projections.append(("linear_fc1", self.linear_fc1))
        return tuple(projections)

    def output_projection(self) -> tuple[str, nn.Linear]:
        if self.down_proj is not None:
            return "down_proj", self.down_proj
        if self.linear_fc2 is not None:
            return "linear_fc2", self.linear_fc2
        raise AttributeError("Vision MLP bundle does not expose an output projection.")

    @property
    def intermediate_size(self) -> int:
        return int(self.output_projection()[1].in_features)


class BaseVisionAdapter:
    """Adapter contract for vision-transformer blocks inside a VLM."""

    def __init__(self, name: str, model_cls: type | tuple[type, ...] | None = None):
        self.name = name
        self.model_cls = model_cls

    def matches(self, model: nn.Module) -> bool:
        if self.model_cls is not None and isinstance(model, self.model_cls):
            return True
        try:
            visual = self.get_visual_model(model)
            blocks = self.get_blocks(model)
            if len(blocks) == 0:
                return False
            block = blocks[0]
            self.get_attention_projections(block)
            self.get_mlp_projections(block)
            getattr(visual, "merger", None)
        except Exception:
            return False
        return True

    def ensure_supported(self, model: nn.Module) -> None:
        if not self.matches(model):
            raise TypeError(
                "Model must match the vision adapter contract for '{}'.".format(self.name)
            )

    def get_visual_model(self, model: nn.Module) -> nn.Module:
        return model.model.visual

    def get_blocks(self, model: nn.Module):
        return self.get_visual_model(model).blocks

    def set_blocks(self, model: nn.Module, blocks) -> None:
        self.ensure_supported(model)
        self.get_visual_model(model).blocks = nn.ModuleList(list(blocks))

    def get_patch_embed_projection(self, model: nn.Module) -> nn.Module | None:
        patch_embed = getattr(self.get_visual_model(model), "patch_embed", None)
        if patch_embed is None:
            return None
        return getattr(patch_embed, "proj", None)

    def set_patch_embed_projection(self, model: nn.Module, projection: nn.Module) -> None:
        patch_embed = getattr(self.get_visual_model(model), "patch_embed", None)
        if patch_embed is None:
            raise AttributeError("Visual model does not expose patch_embed.")
        patch_embed.proj = projection

    def get_norm1(self, block: nn.Module) -> nn.Module:
        return block.norm1

    def get_norm2(self, block: nn.Module) -> nn.Module:
        return block.norm2

    def get_attention_module(self, block: nn.Module) -> nn.Module:
        return block.attn

    def get_mlp_module(self, block: nn.Module) -> nn.Module:
        return block.mlp

    def get_attention_projections(self, block: nn.Module) -> VisionAttentionProjectionBundle:
        attention = self.get_attention_module(block)
        return VisionAttentionProjectionBundle(qkv=attention.qkv, proj=attention.proj)

    def get_mlp_projections(self, block: nn.Module) -> VisionMLPProjectionBundle:
        mlp = self.get_mlp_module(block)
        return VisionMLPProjectionBundle(
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
        )

    def get_mlp_input_projections(self, block: nn.Module) -> tuple[tuple[str, nn.Linear], ...]:
        projections = self.get_mlp_projections(block).input_projections()
        if not projections:
            raise AttributeError("Vision MLP does not expose input projections.")
        return projections

    def get_mlp_output_projection(self, block: nn.Module) -> tuple[str, nn.Linear]:
        return self.get_mlp_projections(block).output_projection()

    def get_mlp_intermediate_size(self, block: nn.Module) -> int:
        return self.get_mlp_projections(block).intermediate_size

    def set_attention_projection(
        self,
        block: nn.Module,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        setattr(self.get_attention_module(block), projection_name, projection)

    def set_mlp_projection(
        self,
        block: nn.Module,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        setattr(self.get_mlp_module(block), projection_name, projection)

    def hidden_size(self, model: nn.Module, block: nn.Module | None = None) -> int:
        if block is None:
            block = self.get_blocks(model)[0]
        return int(self.get_attention_projections(block).proj.out_features)

    def num_attention_heads(self, model: nn.Module, block: nn.Module | None = None) -> int:
        if block is None:
            block = self.get_blocks(model)[0]
        attention = self.get_attention_module(block)
        num_heads = getattr(attention, "num_heads", None)
        if num_heads is None:
            vision_config = getattr(getattr(model, "config", None), "vision_config", None)
            num_heads = getattr(vision_config, "num_heads", None)
        if num_heads is None:
            vision_config = getattr(getattr(model, "config", None), "vision_config", None)
            num_heads = getattr(vision_config, "num_attention_heads", None)
        if num_heads is None:
            raise ValueError("Unable to infer vision attention head count.")
        return int(num_heads)

    def head_dim(self, model: nn.Module, block: nn.Module | None = None) -> int:
        if block is None:
            block = self.get_blocks(model)[0]
        attention = self.get_attention_module(block)
        head_dim = getattr(attention, "head_dim", None)
        if head_dim is not None:
            return int(head_dim)
        hidden_size = self.hidden_size(model, block)
        num_heads = self.num_attention_heads(model, block)
        if hidden_size % num_heads != 0:
            raise ValueError("vision hidden_size must be divisible by num_attention_heads.")
        return hidden_size // num_heads

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
                if not key.startswith("_")
            }
        raise TypeError("Unsupported config type {}.".format(type(config).__name__))

    def patch_attention_metadata(
        self,
        attention_module: nn.Module,
        *,
        num_heads: int,
        head_dim: int,
        hidden_size: int | None = None,
    ) -> None:
        if hasattr(attention_module, "num_heads"):
            attention_module.num_heads = int(num_heads)
        if hasattr(attention_module, "head_dim"):
            attention_module.head_dim = int(head_dim)
        if hidden_size is not None and hasattr(attention_module, "hidden_size"):
            attention_module.hidden_size = int(hidden_size)

    def patch_hidden_size(self, model: nn.Module, hidden_size: int) -> None:
        self.ensure_supported(model)
        self.patch_vision_config(model, "hidden_size", hidden_size)

    def patch_num_attention_heads(self, model: nn.Module, num_heads: int) -> None:
        self.ensure_supported(model)
        self.patch_vision_config(model, "num_heads", num_heads)
        self.patch_vision_config(model, "num_attention_heads", num_heads)

    def patch_intermediate_size(self, model: nn.Module, intermediate_size: int) -> None:
        self.ensure_supported(model)
        self.patch_vision_config(model, "intermediate_size", intermediate_size)

    def patch_num_hidden_layers(self, model: nn.Module, num_hidden_layers: int) -> None:
        self.ensure_supported(model)
        self.patch_vision_config(model, "depth", num_hidden_layers)
        self.patch_vision_config(model, "num_hidden_layers", num_hidden_layers)

    def patch_vision_config(self, model: nn.Module, attr: str, value: int) -> None:
        vision_config = getattr(getattr(model, "config", None), "vision_config", None)
        if vision_config is not None and hasattr(vision_config, attr):
            setattr(vision_config, attr, int(value))

    def metadata(self, model: nn.Module) -> dict[str, Any]:
        block = self.get_blocks(model)[0]
        return {
            "hidden_size": self.hidden_size(model, block),
            "num_attention_heads": self.num_attention_heads(model, block),
            "head_dim": self.head_dim(model, block),
        }
