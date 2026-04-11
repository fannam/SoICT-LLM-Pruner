from __future__ import annotations

import torch.nn as nn
from transformers import PreTrainedModel

from .base import (
    AttentionProjectionBundle,
    BaseModelAdapter,
    MLPProjectionBundle,
)


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


class GenericDecoderModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(name="decoder_layout", model_cls=nn.Module)

    def matches(self, model: nn.Module) -> bool:
        try:
            layers = model.model.layers
            if len(layers) == 0:
                return False
            first_layer = layers[0]
            getattr(first_layer, "self_attn")
            getattr(first_layer, "mlp")
            getattr(first_layer, "input_layernorm")
            getattr(first_layer, "post_attention_layernorm")
            getattr(model.model, "embed_tokens")
            getattr(model.model, "norm")
        except Exception:
            return False
        return True

    def ensure_supported(self, model: nn.Module) -> None:
        if not self.matches(model):
            raise TypeError(
                "Model must match the decoder-layout contract for adapter '{}'.".format(
                    self.name
                )
            )
