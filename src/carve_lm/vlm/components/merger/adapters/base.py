from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from ...language.adapters import object_path


@dataclass(frozen=True)
class MergerProjectionBundle:
    fc1: nn.Linear
    fc2: nn.Linear


class BaseMergerAdapter:
    """Adapter contract for VLM patch-merger modules."""

    def __init__(self, name: str, model_cls: type | tuple[type, ...] | None = None):
        self.name = name
        self.model_cls = model_cls

    def matches(self, model: nn.Module) -> bool:
        if self.model_cls is not None and isinstance(model, self.model_cls):
            return True
        try:
            merger = self.get_merger(model)
            self.get_ln_q(merger)
            self.get_projections(merger)
        except Exception:
            return False
        return True

    def ensure_supported(self, model: nn.Module) -> None:
        if not self.matches(model):
            raise TypeError(
                "Model must match the merger adapter contract for '{}'.".format(self.name)
            )

    def get_merger(self, model: nn.Module) -> nn.Module:
        return model.model.visual.merger

    def get_ln_q(self, merger: nn.Module) -> nn.Module:
        return merger.ln_q

    def get_mlp(self, merger: nn.Module) -> nn.Sequential:
        return merger.mlp

    def get_projections(self, merger: nn.Module) -> MergerProjectionBundle:
        mlp = self.get_mlp(merger)
        return MergerProjectionBundle(fc1=mlp[0], fc2=mlp[2])

    def set_projection(
        self,
        merger: nn.Module,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        if projection_name == "fc1":
            self.get_mlp(merger)[0] = projection
            return
        if projection_name == "fc2":
            self.get_mlp(merger)[2] = projection
            return
        raise KeyError("Unknown merger projection '{}'.".format(projection_name))

    def input_hidden_size(self, model: nn.Module, merger: nn.Module | None = None) -> int:
        if merger is None:
            merger = self.get_merger(model)
        ln_q = self.get_ln_q(merger)
        weight = getattr(ln_q, "weight", None)
        if weight is not None:
            return int(weight.numel())
        projections = self.get_projections(merger)
        merge_factor = self.merge_factor(model, merger)
        return int(projections.fc1.in_features // merge_factor)

    def merge_factor(self, model: nn.Module, merger: nn.Module | None = None) -> int:
        if merger is None:
            merger = self.get_merger(model)
        vision_config = getattr(getattr(model, "config", None), "vision_config", None)
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", None)
        if spatial_merge_size is not None:
            return int(spatial_merge_size) ** 2

        projections = self.get_projections(merger)
        input_hidden_size = self.input_hidden_size_from_ln_q(merger)
        if input_hidden_size <= 0 or projections.fc1.in_features % input_hidden_size != 0:
            raise ValueError("Unable to infer merger spatial merge factor.")
        return int(projections.fc1.in_features // input_hidden_size)

    def input_hidden_size_from_ln_q(self, merger: nn.Module) -> int:
        ln_q = self.get_ln_q(merger)
        weight = getattr(ln_q, "weight", None)
        if weight is None:
            return 0
        return int(weight.numel())

    def output_hidden_size(self, model: nn.Module, merger: nn.Module | None = None) -> int:
        if merger is None:
            merger = self.get_merger(model)
        return int(self.get_projections(merger).fc2.out_features)

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

    def config_to_dict(self, model: nn.Module) -> dict:
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

    def patch_output_hidden_size(self, model: nn.Module, hidden_size: int) -> None:
        self.ensure_supported(model)
        vision_config = getattr(getattr(model, "config", None), "vision_config", None)
        if vision_config is not None and hasattr(vision_config, "out_hidden_size"):
            vision_config.out_hidden_size = int(hidden_size)
