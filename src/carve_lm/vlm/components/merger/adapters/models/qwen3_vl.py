from __future__ import annotations

import torch.nn as nn

from ..base import BaseMergerAdapter, MergerProjectionBundle

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


class Qwen3VLMergerAdapter(BaseMergerAdapter):
    def __init__(self):
        super().__init__(name="qwen3_vl", model_cls=Qwen3VLForConditionalGeneration)

    def get_mergers(self, model: nn.Module) -> tuple[nn.Module, ...]:
        visual = model.model.visual
        mergers = [visual.merger]
        mergers.extend(list(getattr(visual, "deepstack_merger_list", ())))
        return tuple(mergers)

    def get_ln_q(self, merger: nn.Module) -> nn.Module:
        return merger.norm

    def set_ln_q(self, merger: nn.Module, norm: nn.Module) -> None:
        merger.norm = norm

    def get_mlp(self, merger: nn.Module) -> nn.Module:
        return merger

    def get_projections(self, merger: nn.Module) -> MergerProjectionBundle:
        return MergerProjectionBundle(fc1=merger.linear_fc1, fc2=merger.linear_fc2)

    def set_projection(
        self,
        merger: nn.Module,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        if projection_name == "fc1":
            merger.linear_fc1 = projection
            return
        if projection_name == "fc2":
            merger.linear_fc2 = projection
            return
        raise KeyError("Unknown merger projection '{}'.".format(projection_name))

    def input_hidden_size(self, model: nn.Module, merger: nn.Module | None = None) -> int:
        vision_config = getattr(getattr(model, "config", None), "vision_config", None)
        hidden_size = getattr(vision_config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
        if merger is None:
            merger = self.get_merger(model)
        merge_factor = getattr(merger, "merge_factor", None)
        if merge_factor is None and hasattr(merger, "spatial_merge_size"):
            merge_factor = int(merger.spatial_merge_size) ** 2
        if merge_factor is None:
            raise ValueError("Unable to infer Qwen3-VL merger input hidden size without vision_config.")
        return int(self.get_projections(merger).fc1.in_features // int(merge_factor))

    def merge_factor(self, model: nn.Module, merger: nn.Module | None = None) -> int:
        if merger is None:
            merger = self.get_merger(model)
        vision_config = getattr(getattr(model, "config", None), "vision_config", None)
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", None)
        if spatial_merge_size is not None:
            return int(spatial_merge_size) ** 2
        merge_factor = getattr(merger, "merge_factor", None)
        if merge_factor is not None:
            return int(merge_factor)
        if hasattr(merger, "spatial_merge_size"):
            return int(merger.spatial_merge_size) ** 2
        input_hidden_size = self.input_hidden_size(model, merger)
        return int(self.get_projections(merger).fc1.in_features // input_hidden_size)
