from __future__ import annotations

import torch

from ...language.pruners._engine.types import (
    DiscoveryContext,
    LayerMetadata,
    PruningGroup,
    PruningPlan,
    PruningResult,
    SliceSpec,
)
from ..adapters import BaseMergerAdapter, resolve_model_adapter
from ..core import PRUNER_REGISTRY
from ._utils import append_pruning_history, clone_or_share, slice_linear, sorted_topk_indices
from .config import WidthConfig
from .manifest import build_manifest, load_pruned_result, save_pruned_result


@PRUNER_REGISTRY.register("width")
class WidthPruner:
    config_cls = WidthConfig
    canonical_name = "width"
    legacy_mode = "intermediate"

    def __init__(
        self,
        model,
        config: WidthConfig | None = None,
        device: str = "cpu",
        model_adapter: BaseMergerAdapter | str | None = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.adapter = resolve_model_adapter(model, model_adapter)
        self._last_context: DiscoveryContext | None = None
        self._last_scores: dict[str, float] | None = None
        self._last_plan: PruningPlan | None = None
        self._last_result: PruningResult | None = None

    def prune_intermediate_channels(
        self,
        channel_importance,
        target_num_channels: int,
        *,
        clone_model: bool = True,
    ):
        if target_num_channels <= 0:
            raise ValueError("target_num_channels must be positive.")
        scores = self._coerce_intermediate_scores(channel_importance)
        if target_num_channels >= scores.numel():
            raise ValueError("target_num_channels must be smaller than the original intermediate size.")
        keep_indices = sorted_topk_indices(scores, target_num_channels, device=self.device)
        return self._apply_intermediate_indices(keep_indices, clone_model=clone_model)

    def discover(self, example_batch=None) -> DiscoveryContext:
        del example_batch
        mergers = self.adapter.get_mergers(self.model)
        merger = mergers[0]
        projections = self.adapter.get_projections(merger)
        intermediate_size = projections.fc2.in_features
        input_hidden_size = self.adapter.input_hidden_size(self.model, merger)
        output_hidden_size = self.adapter.output_hidden_size(self.model, merger)

        groups = []
        for channel_idx in range(intermediate_size):
            dependent_slices = []
            for merger in mergers:
                projections = self.adapter.get_projections(merger)
                dependent_slices.extend(
                    (
                        SliceSpec(
                            module_path=self.adapter.module_path(self.model, projections.fc1),
                            param_name="weight",
                            axis=0,
                            indices=(channel_idx,),
                            role="fc1_out",
                        ),
                        SliceSpec(
                            module_path=self.adapter.module_path(self.model, projections.fc2),
                            param_name="weight",
                            axis=1,
                            indices=(channel_idx,),
                            role="fc2_in",
                        ),
                    )
                )
            groups.append(
                PruningGroup(
                    group_id="merger.intermediate.{}".format(channel_idx),
                    family="intermediate",
                    layer_idx=None,
                    local_idx=channel_idx,
                    members=("merger.fc1", "merger.fc2"),
                    dependent_slices=tuple(dependent_slices),
                    metadata={"channel_idx": channel_idx},
                )
            )
        self._last_context = DiscoveryContext(
            mode="intermediate",
            family_key=self.adapter.name,
            model_class_path=self.adapter.model_class_path(self.model),
            config_class_path=self.adapter.config_class_path(self.model),
            base_config=self.adapter.config_to_dict(self.model),
            groups=tuple(groups),
            layer_metadata=(
                LayerMetadata(
                    layer_idx=0,
                    num_attention_heads=1,
                    num_key_value_heads=1,
                    head_dim=input_hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_size=output_hidden_size,
                ),
            ),
            hidden_size=output_hidden_size,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=input_hidden_size,
            metadata={"component": "merger"},
        )
        return self._last_context

    def estimate(self, dataloader=None) -> dict[str, float]:
        if self.config is None:
            raise ValueError("A WidthConfig is required for structured estimate().")
        if self._last_context is None:
            self.discover()

        from ..estimators import create_estimator

        estimator = create_estimator(
            self.config.estimator.name,
            self.model,
            device=self.device,
            model_adapter=self.adapter,
        )
        estimate_intermediate = getattr(estimator, "estimate_intermediate_channels", None)
        if estimate_intermediate is None:
            raise TypeError(
                "Estimator '{}' does not expose estimate_intermediate_channels().".format(
                    self.config.estimator.name
                )
            )
        if dataloader is None:
            scores = estimate_intermediate(**self.config.estimator.kwargs)
        else:
            scores = estimate_intermediate(dataloader, **self.config.estimator.kwargs)
        tensor = self._coerce_intermediate_scores(scores)
        self._last_scores = {
            group.group_id: float(tensor[group.local_idx].item())
            for group in self._last_context.groups
        }
        return self._last_scores

    def select(self, scores: dict[str, float] | None = None) -> PruningPlan:
        if self.config is None:
            raise ValueError("A WidthConfig is required for structured select().")
        scores = scores or self._last_scores
        if scores is None:
            raise ValueError("scores must be provided or estimated first.")
        if self._last_context is None:
            self.discover()

        context = self._last_context
        intermediate_size = context.layer_metadata[0].intermediate_size
        keep_count = max(
            1,
            intermediate_size - int(intermediate_size * self.config.pruning_ratio),
        )
        ranked = sorted(
            context.groups,
            key=lambda group: (-scores.get(group.group_id, 0.0), group.group_id),
        )
        selected_groups = sorted(ranked[:keep_count], key=lambda group: group.local_idx)
        selected_ids = tuple(group.group_id for group in selected_groups)
        selected_id_set = set(selected_ids)
        pruned_ids = tuple(group.group_id for group in context.groups if group.group_id not in selected_id_set)
        selected_indices = [group.local_idx for group in selected_groups]
        self._last_plan = PruningPlan(
            mode="intermediate",
            selected_group_ids=selected_ids,
            pruned_group_ids=pruned_ids,
            scores={group.group_id: float(scores.get(group.group_id, 0.0)) for group in context.groups},
            metadata={
                "target_num_channels": keep_count,
                "selected_intermediate_indices": selected_indices,
            },
        )
        return self._last_plan

    def apply(self, plan: PruningPlan | None = None) -> PruningResult:
        if self.config is None:
            raise ValueError("A WidthConfig is required for structured apply().")
        if self._last_context is None:
            self.discover()
        plan = plan or self._last_plan
        if plan is None:
            raise ValueError("plan must be provided or selected first.")

        pruned_model = self._apply_intermediate_indices(
            plan.metadata["selected_intermediate_indices"],
            clone_model=self.config.clone_model,
        )
        append_pruning_history(pruned_model, plan)
        result = PruningResult(
            model=pruned_model,
            context=self._last_context,
            plan=plan,
            manifest=build_manifest(
                mode=self.legacy_mode,
                canonical_pruner=self.canonical_name,
                adapter_name=self.adapter.name,
                config=self.config,
                context=self._last_context,
                plan=plan,
                pruned_model=pruned_model,
            ),
        )
        self._last_result = result
        return result

    def run(self, example_batch=None, dataloader=None) -> PruningResult:
        self.discover(example_batch)
        self.estimate(dataloader)
        self.select()
        return self.apply()

    def save_pruned(self, output_dir, result: PruningResult | None = None):
        result = result or self._last_result
        if result is None:
            raise ValueError("No pruning result is available to save.")
        return save_pruned_result(output_dir, result)

    @classmethod
    def load_pruned(cls, output_dir, device: str | None = None, dtype=None) -> PruningResult:
        return load_pruned_result(cls, output_dir, device=device, dtype=dtype)

    def _apply_intermediate_indices(self, keep_indices: list[int], *, clone_model: bool):
        pruned_model = clone_or_share(self.model, clone_model).to(self.device)
        for merger in self.adapter.get_mergers(pruned_model):
            projections = self.adapter.get_projections(merger)
            self.adapter.set_projection(
                merger,
                "fc1",
                slice_linear(projections.fc1, out_indices=keep_indices, in_indices=None),
            )
            self.adapter.set_projection(
                merger,
                "fc2",
                slice_linear(projections.fc2, out_indices=None, in_indices=keep_indices),
            )
        return pruned_model

    @staticmethod
    def _coerce_intermediate_scores(channel_importance) -> torch.Tensor:
        if isinstance(channel_importance, dict):
            channel_importance = channel_importance["merger_intermediate_channels"]
        tensor = channel_importance if torch.is_tensor(channel_importance) else torch.as_tensor(channel_importance)
        if tensor.ndim != 1:
            raise ValueError("channel_importance must resolve to a 1D tensor.")
        return tensor


__all__ = ["WidthPruner"]
