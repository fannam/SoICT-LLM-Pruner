from __future__ import annotations

import torch

from ...language.adapters import resolve_model_adapter as resolve_language_adapter
from ...language.pruners._engine.discovery import discover_channelwise
from ...language.pruners._engine.estimation import estimate_scores as estimate_language_scores
from ...language.pruners._engine.executors import apply_channelwise_plan, select_channelwise_plan
from ...language.pruners._engine.types import DiscoveryContext, PruningPlan, PruningResult
from ..adapters import BaseMergerAdapter, resolve_model_adapter
from ..core import PRUNER_REGISTRY
from ._utils import append_pruning_history, slice_linear
from .config import BridgeChannelConfig
from .manifest import build_manifest, load_pruned_result, save_pruned_result


@PRUNER_REGISTRY.register("width.bridge")
class BridgeChannelPruner:
    config_cls = BridgeChannelConfig
    canonical_name = "width.bridge"
    legacy_mode = "bridge"

    def __init__(
        self,
        model,
        config: BridgeChannelConfig | None = None,
        device: str = "cpu",
        model_adapter: BaseMergerAdapter | str | None = None,
        language_model_adapter=None,
    ):
        self.model = model
        self.config = config or BridgeChannelConfig(pruning_ratio=0.0)
        self.device = device
        self.adapter = resolve_model_adapter(model, model_adapter)
        language_adapter_name = language_model_adapter
        if language_adapter_name is None and isinstance(model_adapter, str):
            language_adapter_name = model_adapter
        self.language_adapter = resolve_language_adapter(model, language_adapter_name)
        self._last_context: DiscoveryContext | None = None
        self._last_scores: dict[str, float] | None = None
        self._last_plan: PruningPlan | None = None
        self._last_result: PruningResult | None = None

    def discover(self, example_batch=None) -> DiscoveryContext:
        language_context = discover_channelwise(
            self.model,
            self.language_adapter,
            example_batch=example_batch,
        )
        self._last_context = DiscoveryContext(
            mode="bridge",
            family_key=language_context.family_key,
            model_class_path=language_context.model_class_path,
            config_class_path=language_context.config_class_path,
            base_config=language_context.base_config,
            groups=language_context.groups,
            layer_metadata=language_context.layer_metadata,
            hidden_size=language_context.hidden_size,
            num_attention_heads=language_context.num_attention_heads,
            num_key_value_heads=language_context.num_key_value_heads,
            head_dim=language_context.head_dim,
            metadata={
                **language_context.metadata,
                "component": "merger",
                "bridge_target": "language_hidden_channels",
                "merger_adapter_name": self.adapter.name,
            },
        )
        return self._last_context

    def estimate(self, dataloader=None) -> dict[str, float]:
        if self._last_context is None:
            self.discover()

        language_context = DiscoveryContext(
            mode="channel",
            family_key=self._last_context.family_key,
            model_class_path=self._last_context.model_class_path,
            config_class_path=self._last_context.config_class_path,
            base_config=self._last_context.base_config,
            groups=self._last_context.groups,
            layer_metadata=self._last_context.layer_metadata,
            hidden_size=self._last_context.hidden_size,
            num_attention_heads=self._last_context.num_attention_heads,
            num_key_value_heads=self._last_context.num_key_value_heads,
            head_dim=self._last_context.head_dim,
            metadata=dict(self._last_context.metadata),
        )
        language_scores = estimate_language_scores(
            self.model,
            self.language_adapter,
            language_context,
            self.config.language_estimator,
            dataloader,
            device=self.device,
        )
        merger_scores = self._estimate_merger_output_scores(dataloader)

        combined_scores = {}
        for group in self._last_context.groups:
            residual_indices = list(group.metadata["residual_indices"])
            merger_score = float(merger_scores[residual_indices].mean().item())
            combined_scores[group.group_id] = (
                float(self.config.language_weight) * float(language_scores.get(group.group_id, 0.0))
                + float(self.config.merger_weight) * merger_score
            )
        self._last_scores = combined_scores
        return combined_scores

    def select(self, scores: dict[str, float] | None = None) -> PruningPlan:
        scores = scores or self._last_scores
        if scores is None:
            raise ValueError("scores must be provided or estimated first.")
        if self._last_context is None:
            self.discover()

        language_context = DiscoveryContext(
            mode="channel",
            family_key=self._last_context.family_key,
            model_class_path=self._last_context.model_class_path,
            config_class_path=self._last_context.config_class_path,
            base_config=self._last_context.base_config,
            groups=self._last_context.groups,
            layer_metadata=self._last_context.layer_metadata,
            hidden_size=self._last_context.hidden_size,
            num_attention_heads=self._last_context.num_attention_heads,
            num_key_value_heads=self._last_context.num_key_value_heads,
            head_dim=self._last_context.head_dim,
            metadata=dict(self._last_context.metadata),
        )
        plan = select_channelwise_plan(language_context, scores, self.config)
        self._last_plan = PruningPlan(
            mode="bridge",
            selected_group_ids=plan.selected_group_ids,
            pruned_group_ids=plan.pruned_group_ids,
            scores=plan.scores,
            metadata={
                **plan.metadata,
                "language_adapter_name": self.language_adapter.name,
                "merger_adapter_name": self.adapter.name,
            },
        )
        return self._last_plan

    def apply(self, plan: PruningPlan | None = None) -> PruningResult:
        if self._last_context is None:
            self.discover()
        plan = plan or self._last_plan
        if plan is None:
            raise ValueError("plan must be provided or selected first.")

        language_plan = PruningPlan(
            mode="channel",
            selected_group_ids=plan.selected_group_ids,
            pruned_group_ids=plan.pruned_group_ids,
            scores=plan.scores,
            metadata=dict(plan.metadata),
        )
        language_context = DiscoveryContext(
            mode="channel",
            family_key=self._last_context.family_key,
            model_class_path=self._last_context.model_class_path,
            config_class_path=self._last_context.config_class_path,
            base_config=self._last_context.base_config,
            groups=self._last_context.groups,
            layer_metadata=self._last_context.layer_metadata,
            hidden_size=self._last_context.hidden_size,
            num_attention_heads=self._last_context.num_attention_heads,
            num_key_value_heads=self._last_context.num_key_value_heads,
            head_dim=self._last_context.head_dim,
            metadata=dict(self._last_context.metadata),
        )
        pruned_model = apply_channelwise_plan(
            self.model,
            self.language_adapter,
            language_context,
            language_plan,
            self.config,
        )
        self._slice_merger_output(pruned_model, plan.metadata["selected_residual_indices"])
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

    def _estimate_merger_output_scores(self, dataloader=None) -> torch.Tensor:
        from ..estimators import create_estimator

        estimator = create_estimator(
            self.config.merger_estimator.name,
            self.model,
            device=self.device,
            model_adapter=self.adapter,
        )
        estimate_output = getattr(estimator, "estimate_output_channels", None)
        if estimate_output is None:
            raise TypeError(
                "Estimator '{}' does not expose estimate_output_channels().".format(
                    self.config.merger_estimator.name
                )
            )
        if dataloader is None:
            scores = estimate_output(**self.config.merger_estimator.kwargs)
        else:
            scores = estimate_output(dataloader, **self.config.merger_estimator.kwargs)
        if isinstance(scores, dict):
            scores = scores["merger_output_channels"]
        tensor = scores if torch.is_tensor(scores) else torch.as_tensor(scores)
        if tensor.ndim != 1:
            raise ValueError("Merger output scores must resolve to a 1D tensor.")
        return tensor

    def _slice_merger_output(self, model, selected_residual_indices: list[int]) -> None:
        for merger in self.adapter.get_mergers(model):
            fc2 = self.adapter.get_projections(merger).fc2
            self.adapter.set_projection(
                merger,
                "fc2",
                slice_linear(fc2, out_indices=list(selected_residual_indices), in_indices=None),
            )
        self.adapter.patch_output_hidden_size(model, len(selected_residual_indices))


__all__ = ["BridgeChannelPruner"]
