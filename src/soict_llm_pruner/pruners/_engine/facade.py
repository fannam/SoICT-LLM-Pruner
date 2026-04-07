from __future__ import annotations

from ...adapters import resolve_model_adapter
from .config import DepthLayerConfig, WidthChannelConfig, WidthGroupConfig
from .discovery import discover_blockwise, discover_channelwise
from .estimation import estimate_scores
from .executors import (
    apply_blockwise_plan,
    apply_channelwise_plan,
    apply_layerwise_plan,
    build_layerwise_plan,
    select_blockwise_plan,
    select_channelwise_plan,
)
from .manifest import build_manifest, load_pruned_result, save_pruned_result
from .types import DiscoveryContext, PruningPlan, PruningResult


class _BaseStructuredPruner:
    config_cls = None
    canonical_name = None
    legacy_mode = None

    def __init__(self, model, config, device: str = "cpu", model_adapter=None):
        self.model = model
        self.config = config
        self.device = device
        self.adapter = resolve_model_adapter(model, model_adapter)
        self.spec = self.adapter
        self._last_context: DiscoveryContext | None = None
        self._last_scores: dict[str, float] | None = None
        self._last_plan: PruningPlan | None = None
        self._last_result: PruningResult | None = None

    def save_pruned(self, output_dir, result: PruningResult | None = None):
        result = result or self._last_result
        if result is None:
            raise ValueError("No pruning result is available to save.")
        return save_pruned_result(output_dir, result)

    @classmethod
    def load_pruned(cls, output_dir, device: str | None = None, dtype=None) -> PruningResult:
        return load_pruned_result(cls, output_dir, device=device, dtype=dtype)


class WidthGroupPruner(_BaseStructuredPruner):
    config_cls = WidthGroupConfig
    canonical_name = "width.group"
    legacy_mode = "block"

    def discover(self, example_batch=None) -> DiscoveryContext:
        self._last_context = discover_blockwise(self.model, self.adapter, example_batch=example_batch)
        return self._last_context

    def estimate(self, dataloader) -> dict[str, float]:
        if self._last_context is None:
            self.discover()
        self._last_scores = estimate_scores(
            self.model,
            self.adapter,
            self._last_context,
            self.config.estimator,
            dataloader,
            device=self.device,
        )
        return self._last_scores

    def select(self, scores: dict[str, float] | None = None) -> PruningPlan:
        scores = scores or self._last_scores
        if scores is None:
            raise ValueError("scores must be provided or estimated first.")
        if self._last_context is None:
            self.discover()
        self._last_plan = select_blockwise_plan(self._last_context, scores, self.config)
        return self._last_plan

    def apply(self, plan: PruningPlan | None = None) -> PruningResult:
        plan = plan or self._last_plan
        if plan is None:
            raise ValueError("plan must be provided or selected first.")
        if self._last_context is None:
            self.discover()
        pruned_model = apply_blockwise_plan(
            self.model,
            self.adapter,
            self._last_context,
            plan,
            self.config,
        )
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

    def run(self, example_batch, dataloader) -> PruningResult:
        self.discover(example_batch)
        self.estimate(dataloader)
        self.select()
        return self.apply()


class WidthChannelPruner(_BaseStructuredPruner):
    config_cls = WidthChannelConfig
    canonical_name = "width.channel"
    legacy_mode = "channel"

    def discover(self, example_batch=None) -> DiscoveryContext:
        self._last_context = discover_channelwise(self.model, self.adapter, example_batch=example_batch)
        return self._last_context

    def estimate(self, dataloader) -> dict[str, float]:
        if self._last_context is None:
            self.discover()
        self._last_scores = estimate_scores(
            self.model,
            self.adapter,
            self._last_context,
            self.config.estimator,
            dataloader,
            device=self.device,
        )
        return self._last_scores

    def select(self, scores: dict[str, float] | None = None) -> PruningPlan:
        scores = scores or self._last_scores
        if scores is None:
            raise ValueError("scores must be provided or estimated first.")
        if self._last_context is None:
            self.discover()
        self._last_plan = select_channelwise_plan(self._last_context, scores, self.config)
        return self._last_plan

    def apply(self, plan: PruningPlan | None = None) -> PruningResult:
        plan = plan or self._last_plan
        if plan is None:
            raise ValueError("plan must be provided or selected first.")
        if self._last_context is None:
            self.discover()
        pruned_model = apply_channelwise_plan(
            self.model,
            self.adapter,
            self._last_context,
            plan,
            self.config,
        )
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

    def run(self, example_batch, dataloader) -> PruningResult:
        self.discover(example_batch)
        self.estimate(dataloader)
        self.select()
        return self.apply()


class DepthLayerPruner(_BaseStructuredPruner):
    config_cls = DepthLayerConfig
    canonical_name = "depth.layer"
    legacy_mode = "layer"

    def _layer_context(self) -> DiscoveryContext:
        block_context = discover_blockwise(self.model, self.adapter)
        return DiscoveryContext(
            mode="layer",
            family_key=block_context.family_key,
            model_class_path=block_context.model_class_path,
            config_class_path=block_context.config_class_path,
            base_config=block_context.base_config,
            groups=tuple(),
            layer_metadata=block_context.layer_metadata,
            hidden_size=block_context.hidden_size,
            num_attention_heads=block_context.num_attention_heads,
            num_key_value_heads=block_context.num_key_value_heads,
            head_dim=block_context.head_dim,
            metadata={"discovery_kind": "adapter_rules"},
        )

    def apply(self, plan: PruningPlan | None = None) -> PruningResult:
        if self._last_context is None:
            self._last_context = self._layer_context()
        plan = plan or build_layerwise_plan(self.model, self.adapter, self.config)
        self._last_plan = plan
        pruned_model = apply_layerwise_plan(
            self.model,
            self.adapter,
            plan,
            self.config,
        )
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

    def run(self) -> PruningResult:
        return self.apply()


__all__ = [
    "DepthLayerPruner",
    "WidthChannelPruner",
    "WidthGroupPruner",
]
