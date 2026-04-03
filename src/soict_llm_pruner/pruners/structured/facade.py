from __future__ import annotations

from .config import BlockWiseConfig, ChannelWiseConfig, LayerWiseConfig
from .discovery import discover_blockwise, discover_channelwise
from .executors import (
    apply_blockwise_plan,
    apply_channelwise_plan,
    apply_layerwise_plan,
    build_layerwise_plan,
    select_blockwise_plan,
    select_channelwise_plan,
)
from .importance import estimate_importance
from .manifest import build_manifest, load_pruned_result, save_pruned_result
from .spec import resolve_architecture_spec
from .types import DiscoveryContext, PruningPlan, PruningResult


class _BaseStructuredPruner:
    config_cls = None
    mode = None

    def __init__(self, model, config, device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device
        self.spec = resolve_architecture_spec(model)
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


class StructuredBlockPruner(_BaseStructuredPruner):
    config_cls = BlockWiseConfig
    mode = "block"

    def discover(self, example_batch=None) -> DiscoveryContext:
        self._last_context = discover_blockwise(self.model, self.spec, example_batch=example_batch)
        return self._last_context

    def estimate(self, dataloader) -> dict[str, float]:
        if self._last_context is None:
            self.discover()
        self._last_scores = estimate_importance(
            self.model,
            self._last_context,
            self.config.importance,
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
            self.spec,
            self._last_context,
            plan,
            self.config,
        )
        result = PruningResult(
            model=pruned_model,
            context=self._last_context,
            plan=plan,
            manifest=build_manifest(
                mode=self.mode,
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


class StructuredChannelPruner(_BaseStructuredPruner):
    config_cls = ChannelWiseConfig
    mode = "channel"

    def discover(self, example_batch=None) -> DiscoveryContext:
        self._last_context = discover_channelwise(self.model, self.spec, example_batch=example_batch)
        return self._last_context

    def estimate(self, dataloader) -> dict[str, float]:
        if self._last_context is None:
            self.discover()
        self._last_scores = estimate_importance(
            self.model,
            self._last_context,
            self.config.importance,
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
            self.spec,
            self._last_context,
            plan,
            self.config,
        )
        result = PruningResult(
            model=pruned_model,
            context=self._last_context,
            plan=plan,
            manifest=build_manifest(
                mode=self.mode,
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


class StructuredLayerPruner(_BaseStructuredPruner):
    config_cls = LayerWiseConfig
    mode = "layer"

    def _layer_context(self) -> DiscoveryContext:
        block_context = discover_blockwise(self.model, self.spec)
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
            metadata={"discovery_kind": "static_rules"},
        )

    def apply(self, plan: PruningPlan | None = None) -> PruningResult:
        if self._last_context is None:
            self._last_context = self._layer_context()
        plan = plan or build_layerwise_plan(self.model, self.spec, self.config)
        self._last_plan = plan
        pruned_model = apply_layerwise_plan(
            self.model,
            self.spec,
            plan,
            self.config,
        )
        result = PruningResult(
            model=pruned_model,
            context=self._last_context,
            plan=plan,
            manifest=build_manifest(
                mode=self.mode,
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
    "StructuredBlockPruner",
    "StructuredChannelPruner",
    "StructuredLayerPruner",
]
