from __future__ import annotations

from ...language.pruners._engine.types import DiscoveryContext, LayerMetadata, PruningPlan, PruningResult
from ..adapters import BaseVisionAdapter, resolve_model_adapter
from ..core import PRUNER_REGISTRY
from ._utils import append_pruning_history, clone_or_share
from .config import DepthLayerConfig
from .manifest import build_manifest, load_pruned_result, save_pruned_result


@PRUNER_REGISTRY.register("depth.layer")
class DepthLayerPruner:
    config_cls = DepthLayerConfig
    canonical_name = "depth.layer"
    legacy_mode = "layer"

    def __init__(
        self,
        model,
        config: DepthLayerConfig | None = None,
        device: str = "cpu",
        model_adapter: BaseVisionAdapter | str | None = None,
    ):
        self.model = model
        self.device = device
        self.adapter = resolve_model_adapter(model, model_adapter)
        self.config = config or DepthLayerConfig(
            target_num_layers=len(self.adapter.get_blocks(model))
        )
        self._last_context: DiscoveryContext | None = None
        self._last_plan: PruningPlan | None = None
        self._last_result: PruningResult | None = None

    def discover(self, example_batch=None) -> DiscoveryContext:
        del example_batch
        blocks = self.adapter.get_blocks(self.model)
        if len(blocks) == 0:
            raise ValueError("Model does not contain any visual blocks.")

        layer_metadata = tuple(
            LayerMetadata(
                layer_idx=block_idx,
                num_attention_heads=self.adapter.num_attention_heads(self.model, block),
                num_key_value_heads=self.adapter.num_attention_heads(self.model, block),
                head_dim=self.adapter.head_dim(self.model, block),
                intermediate_size=self.adapter.get_mlp_projections(block).down_proj.in_features,
                hidden_size=self.adapter.hidden_size(self.model, block),
            )
            for block_idx, block in enumerate(blocks)
        )
        first = layer_metadata[0]
        self._last_context = DiscoveryContext(
            mode="layer",
            family_key=self.adapter.name,
            model_class_path=self.adapter.model_class_path(self.model),
            config_class_path=self.adapter.config_class_path(self.model),
            base_config=self.adapter.config_to_dict(self.model),
            groups=tuple(),
            layer_metadata=layer_metadata,
            hidden_size=first.hidden_size,
            num_attention_heads=first.num_attention_heads,
            num_key_value_heads=first.num_key_value_heads,
            head_dim=first.head_dim,
            metadata={"component": "vision", "keep_strategy": self.config.keep_strategy},
        )
        return self._last_context

    def select(self) -> PruningPlan:
        if self._last_context is None:
            self.discover()
        num_layers = len(self.adapter.get_blocks(self.model))
        target_num_layers = self.config.target_num_layers
        if target_num_layers > num_layers:
            raise ValueError(
                "target_num_layers={} exceeds the model depth {}.".format(
                    target_num_layers,
                    num_layers,
                )
            )

        selected_layer_indices = tuple(range(target_num_layers))
        pruned_layer_indices = tuple(range(target_num_layers, num_layers))
        self._last_plan = PruningPlan(
            mode="layer",
            selected_group_ids=tuple("vision.layer.{}".format(index) for index in selected_layer_indices),
            pruned_group_ids=tuple("vision.layer.{}".format(index) for index in pruned_layer_indices),
            scores={},
            metadata={
                "target_num_layers": target_num_layers,
                "selected_layer_indices": list(selected_layer_indices),
                "pruned_layer_indices": list(pruned_layer_indices),
            },
        )
        return self._last_plan

    def apply(self, plan: PruningPlan | None = None) -> PruningResult:
        if self._last_context is None:
            self.discover()
        plan = plan or self._last_plan or self.select()
        pruned_model = clone_or_share(self.model, self.config.clone_model).to(self.device)
        blocks = self.adapter.get_blocks(pruned_model)
        keep_count = int(plan.metadata["target_num_layers"])
        self.adapter.set_blocks(pruned_model, blocks[:keep_count])
        self.adapter.patch_num_hidden_layers(pruned_model, keep_count)
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

    def run(self) -> PruningResult:
        self.discover()
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


__all__ = ["DepthLayerPruner"]
