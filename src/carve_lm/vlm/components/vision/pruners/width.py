from __future__ import annotations

from collections.abc import Mapping

import torch

from ...language.pruners._engine.types import (
    DiscoveryContext,
    LayerMetadata,
    PruningGroup,
    PruningPlan,
    PruningResult,
    SliceSpec,
)
from ...merger.adapters import resolve_model_adapter as resolve_merger_adapter
from ..adapters import BaseVisionAdapter, resolve_model_adapter
from ..core import PRUNER_REGISTRY
from ._utils import (
    append_pruning_history,
    clone_or_share,
    slice_linear,
    slice_norm,
    slice_projection_output,
    sorted_topk_indices,
)
from .config import WidthChannelConfig
from .manifest import build_manifest, load_pruned_result, save_pruned_result


@PRUNER_REGISTRY.register("width")
class WidthPruner:
    """Element-level width pruner for Qwen2.5-VL vision blocks."""

    def __init__(
        self,
        model,
        device: str = "cpu",
        model_adapter: BaseVisionAdapter | str | None = None,
    ):
        self.model = model
        self.device = device
        self.adapter = resolve_model_adapter(model, model_adapter)

    def prune_attention_heads(
        self,
        head_importance: Mapping[int, torch.Tensor],
        target_num_heads: int,
        *,
        clone_model: bool = True,
    ):
        blocks = self.adapter.get_blocks(self.model)
        if target_num_heads <= 0:
            raise ValueError("target_num_heads must be positive.")
        if len(blocks) == 0:
            raise ValueError("Model does not contain any visual blocks.")

        original_num_heads = self.adapter.num_attention_heads(self.model, blocks[0])
        if target_num_heads >= original_num_heads:
            raise ValueError("target_num_heads must be smaller than the original number of heads.")

        self._validate_block_scores(head_importance, len(blocks), original_num_heads, "head_importance")
        pruned_model = clone_or_share(self.model, clone_model).to(self.device)

        for block_idx, block in enumerate(self.adapter.get_blocks(pruned_model)):
            num_heads = self.adapter.num_attention_heads(pruned_model, block)
            head_dim = self.adapter.head_dim(pruned_model, block)
            hidden_size = num_heads * head_dim
            scores = head_importance[block_idx]
            keep_heads = sorted_topk_indices(scores, target_num_heads, device=self.device)
            keep_rows = [
                row
                for head_idx in keep_heads
                for row in range(head_idx * head_dim, (head_idx + 1) * head_dim)
            ]
            qkv_rows = [
                offset + row
                for offset in (0, hidden_size, 2 * hidden_size)
                for row in keep_rows
            ]

            attention = self.adapter.get_attention_module(block)
            projections = self.adapter.get_attention_projections(block)
            self.adapter.set_attention_projection(
                block,
                "qkv",
                slice_linear(projections.qkv, out_indices=qkv_rows, in_indices=None),
            )
            self.adapter.set_attention_projection(
                block,
                "proj",
                slice_linear(projections.proj, out_indices=None, in_indices=keep_rows),
            )
            self.adapter.patch_attention_metadata(
                attention,
                num_heads=target_num_heads,
                head_dim=head_dim,
            )

        self.adapter.patch_num_attention_heads(pruned_model, target_num_heads)
        self._append_history(
            pruned_model,
            "vision.width.attention_heads",
            target_num_heads=target_num_heads,
        )
        return pruned_model

    def prune_mlp_neurons(
        self,
        neuron_importance: Mapping[int, torch.Tensor],
        target_num_neurons: int,
        *,
        clone_model: bool = True,
    ):
        blocks = self.adapter.get_blocks(self.model)
        if target_num_neurons <= 0:
            raise ValueError("target_num_neurons must be positive.")
        if len(blocks) == 0:
            raise ValueError("Model does not contain any visual blocks.")

        original_intermediate = self.adapter.get_mlp_intermediate_size(blocks[0])
        if target_num_neurons >= original_intermediate:
            raise ValueError("target_num_neurons must be smaller than the original intermediate size.")

        self._validate_block_scores(neuron_importance, len(blocks), original_intermediate, "neuron_importance")
        pruned_model = clone_or_share(self.model, clone_model).to(self.device)

        for block_idx, block in enumerate(self.adapter.get_blocks(pruned_model)):
            scores = neuron_importance[block_idx]
            keep_neurons = sorted_topk_indices(scores, target_num_neurons, device=self.device)
            mlp = self.adapter.get_mlp_projections(block)
            for projection_name, projection in mlp.input_projections():
                self.adapter.set_mlp_projection(
                    block,
                    projection_name,
                    slice_linear(projection, out_indices=keep_neurons, in_indices=None),
                )
            output_name, output_projection = mlp.output_projection()
            self.adapter.set_mlp_projection(
                block,
                output_name,
                slice_linear(output_projection, out_indices=None, in_indices=keep_neurons),
            )

        self.adapter.patch_intermediate_size(pruned_model, target_num_neurons)
        self._append_history(
            pruned_model,
            "vision.width.mlp_neurons",
            target_num_neurons=target_num_neurons,
        )
        return pruned_model

    def prune_hidden_channels(
        self,
        hidden_importance=None,
        target_hidden_size: int | None = None,
        *,
        keep_channel_indices: list[int] | tuple[int, ...] | None = None,
        clone_model: bool = True,
        record_history: bool = True,
    ):
        blocks = self.adapter.get_blocks(self.model)
        if len(blocks) == 0:
            raise ValueError("Model does not contain any visual blocks.")

        first_block = blocks[0]
        num_heads = self.adapter.num_attention_heads(self.model, first_block)
        old_head_dim = self.adapter.head_dim(self.model, first_block)
        old_hidden_size = num_heads * old_head_dim

        if keep_channel_indices is None:
            if target_hidden_size is None:
                raise TypeError("target_hidden_size or keep_channel_indices must be provided.")
            if target_hidden_size <= 0 or target_hidden_size >= old_hidden_size:
                raise ValueError("target_hidden_size must be in [1, {}).".format(old_hidden_size))
            if target_hidden_size % num_heads != 0:
                raise ValueError("target_hidden_size must preserve whole per-head channel bundles.")
            bundle_scores = self._bundle_scores(hidden_importance, old_head_dim, num_heads)
            keep_count = target_hidden_size // num_heads
            keep_channel_indices = sorted_topk_indices(bundle_scores, keep_count, device=self.device)
        else:
            keep_channel_indices = sorted({int(index) for index in keep_channel_indices})
            if not keep_channel_indices:
                raise ValueError("At least one hidden channel bundle must be kept.")
            if min(keep_channel_indices) < 0 or max(keep_channel_indices) >= old_head_dim:
                raise ValueError("keep_channel_indices must be valid per-head channel offsets.")

        keep_channel_indices = list(keep_channel_indices)
        new_head_dim = len(keep_channel_indices)
        new_hidden_size = num_heads * new_head_dim
        residual_indices = self._residual_indices(
            num_heads=num_heads,
            head_dim=old_head_dim,
            channel_indices=keep_channel_indices,
        )
        qkv_out_indices = [
            offset + index
            for offset in (0, old_hidden_size, 2 * old_hidden_size)
            for index in residual_indices
        ]

        pruned_model = clone_or_share(self.model, clone_model).to(self.device)

        patch_projection = self.adapter.get_patch_embed_projection(pruned_model)
        if patch_projection is not None:
            self.adapter.set_patch_embed_projection(
                pruned_model,
                slice_projection_output(patch_projection, residual_indices),
            )

        for block in self.adapter.get_blocks(pruned_model):
            block.norm1 = slice_norm(self.adapter.get_norm1(block), residual_indices)
            block.norm2 = slice_norm(self.adapter.get_norm2(block), residual_indices)

            attention = self.adapter.get_attention_module(block)
            projections = self.adapter.get_attention_projections(block)
            self.adapter.set_attention_projection(
                block,
                "qkv",
                slice_linear(
                    projections.qkv,
                    out_indices=qkv_out_indices,
                    in_indices=residual_indices,
                ),
            )
            self.adapter.set_attention_projection(
                block,
                "proj",
                slice_linear(
                    projections.proj,
                    out_indices=residual_indices,
                    in_indices=residual_indices,
                ),
            )
            self.adapter.patch_attention_metadata(
                attention,
                num_heads=num_heads,
                head_dim=new_head_dim,
                hidden_size=new_hidden_size,
            )

            mlp = self.adapter.get_mlp_projections(block)
            for projection_name, projection in mlp.input_projections():
                self.adapter.set_mlp_projection(
                    block,
                    projection_name,
                    slice_linear(projection, out_indices=None, in_indices=residual_indices),
                )
            output_name, output_projection = mlp.output_projection()
            self.adapter.set_mlp_projection(
                block,
                output_name,
                slice_linear(output_projection, out_indices=residual_indices, in_indices=None),
            )

        self._slice_merger_input_boundary(pruned_model, residual_indices, old_hidden_size)
        self.adapter.patch_hidden_size(pruned_model, new_hidden_size)
        if record_history:
            self._append_history(
                pruned_model,
                "vision.width.hidden_channels",
                target_hidden_size=new_hidden_size,
                selected_channel_indices=keep_channel_indices,
                selected_residual_indices=residual_indices,
            )
        return pruned_model

    def _slice_merger_input_boundary(
        self,
        model,
        residual_indices: list[int],
        old_hidden_size: int,
    ) -> None:
        try:
            merger_adapter = resolve_merger_adapter(model, self.adapter.name)
        except Exception:
            return

        for merger in merger_adapter.get_mergers(model):
            ln_q = merger_adapter.get_ln_q(merger)
            if ln_q is not None:
                norm_width = merger_adapter.input_norm_width(merger)
                if norm_width == old_hidden_size:
                    norm_indices = residual_indices
                else:
                    merge_factor = merger_adapter.merge_factor(model, merger)
                    norm_indices = [
                        merge_idx * old_hidden_size + index
                        for merge_idx in range(merge_factor)
                        for index in residual_indices
                    ]
                merger_adapter.set_ln_q(merger, slice_norm(ln_q, norm_indices))

            projections = merger_adapter.get_projections(merger)
            merge_factor = merger_adapter.merge_factor(model, merger)
            if hasattr(merger, "hidden_size"):
                merger.hidden_size = len(residual_indices) * merge_factor
            grouped_input_indices = [
                merge_idx * old_hidden_size + index
                for merge_idx in range(merge_factor)
                for index in residual_indices
            ]
            merger_adapter.set_projection(
                merger,
                "fc1",
                slice_linear(projections.fc1, out_indices=None, in_indices=grouped_input_indices),
            )

    @staticmethod
    def _residual_indices(
        *,
        num_heads: int,
        head_dim: int,
        channel_indices: list[int],
    ) -> list[int]:
        return [
            head_idx * head_dim + channel_idx
            for head_idx in range(num_heads)
            for channel_idx in sorted(channel_indices)
        ]

    @staticmethod
    def _validate_block_scores(
        scores: Mapping[int, torch.Tensor],
        num_blocks: int,
        expected_numel: int,
        score_name: str,
    ) -> None:
        missing_blocks = [block_idx for block_idx in range(num_blocks) if block_idx not in scores]
        if missing_blocks:
            raise ValueError("{} is missing scores for blocks {}.".format(score_name, missing_blocks))
        for block_idx in range(num_blocks):
            tensor = scores[block_idx] if torch.is_tensor(scores[block_idx]) else torch.as_tensor(scores[block_idx])
            if tensor.ndim != 1 or tensor.numel() != expected_numel:
                raise ValueError(
                    "{}[{}] must contain {} 1D scores, received shape {}.".format(
                        score_name,
                        block_idx,
                        expected_numel,
                        tuple(tensor.shape),
                    )
                )

    def _bundle_scores(self, hidden_importance, old_head_dim: int, num_heads: int) -> torch.Tensor:
        if hidden_importance is None:
            raise TypeError("hidden_importance must be provided when selecting by target_hidden_size.")
        if isinstance(hidden_importance, Mapping):
            values = [
                value if torch.is_tensor(value) else torch.as_tensor(value)
                for value in hidden_importance.values()
            ]
            global_scores = torch.stack(values, dim=0).sum(dim=0)
        else:
            global_scores = (
                hidden_importance
                if torch.is_tensor(hidden_importance)
                else torch.as_tensor(hidden_importance)
            )

        expected_hidden = old_head_dim * num_heads
        if global_scores.ndim != 1 or global_scores.numel() != expected_hidden:
            raise ValueError(
                "hidden_importance must resolve to {} 1D scores, received shape {}.".format(
                    expected_hidden,
                    tuple(global_scores.shape),
                )
            )

        return torch.stack(
            [
                global_scores[
                    self._residual_indices(
                        num_heads=num_heads,
                        head_dim=old_head_dim,
                        channel_indices=[channel_idx],
                    )
                ].sum()
                for channel_idx in range(old_head_dim)
            ]
        )

    @staticmethod
    def _append_history(model, strategy: str, **metadata) -> None:
        history = list(getattr(model.config, "pruning_history", []))
        history.append({"strategy": strategy, **metadata})
        setattr(model.config, "pruning_history", history)
        setattr(model.config, "last_pruning_strategy", strategy)


@PRUNER_REGISTRY.register("width.channel")
class WidthChannelPruner(WidthPruner):
    config_cls = WidthChannelConfig
    canonical_name = "width.channel"
    legacy_mode = "channel"

    def __init__(
        self,
        model,
        config: WidthChannelConfig | None = None,
        device: str = "cpu",
        model_adapter: BaseVisionAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        self.config = config or WidthChannelConfig(pruning_ratio=0.0)
        self._last_context: DiscoveryContext | None = None
        self._last_scores: dict[str, float] | None = None
        self._last_plan: PruningPlan | None = None
        self._last_result: PruningResult | None = None

    def discover(self, example_batch=None) -> DiscoveryContext:
        del example_batch
        blocks = self.adapter.get_blocks(self.model)
        if len(blocks) == 0:
            raise ValueError("Model does not contain any visual blocks.")

        first = _build_layer_metadata(self.model, self.adapter, 0)
        for block_idx in range(1, len(blocks)):
            metadata = _build_layer_metadata(self.model, self.adapter, block_idx)
            if metadata.hidden_size != first.hidden_size:
                raise ValueError("Vision channel pruning requires uniform hidden size.")
            if metadata.num_attention_heads != first.num_attention_heads:
                raise ValueError("Vision channel pruning requires uniform num_heads.")
            if metadata.head_dim != first.head_dim:
                raise ValueError("Vision channel pruning requires uniform head_dim.")

        layer_metadata = tuple(
            _build_layer_metadata(self.model, self.adapter, block_idx)
            for block_idx in range(len(blocks))
        )
        groups = []
        for channel_idx in range(first.head_dim):
            residual_indices = self._residual_indices(
                num_heads=first.num_attention_heads,
                head_dim=first.head_dim,
                channel_indices=[channel_idx],
            )
            dependent_slices = []
            patch_projection = self.adapter.get_patch_embed_projection(self.model)
            if patch_projection is not None:
                dependent_slices.append(
                    SliceSpec(
                        module_path=self.adapter.module_path(self.model, patch_projection),
                        param_name="weight",
                        axis=0,
                        indices=tuple(residual_indices),
                        role="patch_embed_out",
                    )
                )
            for block_idx in range(len(blocks)):
                block = blocks[block_idx]
                attention = self.adapter.get_attention_projections(block)
                mlp = self.adapter.get_mlp_projections(block)
                output_name, output_projection = mlp.output_projection()
                qkv_out_indices = [
                    offset + index
                    for offset in (0, first.hidden_size, 2 * first.hidden_size)
                    for index in residual_indices
                ]
                dependent_slices.extend(
                    (
                        SliceSpec(
                            module_path=self.adapter.module_path(self.model, self.adapter.get_norm1(block)),
                            param_name="weight",
                            axis=0,
                            indices=tuple(residual_indices),
                            role="norm1",
                        ),
                        SliceSpec(
                            module_path=self.adapter.module_path(self.model, self.adapter.get_norm2(block)),
                            param_name="weight",
                            axis=0,
                            indices=tuple(residual_indices),
                            role="norm2",
                        ),
                        SliceSpec(
                            module_path=self.adapter.module_path(self.model, attention.qkv),
                            param_name="weight",
                            axis=0,
                            indices=tuple(qkv_out_indices),
                            role="qkv_out",
                        ),
                        SliceSpec(
                            module_path=self.adapter.module_path(self.model, attention.qkv),
                            param_name="weight",
                            axis=1,
                            indices=tuple(residual_indices),
                            role="qkv_in",
                        ),
                        SliceSpec(
                            module_path=self.adapter.module_path(self.model, attention.proj),
                            param_name="weight",
                            axis=0,
                            indices=tuple(residual_indices),
                            role="proj_out",
                        ),
                        SliceSpec(
                            module_path=self.adapter.module_path(self.model, attention.proj),
                            param_name="weight",
                            axis=1,
                            indices=tuple(residual_indices),
                            role="proj_in",
                        ),
                    )
                )
                for projection_name, projection in mlp.input_projections():
                    dependent_slices.append(
                        SliceSpec(
                            module_path=self.adapter.module_path(self.model, projection),
                            param_name="weight",
                            axis=1,
                            indices=tuple(residual_indices),
                            role="{}_in".format(projection_name),
                        )
                    )
                dependent_slices.append(
                    SliceSpec(
                        module_path=self.adapter.module_path(self.model, output_projection),
                        param_name="weight",
                        axis=0,
                        indices=tuple(residual_indices),
                        role="{}_out".format(output_name),
                    )
                )

            groups.append(
                PruningGroup(
                    group_id="vision.channel.bundle{}".format(channel_idx),
                    family="channel",
                    layer_idx=None,
                    local_idx=channel_idx,
                    members=("vision_hidden_stream", "attention", "mlp", "merger_input"),
                    dependent_slices=tuple(dependent_slices),
                    width=len(residual_indices),
                    metadata={
                        "residual_indices": tuple(residual_indices),
                        "channel_idx": channel_idx,
                    },
                )
            )

        self._last_context = DiscoveryContext(
            mode="channel",
            family_key=self.adapter.name,
            model_class_path=self.adapter.model_class_path(self.model),
            config_class_path=self.adapter.config_class_path(self.model),
            base_config=self.adapter.config_to_dict(self.model),
            groups=tuple(groups),
            layer_metadata=layer_metadata,
            hidden_size=first.hidden_size,
            num_attention_heads=first.num_attention_heads,
            num_key_value_heads=first.num_attention_heads,
            head_dim=first.head_dim,
            metadata={
                "component": "vision",
                "channel_group_kind": "per_head_channel_bundle",
            },
        )
        return self._last_context

    def estimate(self, dataloader=None) -> dict[str, float]:
        if self._last_context is None:
            self.discover()
        from ..estimators import create_estimator

        estimator = create_estimator(
            self.config.estimator.name,
            self.model,
            device=self.device,
            model_adapter=self.adapter,
        )
        estimate_hidden_channels = getattr(estimator, "estimate_hidden_channels", None)
        if estimate_hidden_channels is None:
            raise TypeError(
                "Estimator '{}' does not expose estimate_hidden_channels().".format(
                    self.config.estimator.name
                )
            )

        if dataloader is None:
            hidden_scores = estimate_hidden_channels(**self.config.estimator.kwargs)
        else:
            hidden_scores = estimate_hidden_channels(dataloader, **self.config.estimator.kwargs)

        first = self._last_context.layer_metadata[0]
        bundle_scores = self._bundle_scores(
            hidden_scores,
            old_head_dim=first.head_dim,
            num_heads=first.num_attention_heads,
        )
        self._last_scores = {
            group.group_id: float(bundle_scores[group.local_idx].item())
            for group in self._last_context.groups
        }
        return self._last_scores

    def select(self, scores: dict[str, float] | None = None) -> PruningPlan:
        scores = scores or self._last_scores
        if scores is None:
            raise ValueError("scores must be provided or estimated first.")
        if self._last_context is None:
            self.discover()

        context = self._last_context
        group_width = context.groups[0].width if context.groups else context.num_attention_heads
        raw_keep_hidden = max(1, context.hidden_size - int(context.hidden_size * self.config.pruning_ratio))
        round_to = self.config.round_to or context.num_attention_heads
        if round_to % group_width != 0:
            raise ValueError(
                "round_to={} must be a multiple of the channel bundle width {}.".format(
                    round_to,
                    group_width,
                )
            )
        raw_keep_hidden -= raw_keep_hidden % round_to
        if raw_keep_hidden <= 0:
            raw_keep_hidden = round_to
        keep_group_count = max(1, raw_keep_hidden // group_width)
        ranked = sorted(
            context.groups,
            key=lambda group: (-scores.get(group.group_id, 0.0), group.group_id),
        )
        selected_groups = sorted(ranked[:keep_group_count], key=lambda group: group.local_idx)
        selected_ids = tuple(group.group_id for group in selected_groups)
        selected_id_set = set(selected_ids)
        pruned_ids = tuple(group.group_id for group in context.groups if group.group_id not in selected_id_set)
        selected_channel_indices = [group.local_idx for group in selected_groups]
        selected_residual_indices = self._residual_indices(
            num_heads=context.num_attention_heads,
            head_dim=context.head_dim,
            channel_indices=selected_channel_indices,
        )

        self._last_plan = PruningPlan(
            mode="channel",
            selected_group_ids=selected_ids,
            pruned_group_ids=pruned_ids,
            scores={group.group_id: float(scores.get(group.group_id, 0.0)) for group in context.groups},
            metadata={
                "target_hidden_size": len(selected_residual_indices),
                "round_to": round_to,
                "selected_channel_count": len(selected_groups),
                "selected_channel_indices": selected_channel_indices,
                "selected_residual_indices": selected_residual_indices,
            },
        )
        return self._last_plan

    def apply(self, plan: PruningPlan | None = None) -> PruningResult:
        plan = plan or self._last_plan
        if plan is None:
            raise ValueError("plan must be provided or selected first.")
        if self._last_context is None:
            self.discover()

        pruned_model = self.prune_hidden_channels(
            keep_channel_indices=plan.metadata["selected_channel_indices"],
            clone_model=self.config.clone_model,
            record_history=False,
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


def _build_layer_metadata(model, adapter: BaseVisionAdapter, block_idx: int) -> LayerMetadata:
    block = adapter.get_blocks(model)[block_idx]
    attention = adapter.get_attention_projections(block)
    num_heads = adapter.num_attention_heads(model, block)
    head_dim = adapter.head_dim(model, block)
    return LayerMetadata(
        layer_idx=block_idx,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=adapter.get_mlp_intermediate_size(block),
        hidden_size=attention.proj.out_features,
    )


__all__ = ["WidthChannelPruner", "WidthPruner"]
