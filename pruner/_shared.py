from __future__ import annotations

import copy
from typing import Dict, Set

from transformers import PreTrainedModel

from soict_llm_pruner_core import BaseModelAdapter, resolve_model_adapter


class _BasePruner:
    def __init__(
        self,
        model: PreTrainedModel,
        device: str = "cuda",
        model_adapter: BaseModelAdapter | str | None = None,
    ):
        self.adapter = resolve_model_adapter(model, model_adapter)
        self.model = model
        self.device = device


class _BaseLayerPruner(_BasePruner):
    """
    Prune the lowest-importance attention or MLP layers.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        device: str = "cuda",
        model_adapter: BaseModelAdapter | str | None = None,
    ):
        super().__init__(model=model, device=device, model_adapter=model_adapter)
        setattr(self.model.config, "attention_layer_to_prune", [])
        setattr(self.model.config, "mlp_layer_to_prune", [])

    @staticmethod
    def _get_keep_indices(scores, keep_k: int) -> Set[int]:
        if keep_k < 0:
            raise ValueError("keep_k must be non-negative")
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        return set(ranked_indices[:keep_k])

    def prune(self, importance_scores: Dict[str, list], prune_counts: Dict[str, int]):
        pruned_model = copy.deepcopy(self.model)
        layers = self.adapter.get_layers(pruned_model)
        attention_scores = importance_scores.get("attention", [])
        mlp_scores = importance_scores.get("mlp", [])
        num_layers = len(layers)

        attention_prune_count = prune_counts.get("attention", 0)
        mlp_prune_count = prune_counts.get("mlp", 0)
        if not 0 <= attention_prune_count <= num_layers:
            raise ValueError("attention prune count must be between 0 and {}".format(num_layers))
        if not 0 <= mlp_prune_count <= num_layers:
            raise ValueError("mlp prune count must be between 0 and {}".format(num_layers))
        if attention_scores and len(attention_scores) != num_layers:
            raise ValueError("attention importance must contain {} scores".format(num_layers))
        if mlp_scores and len(mlp_scores) != num_layers:
            raise ValueError("mlp importance must contain {} scores".format(num_layers))
        if attention_prune_count and not attention_scores:
            raise ValueError("attention importance scores are required to prune attention layers")
        if mlp_prune_count and not mlp_scores:
            raise ValueError("mlp importance scores are required to prune MLP layers")

        attention_layers_to_prune = set()
        if attention_scores:
            keep_attention = self._get_keep_indices(attention_scores, num_layers - attention_prune_count)
            attention_layers_to_prune = set(range(num_layers)) - keep_attention
            for layer_idx in attention_layers_to_prune:
                self.adapter.set_attention_module(
                    layers[layer_idx],
                    self.adapter.make_identity_attention(),
                )
        pruned_model.config.attention_layer_to_prune = sorted(attention_layers_to_prune)

        mlp_layers_to_prune = set()
        if mlp_scores:
            keep_mlp = self._get_keep_indices(mlp_scores, num_layers - mlp_prune_count)
            mlp_layers_to_prune = set(range(num_layers)) - keep_mlp
            for layer_idx in mlp_layers_to_prune:
                self.adapter.set_mlp_module(
                    layers[layer_idx],
                    self.adapter.make_identity_mlp(),
                )
        pruned_model.config.mlp_layer_to_prune = sorted(mlp_layers_to_prune)

        print("Pruned attention layers: {}".format(attention_layers_to_prune))
        print("Pruned MLP layers: {}".format(mlp_layers_to_prune))
        return pruned_model


class _BaseBlockPruner(_BasePruner):
    """
    Perform depth pruning by removing contiguous decoder blocks.
    """

    def __init__(
        self,
        original_model: PreTrainedModel,
        device: str = "cpu",
        model_adapter: BaseModelAdapter | str | None = None,
    ):
        super().__init__(model=original_model, device=device, model_adapter=model_adapter)
        self.original_model = original_model
        self.original_num_layers = len(self.adapter.get_layers(original_model))

    def _select_non_overlapping_blocks(
        self,
        block_importance: list,
        num_blocks_to_prune: int,
        block_size: int,
    ) -> tuple[list[int], list[int]]:
        sorted_starts = sorted(
            range(len(block_importance)),
            key=lambda start_idx: block_importance[start_idx],
        )
        selected_starts = []
        occupied_layers: set[int] = set()

        for start_idx in sorted_starts:
            candidate_layers = range(start_idx, start_idx + block_size)
            if any(layer_idx in occupied_layers for layer_idx in candidate_layers):
                continue
            selected_starts.append(start_idx)
            occupied_layers.update(candidate_layers)
            if len(selected_starts) == num_blocks_to_prune:
                break

        if len(selected_starts) != num_blocks_to_prune:
            raise ValueError(
                "Unable to select {} non-overlapping blocks of size {} from {} layers.".format(
                    num_blocks_to_prune,
                    block_size,
                    self.original_num_layers,
                )
            )

        return sorted(selected_starts), sorted(occupied_layers)

    def prune(self, block_importance: list, num_block_to_prune: int, block_size: int = 1):
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if block_size > self.original_num_layers:
            raise ValueError(
                "block_size must be less than or equal to {}".format(self.original_num_layers)
            )

        expected_scores = self.original_num_layers - block_size + 1
        if len(block_importance) != expected_scores:
            raise ValueError(
                "block_importance must contain {} scores for block_size={}".format(
                    expected_scores,
                    block_size,
                )
            )
        if not 1 <= num_block_to_prune <= expected_scores:
            raise ValueError(
                "num_block_to_prune must be in [1, {}] for block_size={}".format(
                    expected_scores,
                    block_size,
                )
            )

        block_start_indices, layers_to_prune = self._select_non_overlapping_blocks(
            block_importance=block_importance,
            num_blocks_to_prune=num_block_to_prune,
            block_size=block_size,
        )

        pruned_model = copy.deepcopy(self.original_model)
        layers = self.adapter.get_layers(pruned_model)
        for layer_idx in reversed(layers_to_prune):
            del layers[layer_idx]

        self.adapter.set_num_hidden_layers(pruned_model.config, len(layers))
        pruned_model.config.block_start_indices_to_prune = block_start_indices
        pruned_model.config.block_layers_to_prune = layers_to_prune
        pruned_model.config.pruned_block_size = block_size
        return pruned_model
