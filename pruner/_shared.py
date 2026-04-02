from __future__ import annotations

import copy
from typing import Dict, Set, Type

from transformers import PreTrainedModel

from utils import AttentionPasser, FeedForwardPasser


class _BasePruner:
    model_cls: Type[PreTrainedModel] = PreTrainedModel
    model_name = "PreTrainedModel"

    def __init__(self, model: PreTrainedModel, device: str = "cuda"):
        if not isinstance(model, self.model_cls):
            raise TypeError("Model must be {}".format(self.model_name))
        self.model = model
        self.device = device


class _BaseLayerPruner(_BasePruner):
    """
    Prune the lowest-importance attention or MLP layers.
    """

    def __init__(self, model: PreTrainedModel, device: str = "cuda"):
        super().__init__(model=model, device=device)
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
        attention_scores = importance_scores.get("attention", [])
        mlp_scores = importance_scores.get("mlp", [])
        num_layers = len(pruned_model.model.layers)

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
                pruned_model.model.layers[layer_idx].self_attn = AttentionPasser()
        pruned_model.config.attention_layer_to_prune = sorted(attention_layers_to_prune)

        mlp_layers_to_prune = set()
        if mlp_scores:
            keep_mlp = self._get_keep_indices(mlp_scores, num_layers - mlp_prune_count)
            mlp_layers_to_prune = set(range(num_layers)) - keep_mlp
            for layer_idx in mlp_layers_to_prune:
                pruned_model.model.layers[layer_idx].mlp = FeedForwardPasser()
        pruned_model.config.mlp_layer_to_prune = sorted(mlp_layers_to_prune)

        print("Pruned attention layers: {}".format(attention_layers_to_prune))
        print("Pruned MLP layers: {}".format(mlp_layers_to_prune))
        return pruned_model


class _BaseBlockPruner(_BasePruner):
    """
    Perform depth pruning by removing entire decoder layers.
    """

    def __init__(self, original_model: PreTrainedModel, device: str = "cpu"):
        super().__init__(model=original_model, device=device)
        self.original_model = original_model
        self.original_num_layers = original_model.config.num_hidden_layers

    def prune(self, block_importance: list, num_block_to_prune: int):
        if len(block_importance) != self.original_num_layers:
            raise ValueError(
                "block_importance must contain {} scores".format(self.original_num_layers)
            )
        if not 1 <= num_block_to_prune < self.original_num_layers:
            raise ValueError(
                "num_block_to_prune must be in [1, {}]".format(self.original_num_layers - 1)
            )

        sorted_indices = sorted(
            range(self.original_num_layers),
            key=lambda layer_idx: block_importance[layer_idx],
        )
        layers_to_prune = sorted_indices[:num_block_to_prune]

        pruned_model = copy.deepcopy(self.original_model)
        layers = pruned_model.model.layers
        for layer_idx in sorted(layers_to_prune, reverse=True):
            del layers[layer_idx]

        pruned_model.config.num_hidden_layers = len(layers)
        pruned_model.config.block_layers_to_prune = layers_to_prune
        return pruned_model
