import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import defaultdict
import gc
import copy
from transformers import LlamaForCausalLM, PreTrainedModel, Qwen2ForCausalLM
from utils import AttentionPasser, FeedForwardPasser

class Llama3LayerPruner:
    """
    Prunes lowest-importance attention or MLP layers.
    """
    def __init__(self, model: LlamaForCausalLM, device):
        assert isinstance(model, LlamaForCausalLM), "Model must be LlamaForCausalLM"
        self.model = model
        self.device = device
        self.dtype = self.model.dtype
        setattr(self.model.config, 'attention_layer_to_prune', [])
        setattr(self.model.config, 'mlp_layer_to_prune', [])

    def _get_keep_indices(self, scores, keep_k):
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return set(idxs[:keep_k])

    def prune(self, importance_scores: dict, prune_counts: dict):
        """
        Args:
            importance_scores (dict): importance scores for attention and MLP layers
            prune_counts (dict): number of layers to prune for attention and MLP
        Returns:
            Pruned model
        """
        pruned_model = copy.deepcopy(self.model)
        attn_scores = importance_scores.get('attention', [])
        mlp_scores = importance_scores.get('mlp', [])
        n = len(attn_scores)
        keep_attn = self._get_keep_indices(attn_scores, n - prune_counts.get('attention', 0))
        to_prune_attn = set(range(n)) - keep_attn
        for i in to_prune_attn:
            pruned_model.model.layers[i].self_attn = AttentionPasser()
        pruned_model.config.attention_layer_to_prune = sorted(to_prune_attn)
        keep_mlp = self._get_keep_indices(mlp_scores, n - prune_counts.get('mlp', 0))
        to_prune_mlp = set(range(n)) - keep_mlp
        for i in to_prune_mlp:
            pruned_model.model.layers[i].mlp = FeedForwardPasser()
        pruned_model.config.mlp_layer_to_prune = sorted(to_prune_mlp)
        print(f"Pruned attention layers: {to_prune_attn}")
        print(f"Pruned MLP layers: {to_prune_mlp}")
        return pruned_model

class Qwen2LayerPruner:
    """
    Prunes lowest-importance attention or MLP layers.
    """
    def __init__(self, model: PreTrainedModel, device="cuda"):
        assert isinstance(model, Qwen2ForCausalLM), "Model must be Qwen2ForCausalLM"
        self.model = model
        self.device = device
        setattr(self.model.config, 'attention_layer_to_prune', [])
        setattr(self.model.config, 'mlp_layer_to_prune', [])

    def _get_keep_indices(self, scores, keep_k):
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return set(idxs[:keep_k])

    def prune(self, importance_scores: dict, prune_counts: dict):
        """
        Args:
            importance_scores (dict): importance scores for attention and MLP layers
            prune_counts (dict): number of layers to prune for attention and MLP
        Returns:
            Pruned model
        """
        pruned_model = copy.deepcopy(self.model)
        attn_scores = importance_scores.get('attention', [])
        mlp_scores = importance_scores.get('mlp', [])
        n = len(attn_scores)
        keep_attn = self._get_keep_indices(attn_scores, n - prune_counts.get('attention', 0))
        to_prune_attn = set(range(n)) - keep_attn
        for i in to_prune_attn:
            pruned_model.model.layers[i].self_attn = AttentionPasser()
        pruned_model.config.attention_layer_to_prune = sorted(to_prune_attn)
        keep_mlp = self._get_keep_indices(mlp_scores, n - prune_counts.get('mlp', 0))
        to_prune_mlp = set(range(n)) - keep_mlp
        for i in to_prune_mlp:
            pruned_model.model.layers[i].mlp = FeedForwardPasser()
        pruned_model.config.mlp_layer_to_prune = sorted(to_prune_mlp)
        print(f"Pruned attention layers: {to_prune_attn}")
        print(f"Pruned MLP layers: {to_prune_mlp}")
        return pruned_model