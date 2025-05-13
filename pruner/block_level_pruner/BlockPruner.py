import copy
from transformers import PreTrainedModel, LlamaForCausalLM, Qwen2ForCausalLM

class Llama3BlockPruner:
    """
    Performs depth pruning by removing entire decoder layers and updating model config.
    """
    def __init__(self, original_model: LlamaForCausalLM, device: str = 'cpu'):
        self.original_model = original_model
        self.device = device
        self.original_num_layers = original_model.config.num_hidden_layers

    def prune(self, block_importance: list, num_block_to_prune: int):
        """
        Remove the `num_block_to_prune` lowest-importance layers.

        Args:
            block_importance (list[float]): importance score per layer
            num_block_to_prune (int): how many layers to remove
        Returns:
            LlamaForCausalLM: a new model instance with layers removed and config updated
        """
        if not (1 <= num_block_to_prune < self.original_num_layers):
            raise ValueError(f"num_block_to_prune must be in [1, {self.original_num_layers-1}]")

        sorted_idx = sorted(range(self.original_num_layers), key=lambda i: block_importance[i])
        to_prune = sorted_idx[:num_block_to_prune]

        pruned_model = copy.deepcopy(self.original_model)
        layers = pruned_model.model.layers  

        for idx in sorted(to_prune, reverse=True):
            del layers[idx]

        pruned_model.config.num_hidden_layers = len(layers)
        pruned_model.config.block_layers_to_prune = to_prune

        return pruned_model


class Qwen2BlockPruner:
    """
    Performs depth pruning by removing entire decoder layers and updating model config.
    """
    def __init__(self, original_model: Qwen2ForCausalLM, device: str = 'cpu'):
        self.original_model = original_model
        self.device = device
        self.original_num_layers = original_model.config.num_hidden_layers

    def prune(self, block_importance: list, num_block_to_prune: int):
        """
        Remove the `num_block_to_prune` lowest-importance layers.

        Args:
            block_importance (list[float]): importance score per layer
            num_block_to_prune (int): how many layers to remove
        Returns:
            Qwen2ForCausalLM: a new model instance with layers removed and config updated
        """
        if not (1 <= num_block_to_prune < self.original_num_layers):
            raise ValueError(f"num_block_to_prune must be in [1, {self.original_num_layers-1}]")

        sorted_idx = sorted(range(self.original_num_layers), key=lambda i: block_importance[i])
        to_prune = sorted_idx[:num_block_to_prune]

        pruned_model = copy.deepcopy(self.original_model)
        layers = pruned_model.model.layers  

        for idx in sorted(to_prune, reverse=True):
            del layers[idx]

        pruned_model.config.num_hidden_layers = len(layers)
        pruned_model.config.block_layers_to_prune = to_prune

        return pruned_model