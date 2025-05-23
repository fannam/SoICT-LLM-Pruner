import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import defaultdict
import gc
import copy
from transformers import LlamaForCausalLM, PreTrainedModel, Qwen2ForCausalLM
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
from utils import IdentityLayer

class Llama3SimilarityBlockEstimator:
    """
    Estimates importance scores for contiguous decoder blocks.

    Each block spans `block_size` consecutive layers. For start index i:
      - x_i: input to attention LayerNorm of layer i
      - h_end+1: output of MLP residual at layer (i + block_size - 1)
    Importance = 1 - cos(flat(x_i), flat(h_end+1)).
    """
    def __init__(self, model: PreTrainedModel, block_size: int, device: str = "cuda"):
        assert isinstance(block_size, int) and block_size >= 1, "block_size must be a positive integer"
        assert isinstance(model, LlamaForCausalLM)
        self.model = model.eval().to(device)
        self.device = device
        self.block_size = block_size

    def _register_hooks(self):
        self._attn_inputs = {}
        self._mlp_inputs = {}
        self._mlp_outputs = {}
        self._hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            self._hooks.append(
                layer.input_layernorm.register_forward_hook(
                    lambda m, inp, out, key=idx: self._attn_inputs.update({key: inp[0].detach().clone()})
                )
            )
            self._hooks.append(
                layer.post_attention_layernorm.register_forward_hook(
                    lambda m, inp, out, key=idx: self._mlp_inputs.update({key: inp[0].detach().clone()})
                )
            )
            self._hooks.append(
                layer.mlp.register_forward_hook(
                    lambda m, inp, out, key=idx: self._mlp_outputs.update({key: (out[0] if isinstance(out, tuple) else out).detach().clone()})
                )
            )

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()

    @torch.no_grad()
    def estimate(self, dataloader):
        num_layers = self.model.config.num_hidden_layers
        max_start = num_layers - self.block_size + 1
        results = []

        self._register_hooks()
        for batch in tqdm(dataloader, desc=f"Estimating block size={self.block_size}"):
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            self._attn_inputs.clear(); self._mlp_inputs.clear(); self._mlp_outputs.clear()
            _ = self.model(**batch)
            for start in range(max_start):
                end = start + self.block_size - 1
                x = self._attn_inputs.get(start)
                inp = self._mlp_inputs.get(end)
                out = self._mlp_outputs.get(end)
                if x is None or inp is None or out is None:
                    continue
                h_next = inp + out
                flat_x = x.view(-1, x.size(-1)).float()
                flat_h = h_next.view(-1, h_next.size(-1)).float()
                cos = F.cosine_similarity(flat_x, flat_h, dim=1).nan_to_num(1.0).mean().item()
                importance = 1.0 - cos
                if len(results) < max_start:
                    results = [[] for _ in range(max_start)]
                results[start].append(importance)
        self._remove_hooks()
        return [sum(lst)/len(lst) if lst else 0.0 for lst in results]
    
class Qwen2SimilarityBlockEstimator:
    """
    Estimates importance scores for contiguous decoder blocks.

    Each block spans `block_size` consecutive layers. For start index i:
      - x_i: input to attention LayerNorm of layer i
      - h_end+1: output of MLP residual at layer (i + block_size - 1)
    Importance = 1 - cos(flat(x_i), flat(h_end+1)).
    """
    def __init__(self, model: PreTrainedModel, block_size: int, device: str = "cuda"):
        assert isinstance(block_size, int) and block_size >= 1, "block_size must be a positive integer"
        assert isinstance(model, Qwen2ForCausalLM)
        self.model = model.eval().to(device)
        self.device = device
        self.block_size = block_size

    def _register_hooks(self):
        self._attn_inputs = {}
        self._mlp_inputs = {}
        self._mlp_outputs = {}
        self._hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            self._hooks.append(
                layer.input_layernorm.register_forward_hook(
                    lambda m, inp, out, key=idx: self._attn_inputs.update({key: inp[0].detach().clone()})
                )
            )
            self._hooks.append(
                layer.post_attention_layernorm.register_forward_hook(
                    lambda m, inp, out, key=idx: self._mlp_inputs.update({key: inp[0].detach().clone()})
                )
            )
            self._hooks.append(
                layer.mlp.register_forward_hook(
                    lambda m, inp, out, key=idx: self._mlp_outputs.update({key: (out[0] if isinstance(out, tuple) else out).detach().clone()})
                )
            )

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()

    @torch.no_grad()
    def estimate(self, dataloader):
        num_layers = self.model.config.num_hidden_layers
        max_start = num_layers - self.block_size + 1
        results = []

        self._register_hooks()
        for batch in tqdm(dataloader, desc=f"Estimating block size={self.block_size}"):
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            self._attn_inputs.clear(); self._mlp_inputs.clear(); self._mlp_outputs.clear()
            _ = self.model(**batch)
            for start in range(max_start):
                end = start + self.block_size - 1
                x = self._attn_inputs.get(start)
                inp = self._mlp_inputs.get(end)
                out = self._mlp_outputs.get(end)
                if x is None or inp is None or out is None:
                    continue
                h_next = inp + out
                flat_x = x.view(-1, x.size(-1)).float()
                flat_h = h_next.view(-1, h_next.size(-1)).float()
                cos = F.cosine_similarity(flat_x, flat_h, dim=1).nan_to_num(1.0).mean().item()
                importance = 1.0 - cos
                if len(results) < max_start:
                    results = [[] for _ in range(max_start)]
                results[start].append(importance)
        self._remove_hooks()
        return [sum(lst)/len(lst) if lst else 0.0 for lst in results]



class Qwen2BlockPerplexityEstimator:
    """
    Estimates importance scores for contiguous decoder blocks based on perplexity changes.
    Each block spans `block_size` consecutive layers.
    Importance is the change in perplexity when the block is replaced by IdentityLayers.
    """
    def __init__(self, model: nn.Module, tokenizer, block_size: int, device: str = "cpu"):
        assert isinstance(block_size, int) and block_size >= 1, "block_size must be a positive integer"
        assert isinstance(model, Qwen2ForCausalLM)
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device
        self.model.to(self.device)

        if not hasattr(model, 'config'):
            raise ValueError("Model does not have a 'config' attribute.")
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers

        if block_size > self.num_layers:
            raise ValueError(f"block_size ({block_size}) cannot be greater than num_layers ({self.num_layers})")
        
        if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
            raise ValueError("Model structure (model.model.layers) not as expected.")


    def _calculate_perplexity(self, current_model: nn.Module, dataloader: DataLoader, n_samples: int) -> float:
        """Helper to calculate perplexity for the current model state."""
        current_model.eval()
        total_loss = 0.0
        samples_done = 0
        batches_to_process = math.ceil(n_samples / dataloader.batch_size) if dataloader.batch_size else n_samples
        
        with torch.no_grad():
            pbar = tqdm(dataloader, total=min(batches_to_process, len(dataloader)), desc="Calculating PPL for block", leave=False)
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= batches_to_process:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = current_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False
                )
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                samples_done += input_ids.size(0)
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / samples_done if samples_done > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss > 0 and not math.isinf(avg_loss) and not math.isnan(avg_loss) else float('inf')
        return perplexity

    def estimate(self, dataloader: DataLoader, n_samples: int = 1024, 
                 importance_metric: str = "perplexity_increase") -> list[float]:
        """
        Estimates block importance based on perplexity.

        Args:
            dataloader: DataLoader providing tokenized input batches.
            n_samples (int): Number of samples from dataloader for perplexity calculation.
            importance_metric (str): "perplexity_increase" or "perplexity_ratio".

        Returns:
            list[float]: A list of importance scores, one for each possible block start.
                         The length is num_layers - block_size + 1.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for perplexity-based importance estimation.")

        self.model.eval()
        self.model.to(self.device)

        print(f"Calculating baseline perplexity (block removal) using {n_samples} samples...")
        baseline_perplexity = self._calculate_perplexity(self.model, dataloader, n_samples)
        print(f"Baseline Perplexity for block estimation: {baseline_perplexity:.4f}")

        max_start_idx = self.num_layers - self.block_size + 1
        block_importances = []
        
        identity_layer_instance = IdentityLayer().to(self.device)

        for start_idx in tqdm(range(max_start_idx), desc=f"Estimating Block Importance (Size {self.block_size}, PPL)"):
            layer_indices_in_block = list(range(start_idx, start_idx + self.block_size))
            original_layers_in_block = {}

            # Replace block with identity layers
            for i in layer_indices_in_block:
                original_layers_in_block[i] = self.model.model.layers[i]
                self.model.model.layers[i] = identity_layer_instance
            
            current_perplexity = self._calculate_perplexity(self.model, dataloader, n_samples)
            
            # Restore original block
            for i in layer_indices_in_block:
                self.model.model.layers[i] = original_layers_in_block[i]

            if importance_metric == "perplexity_increase":
                score = current_perplexity - baseline_perplexity
            elif importance_metric == "perplexity_ratio":
                score = current_perplexity / baseline_perplexity if baseline_perplexity > 0 and not math.isinf(baseline_perplexity) else float('inf')
            else:
                raise ValueError(f"Unknown importance_metric: {importance_metric}")
            
            block_importances.append(score)
            tqdm.write(f"Block starting at {start_idx} (Size {self.block_size}): PPL={current_perplexity:.2f}, Score={score:.2f}")
            
        return block_importances
    

class Llama3BlockPerplexityEstimator:
    """
    Estimates importance scores for contiguous decoder blocks based on perplexity changes.
    Each block spans `block_size` consecutive layers.
    Importance is the change in perplexity when the block is replaced by IdentityLayers.
    """
    def __init__(self, model: nn.Module, tokenizer, block_size: int, device: str = "cpu"):
        assert isinstance(block_size, int) and block_size >= 1, "block_size must be a positive integer"
        assert isinstance(model, LlamaForCausalLM)
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device
        self.model.to(self.device)

        if not hasattr(model, 'config'):
            raise ValueError("Model does not have a 'config' attribute.")
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers

        if block_size > self.num_layers:
            raise ValueError(f"block_size ({block_size}) cannot be greater than num_layers ({self.num_layers})")
        
        if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
            raise ValueError("Model structure (model.model.layers) not as expected.")


    def _calculate_perplexity(self, current_model: nn.Module, dataloader: DataLoader, n_samples: int) -> float:
        """Helper to calculate perplexity for the current model state."""
        current_model.eval()
        total_loss = 0.0
        samples_done = 0
        batches_to_process = math.ceil(n_samples / dataloader.batch_size) if dataloader.batch_size else n_samples
        
        with torch.no_grad():
            pbar = tqdm(dataloader, total=min(batches_to_process, len(dataloader)), desc="Calculating PPL for block", leave=False)
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= batches_to_process:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = current_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False
                )
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                samples_done += input_ids.size(0)
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / samples_done if samples_done > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss > 0 and not math.isinf(avg_loss) and not math.isnan(avg_loss) else float('inf')
        return perplexity

    def estimate(self, dataloader: DataLoader, n_samples: int = 1024, 
                 importance_metric: str = "perplexity_increase") -> list[float]:
        """
        Estimates block importance based on perplexity.

        Args:
            dataloader: DataLoader providing tokenized input batches.
            n_samples (int): Number of samples from dataloader for perplexity calculation.
            importance_metric (str): "perplexity_increase" or "perplexity_ratio".

        Returns:
            list[float]: A list of importance scores, one for each possible block start.
                         The length is num_layers - block_size + 1.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for perplexity-based importance estimation.")

        self.model.eval()
        self.model.to(self.device)

        print(f"Calculating baseline perplexity (block removal) using {n_samples} samples...")
        baseline_perplexity = self._calculate_perplexity(self.model, dataloader, n_samples)
        print(f"Baseline Perplexity for block estimation: {baseline_perplexity:.4f}")

        max_start_idx = self.num_layers - self.block_size + 1
        block_importances = []
        
        identity_layer_instance = IdentityLayer().to(self.device)

        for start_idx in tqdm(range(max_start_idx), desc=f"Estimating Block Importance (Size {self.block_size}, PPL)"):
            layer_indices_in_block = list(range(start_idx, start_idx + self.block_size))
            original_layers_in_block = {}

            # Replace block with identity layers
            for i in layer_indices_in_block:
                original_layers_in_block[i] = self.model.model.layers[i]
                self.model.model.layers[i] = identity_layer_instance
            
            current_perplexity = self._calculate_perplexity(self.model, dataloader, n_samples)
            
            # Restore original block
            for i in layer_indices_in_block:
                self.model.model.layers[i] = original_layers_in_block[i]

            if importance_metric == "perplexity_increase":
                score = current_perplexity - baseline_perplexity
            elif importance_metric == "perplexity_ratio":
                score = current_perplexity / baseline_perplexity if baseline_perplexity > 0 and not math.isinf(baseline_perplexity) else float('inf')
            else:
                raise ValueError(f"Unknown importance_metric: {importance_metric}")
            
            block_importances.append(score)
            tqdm.write(f"Block starting at {start_idx} (Size {self.block_size}): PPL={current_perplexity:.2f}, Score={score:.2f}")
            
        return block_importances