import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import defaultdict
import gc
import copy
from transformers import LlamaForCausalLM, PreTrainedModel, Qwen2ForCausalLM

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
        assert isinstance(model, LlamaForCausalLM) or isinstance(model, Qwen2ForCausalLM)
        self.model = model.eval().to(device)
        self.device = device
        self.block_size = block_size

    def _register_hooks(self):
        self._attn_inputs = {}
        self._mlp_inputs = {}
        self._mlp_outputs = {}
        self._hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            # Capture input to attention LayerNorm of layer idx
            self._hooks.append(
                layer.input_layernorm.register_forward_hook(
                    lambda m, inp, out, key=idx: self._attn_inputs.update({key: inp[0].detach().clone()})
                )
            )
            # Capture MLP input (post-attention norm)
            self._hooks.append(
                layer.post_attention_layernorm.register_forward_hook(
                    lambda m, inp, out, key=idx: self._mlp_inputs.update({key: inp[0].detach().clone()})
                )
            )
            # Capture MLP output
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
            # after capture, compute for each start
            for start in range(max_start):
                end = start + self.block_size - 1
                x = self._attn_inputs.get(start)
                inp = self._mlp_inputs.get(end)
                out = self._mlp_outputs.get(end)
                if x is None or inp is None or out is None:
                    continue
                # compute h_{end+1}
                h_next = inp + out
                flat_x = x.view(-1, x.size(-1)).float()
                flat_h = h_next.view(-1, h_next.size(-1)).float()
                cos = F.cosine_similarity(flat_x, flat_h, dim=1).nan_to_num(1.0).mean().item()
                importance = 1.0 - cos
                # ensure results list length
                if len(results) < max_start:
                    results = [[] for _ in range(max_start)]
                results[start].append(importance)
        self._remove_hooks()
        # average across batches
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
        assert isinstance(model, LlamaForCausalLM) or isinstance(model, Qwen2ForCausalLM)
        self.model = model.eval().to(device)
        self.device = device
        self.block_size = block_size

    def _register_hooks(self):
        self._attn_inputs = {}
        self._mlp_inputs = {}
        self._mlp_outputs = {}
        self._hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            # Capture input to attention LayerNorm of layer idx
            self._hooks.append(
                layer.input_layernorm.register_forward_hook(
                    lambda m, inp, out, key=idx: self._attn_inputs.update({key: inp[0].detach().clone()})
                )
            )
            # Capture MLP input (post-attention norm)
            self._hooks.append(
                layer.post_attention_layernorm.register_forward_hook(
                    lambda m, inp, out, key=idx: self._mlp_inputs.update({key: inp[0].detach().clone()})
                )
            )
            # Capture MLP output
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
            # after capture, compute for each start
            for start in range(max_start):
                end = start + self.block_size - 1
                x = self._attn_inputs.get(start)
                inp = self._mlp_inputs.get(end)
                out = self._mlp_outputs.get(end)
                if x is None or inp is None or out is None:
                    continue
                # compute h_{end+1}
                h_next = inp + out
                flat_x = x.view(-1, x.size(-1)).float()
                flat_h = h_next.view(-1, h_next.size(-1)).float()
                cos = F.cosine_similarity(flat_x, flat_h, dim=1).nan_to_num(1.0).mean().item()
                importance = 1.0 - cos
                # ensure results list length
                if len(results) < max_start:
                    results = [[] for _ in range(max_start)]
                results[start].append(importance)
        self._remove_hooks()
        # average across batches
        return [sum(lst)/len(lst) if lst else 0.0 for lst in results]