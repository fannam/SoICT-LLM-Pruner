import copy
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from collections import defaultdict
from tqdm.auto import tqdm

class Llama3ActivationElementEstimator:
    """
    Estimate activation-based importance for:
      - Attention heads: F_head^(i) = aggregated per-head norm over activations
      - MLP neurons:     F_neuron^(i) = aggregated neuron activations at gate_proj
    Aggregation methods supported: "sum", "mean", "l2", "var".
    """
    def __init__(self, model: nn.Module, dataloader, device: str = "cuda"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        cfg = copy.deepcopy(model.config)
        # derive attention parameters
        self.num_heads = cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        # derive MLP parameters from first layer
        first_mlp = model.model.layers[0].mlp
        self.intermediate_size = first_mlp.gate_proj.out_features

    def estimate_attention_heads(self, agg: str = "l2"):
        """
        Compute activation-based importance for attention heads.
        Hooks the input to each layer.self_attn.o_proj, computes per-head L2 norms,
        and aggregates over batch and sequence according to `agg`.
        Returns a dict layer_idx -> importance tensor (num_heads,).
        """
        self.model.eval()
        imp = {i: torch.zeros(self.num_heads, device=self.device)
               for i in range(len(self.model.model.layers))}
        count = defaultdict(int)
        hooks = []

        def aggregate(x: torch.Tensor):
            # x: (B, S, num_heads)
            flat = x.reshape(-1, x.size(-1))  # (B*S, num_heads)
            if agg == "sum":
                return flat.sum(dim=0)
            elif agg == "mean":
                return flat.mean(dim=0)
            elif agg == "l2":
                return torch.sqrt((flat**2).sum(dim=0))
            elif agg == "var":
                return flat.var(dim=0)
            else:
                raise ValueError(f"Unknown agg: {agg}")

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                ctx = inputs[0].detach()  # (B, S, H)
                B, S, H = ctx.shape
                # reshape to per-head
                h = ctx.view(B, S, self.num_heads, self.head_dim)
                # compute per-head L2 norm
                norms = torch.norm(h, dim=-1)  # (B, S, num_heads)
                # aggregate over batch & seq
                imp[layer_idx] += aggregate(norms)
                count[layer_idx] += B * S
            return hook

        # register hooks for all layers
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.o_proj.register_forward_hook(make_hook(idx))
            hooks.append(hook)

        # forward pass
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Estimating attention heads importance"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)

        # remove hooks
        for h in hooks:
            h.remove()

        # normalize according to count (for mean/l2/var semantics)
        importance = {}
        for idx, vec in imp.items():
            denom = float(count[idx]) if count[idx] > 0 else 1.0
            importance[idx] = (vec / denom).cpu()

        return importance

    def estimate_mlp_neurons(self, agg: str = "l2"):
        """
        Compute activation-based importance for MLP intermediate neurons.
        Hooks the input to each layer.mlp.gate_proj and aggregates
        over batch and sequence to produce a Dint importance vector per layer.
        """
        self.model.eval()
        imp = {}
        count = defaultdict(int)
        hooks = []

        def aggregate(x: torch.Tensor):
            flat = x.reshape(-1, x.size(-1))  # (B*S, Dint)
            if agg == "sum":
                return flat.sum(dim=0)
            elif agg == "mean":
                return flat.mean(dim=0)
            elif agg == "l2":
                return torch.sqrt((flat**2).sum(dim=0))
            elif agg == "var":
                return flat.var(dim=0)
            else:
                raise ValueError(f"Unknown agg: {agg}")

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                h = inputs[0].detach()  # (B, S, Dint)
                imp[layer_idx] += aggregate(h)
                B, S, _ = h.shape
                count[layer_idx] += B * S
            return hook

        for idx, layer in enumerate(self.model.model.layers):
            Dint = layer.mlp.down_proj.in_features
            imp[idx] = torch.zeros(Dint, device=self.device)
            hook = layer.mlp.down_proj.register_forward_hook(make_hook(idx))
            hooks.append(hook)

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='Estimating MLP neuron importance'):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**inputs)

        for h in hooks:
            h.remove()

        importance = {}
        for idx, vec in imp.items():
            denom = float(count[idx]) if count[idx] > 0 else 1.0
            importance[idx] = (vec / denom).cpu()

        return importance

    def estimate_embedding_channels(self, agg: str = "l2"):
        """
        Compute activation-based importance for embedding channels using:
          - input_layernorm
          - post_attention_layernorm
          - final norm
        Returns a dict:
          key (str) -> importance tensor (hidden_size,)
        Keys: "input_layernorm_{i}", "post_attention_layernorm_{i}" for each block,
              and "final_norm" for the top-level norm.
        """
        self.model.eval()
        imp = {}
        count = defaultdict(int)
        hooks = []

        # initialize buffers
        for idx, layer in enumerate(self.model.model.layers):
            imp[f"input_layernorm_{idx}"] = torch.zeros(self.hidden_size, device=self.device)
            imp[f"post_attention_layernorm_{idx}"] = torch.zeros(self.hidden_size, device=self.device)
        imp["final_norm"] = torch.zeros(self.hidden_size, device=self.device)

        def aggregate(x: torch.Tensor):
            flat = x.reshape(-1, x.size(-1))
            if agg == "sum":
                return flat.sum(dim=0)
            elif agg == "mean":
                return flat.mean(dim=0)
            elif agg == "l2":
                return torch.sqrt((flat**2).sum(dim=0))
            elif agg == "var":
                return flat.var(dim=0)
            else:
                raise ValueError(f"Unknown agg: {agg}")

        def make_hook(key):
            def hook(module, inputs, output):
                h = output.detach()  # (B, S, H)
                B, S, _ = h.shape
                imp[key] += aggregate(h)
                count[key] += B * S
            return hook

        # register hooks on block norms
        for idx, layer in enumerate(self.model.model.layers):
            hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"input_layernorm_{idx}")))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook(f"post_attention_layernorm_{idx}")))
        # final norm
        hooks.append(self.model.model.norm.register_forward_hook(make_hook("final_norm")))

        # run forward
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Estimating embedding channels importance"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)

        # remove hooks
        for h in hooks:
            h.remove()

        # normalize
        importance = {}
        for key, vec in imp.items():
            denom = float(count[key]) if count[key] > 0 else 1.0
            importance[key] = (vec / denom).cpu()
        return importance

class Qwen2ActivationElementEstimator:
    """
    Estimate activation-based importance for:
      - Attention heads: F_head^(i) = aggregated per-head norm over activations
      - MLP neurons:     F_neuron^(i) = aggregated neuron activations at gate_proj
    Aggregation methods supported: "sum", "mean", "l2", "var".
    """
    def __init__(self, model: nn.Module, dataloader, device: str = "cuda"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        cfg = copy.deepcopy(model.config)
        # derive attention parameters
        self.num_heads = cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        # derive MLP parameters from first layer
        first_mlp = model.model.layers[0].mlp
        self.intermediate_size = first_mlp.gate_proj.out_features

    def estimate_attention_heads(self, agg: str = "l2"):
        """
        Compute activation-based importance for attention heads.
        Hooks the input to each layer.self_attn.o_proj, computes per-head L2 norms,
        and aggregates over batch and sequence according to `agg`.
        Returns a dict layer_idx -> importance tensor (num_heads,).
        """
        self.model.eval()
        imp = {i: torch.zeros(self.num_heads, device=self.device)
               for i in range(len(self.model.model.layers))}
        count = defaultdict(int)
        hooks = []

        def aggregate(x: torch.Tensor):
            # x: (B, S, num_heads)
            flat = x.reshape(-1, x.size(-1))  # (B*S, num_heads)
            if agg == "sum":
                return flat.sum(dim=0)
            elif agg == "mean":
                return flat.mean(dim=0)
            elif agg == "l2":
                return torch.sqrt((flat**2).sum(dim=0))
            elif agg == "var":
                return flat.var(dim=0)
            else:
                raise ValueError(f"Unknown agg: {agg}")

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                ctx = inputs[0].detach()  # (B, S, H)
                B, S, H = ctx.shape
                # reshape to per-head
                h = ctx.view(B, S, self.num_heads, self.head_dim)
                # compute per-head L2 norm
                norms = torch.norm(h, dim=-1)  # (B, S, num_heads)
                # aggregate over batch & seq
                imp[layer_idx] += aggregate(norms)
                count[layer_idx] += B * S
            return hook

        # register hooks for all layers
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.o_proj.register_forward_hook(make_hook(idx))
            hooks.append(hook)

        # forward pass
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Estimating attention heads importance"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)

        # remove hooks
        for h in hooks:
            h.remove()

        # normalize according to count (for mean/l2/var semantics)
        importance = {}
        for idx, vec in imp.items():
            denom = float(count[idx]) if count[idx] > 0 else 1.0
            importance[idx] = (vec / denom).cpu()

        return importance

    def estimate_mlp_neurons(self, agg: str = "l2"):
        """
        Compute activation-based importance for MLP intermediate neurons.
        Hooks the input to each layer.mlp.gate_proj and aggregates
        over batch and sequence to produce a Dint importance vector per layer.
        """
        self.model.eval()
        imp = {}
        count = defaultdict(int)
        hooks = []

        def aggregate(x: torch.Tensor):
            flat = x.reshape(-1, x.size(-1))  # (B*S, Dint)
            if agg == "sum":
                return flat.sum(dim=0)
            elif agg == "mean":
                return flat.mean(dim=0)
            elif agg == "l2":
                return torch.sqrt((flat**2).sum(dim=0))
            elif agg == "var":
                return flat.var(dim=0)
            else:
                raise ValueError(f"Unknown agg: {agg}")

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                h = inputs[0].detach()  # (B, S, Dint)
                imp[layer_idx] += aggregate(h)
                B, S, _ = h.shape
                count[layer_idx] += B * S
            return hook

        for idx, layer in enumerate(self.model.model.layers):
            Dint = layer.mlp.down_proj.in_features
            imp[idx] = torch.zeros(Dint, device=self.device)
            hook = layer.mlp.down_proj.register_forward_hook(make_hook(idx))
            hooks.append(hook)

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='Estimating MLP neuron importance'):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**inputs)

        for h in hooks:
            h.remove()

        importance = {}
        for idx, vec in imp.items():
            denom = float(count[idx]) if count[idx] > 0 else 1.0
            importance[idx] = (vec / denom).cpu()

        return importance

    def estimate_embedding_channels(self, agg: str = "l2"):
        """
        Compute activation-based importance for embedding channels using:
          - input_layernorm
          - post_attention_layernorm
          - final norm
        Returns a dict:
          key (str) -> importance tensor (hidden_size,)
        Keys: "input_layernorm_{i}", "post_attention_layernorm_{i}" for each block,
              and "final_norm" for the top-level norm.
        """
        self.model.eval()
        imp = {}
        count = defaultdict(int)
        hooks = []

        # initialize buffers
        for idx, layer in enumerate(self.model.model.layers):
            imp[f"input_layernorm_{idx}"] = torch.zeros(self.hidden_size, device=self.device)
            imp[f"post_attention_layernorm_{idx}"] = torch.zeros(self.hidden_size, device=self.device)
        imp["final_norm"] = torch.zeros(self.hidden_size, device=self.device)

        def aggregate(x: torch.Tensor):
            flat = x.reshape(-1, x.size(-1))
            if agg == "sum":
                return flat.sum(dim=0)
            elif agg == "mean":
                return flat.mean(dim=0)
            elif agg == "l2":
                return torch.sqrt((flat**2).sum(dim=0))
            elif agg == "var":
                return flat.var(dim=0)
            else:
                raise ValueError(f"Unknown agg: {agg}")

        def make_hook(key):
            def hook(module, inputs, output):
                h = output.detach()  # (B, S, H)
                B, S, _ = h.shape
                imp[key] += aggregate(h)
                count[key] += B * S
            return hook

        # register hooks on block norms
        for idx, layer in enumerate(self.model.model.layers):
            hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"input_layernorm_{idx}")))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook(f"post_attention_layernorm_{idx}")))
        # final norm
        hooks.append(self.model.model.norm.register_forward_hook(make_hook("final_norm")))

        # run forward
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Estimating embedding channels importance"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)

        # remove hooks
        for h in hooks:
            h.remove()

        # normalize
        importance = {}
        for key, vec in imp.items():
            denom = float(count[key]) if count[key] > 0 else 1.0
            importance[key] = (vec / denom).cpu()
        return importance