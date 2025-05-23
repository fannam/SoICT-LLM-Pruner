import copy
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2ForCausalLM, LlamaForCausalLM
from collections import defaultdict
from tqdm.auto import tqdm

class Llama3ActivationElementEstimator:
    """
    Estimate activation-based importance for:
      - Attention heads: F_head^(i) = aggregated per-head norm over activations
      - MLP neurons:     F_neuron^(i) = aggregated neuron activations at gate_proj
    Aggregation methods supported: "sum", "mean", "l2", "var".
    """
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        assert isinstance(model, LlamaForCausalLM), "Model must be a LlamaForCausalLM instance"
        cfg = copy.deepcopy(model.config)
        self.num_heads = cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        first_mlp = model.model.layers[0].mlp
        self.intermediate_size = first_mlp.gate_proj.out_features

    def estimate_attention_heads(self, dataloader, agg: str = "l2"):
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

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                ctx = inputs[0].detach()  
                B, S, H = ctx.shape
                h = ctx.view(B, S, self.num_heads, self.head_dim)
                norms = torch.norm(h, dim=-1) 
                imp[layer_idx] += aggregate(norms)
                count[layer_idx] += B * S
            return hook

        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.o_proj.register_forward_hook(make_hook(idx))
            hooks.append(hook)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Estimating attention heads importance"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)

        for h in hooks:
            h.remove()

        importance = {}
        for idx, vec in imp.items():
            denom = float(count[idx]) if count[idx] > 0 else 1.0
            importance[idx] = (vec / denom).cpu()

        return importance

    def estimate_mlp_neurons(self, dataloader, agg: str = "l2"):
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

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                h = inputs[0].detach()  
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
            for batch in tqdm(dataloader, desc='Estimating MLP neuron importance'):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**inputs)

        for h in hooks:
            h.remove()

        importance = {}
        for idx, vec in imp.items():
            denom = float(count[idx]) if count[idx] > 0 else 1.0
            importance[idx] = (vec / denom).cpu()

        return importance

    def estimate_embedding_channels(self, dataloader,agg: str = "l2"):
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
                h = output.detach()  
                B, S, _ = h.shape
                imp[key] += aggregate(h)
                count[key] += B * S
            return hook

        for idx, layer in enumerate(self.model.model.layers):
            hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"input_layernorm_{idx}")))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook(f"post_attention_layernorm_{idx}")))
        hooks.append(self.model.model.norm.register_forward_hook(make_hook("final_norm")))

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Estimating embedding channels importance"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)

        for h in hooks:
            h.remove()

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
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        assert isinstance(model, Qwen2ForCausalLM), "Model must be a Qwen2ForCausalLM instance"
        cfg = copy.deepcopy(model.config)
        self.num_heads = cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        first_mlp = model.model.layers[0].mlp
        self.intermediate_size = first_mlp.gate_proj.out_features

    def estimate_attention_heads(self, dataloader, agg: str = "l2"):
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

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                ctx = inputs[0].detach()  
                B, S, H = ctx.shape
                h = ctx.view(B, S, self.num_heads, self.head_dim)
                norms = torch.norm(h, dim=-1)  
                imp[layer_idx] += aggregate(norms)
                count[layer_idx] += B * S
            return hook

        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.o_proj.register_forward_hook(make_hook(idx))
            hooks.append(hook)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Estimating attention heads importance"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)

        for h in hooks:
            h.remove()

        importance = {}
        for idx, vec in imp.items():
            denom = float(count[idx]) if count[idx] > 0 else 1.0
            importance[idx] = (vec / denom).cpu()

        return importance

    def estimate_mlp_neurons(self, dataloader, agg: str = "l2"):
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

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                h = inputs[0].detach()  
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
            for batch in tqdm(dataloader, desc='Estimating MLP neuron importance'):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**inputs)

        for h in hooks:
            h.remove()

        importance = {}
        for idx, vec in imp.items():
            denom = float(count[idx]) if count[idx] > 0 else 1.0
            importance[idx] = (vec / denom).cpu()

        return importance

    def estimate_embedding_channels(self, dataloader, agg: str = "l2"):
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
                h = output.detach()  
                B, S, _ = h.shape
                imp[key] += aggregate(h)
                count[key] += B * S
            return hook

        for idx, layer in enumerate(self.model.model.layers):
            hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"input_layernorm_{idx}")))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook(f"post_attention_layernorm_{idx}")))
        hooks.append(self.model.model.norm.register_forward_hook(make_hook("final_norm")))

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Estimating embedding channels importance"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)

        for h in hooks:
            h.remove()

        importance = {}
        for key, vec in imp.items():
            denom = float(count[key]) if count[key] > 0 else 1.0
            importance[key] = (vec / denom).cpu()
        return importance
    

class Qwen2WeightMagnitudeEstimator:
    """
    Estimate weight magnitude-based importance for a Qwen2 model:
      - Embedding channels: Importance is the norm of weights for the i-th channel
                          in the model.model.embed_tokens.weight matrix.
      - MLP neurons:      Importance is the norm of incoming weights to the i-th neuron
                          in each layer's mlp.gate_proj.weight matrix.
      - Attention heads:  Importance is the sum of norms of weights for the i-th head
                          across q_proj, k_proj, v_proj, and o_proj.
    Aggregation for norm can be "l1" or "l2".
    """
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initializes the estimator.

        Args:
            model (nn.Module): The Qwen2ForCausalLM model instance (or a similar HF model
                               with a .config attribute and standard layer structure).
            device (str): The device to perform calculations on ('cpu' or 'cuda').
                          Weights will be temporarily moved to this device if not already there.
                          Final importance tensors are returned on CPU.
        """
        self.model = model
        self.device = device
        assert isinstance(model, Qwen2ForCausalLM), "Model must be a Qwen2ForCausalLM instance"

        if not hasattr(model, 'config'):
            raise ValueError("Model does not have a 'config' attribute. "
                             "Please pass a Hugging Face PreTrainedModel instance.")
        
        config = model.config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        # Fallback for models that might not explicitly have num_key_value_heads (older MHA models)
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_attention_heads)

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        self.head_dim = self.hidden_size // self.num_attention_heads

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) for GQA."
            )

        # Basic structural checks
        if not hasattr(model, 'model'):
            raise ValueError("The provided model does not have a 'model' attribute (e.g., Qwen2ForCausalLM.model).")
        if not hasattr(model.model, 'embed_tokens') or not hasattr(model.model.embed_tokens, 'weight'):
            raise ValueError("Missing model.model.embed_tokens.weight.")
        if self.num_layers > 0:
            if not hasattr(model.model, 'layers') or len(model.model.layers) == 0:
                raise ValueError("model.model.layers is missing or empty.")
            first_layer = model.model.layers[0]
            if not hasattr(first_layer, 'mlp') or not hasattr(first_layer.mlp, 'gate_proj') or \
               not hasattr(first_layer.mlp.gate_proj, 'weight'):
                raise ValueError("Missing model.model.layers[0].mlp.gate_proj.weight.")
            if not hasattr(first_layer, 'self_attn'):
                raise ValueError("Missing model.model.layers[0].self_attn.")
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if not hasattr(first_layer.self_attn, proj_name) or \
                   not hasattr(getattr(first_layer.self_attn, proj_name), 'weight'):
                    raise ValueError(f"Missing model.model.layers[0].self_attn.{proj_name}.weight.")
            
            # Verify K,V projection output features if possible
            # k_proj_out_features = self.num_key_value_heads * self.head_dim
            # actual_k_out = first_layer.self_attn.k_proj.weight.shape[0] # out_features for Linear
            # if actual_k_out != k_proj_out_features:
            #     print(f"Warning: k_proj.weight.shape[0] ({actual_k_out}) does not match "
            #           f"num_key_value_heads * head_dim ({k_proj_out_features}). "
            #           f"This might indicate head_dim for K/V differs from Q, or a specific GQA setup. "
            #           f"The estimator assumes head_dim ({self.head_dim}) is consistent for Q, K, V projections.")


    def _calculate_norm(self, tensor: torch.Tensor, agg: str, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Helper function to calculate L1 or L2 norm."""
        if agg.lower() == "l1":
            return torch.norm(tensor, p=1, dim=dim)
        elif agg.lower() == "l2":
            return torch.norm(tensor, p=2, dim=dim)
        else:
            raise ValueError(f"Unknown aggregation (norm type): {agg}. Choose 'l1' or 'l2'.")

    def estimate_embedding_channels(self, agg: str = "l2") -> dict[str, torch.Tensor]:
        """
        Computes weight magnitude-based importance for embedding channels.
        Importance is the specified norm of weights for each channel across all tokens
        in the model.model.embed_tokens.weight tensor.

        Args:
            agg (str): The type of norm to use for aggregation ("l1" or "l2"). Default is "l2".

        Returns:
            dict: A dictionary `{"embedding_channels": importance_tensor}` where
                  `importance_tensor` is a 1D tensor of shape (hidden_size,)
                  containing the importance score for each embedding channel, moved to CPU.
        """
        self.model.eval()
        with torch.no_grad():
            embedding_weights = self.model.model.embed_tokens.weight.detach().to(self.device)
            importance = self._calculate_norm(embedding_weights, agg, dim=0)
        return {"embedding_channels": importance.cpu()}

    def estimate_mlp_neurons(self, agg: str = "l2") -> dict[int, torch.Tensor]:
        """
        Computes weight magnitude-based importance for MLP intermediate neurons.
        Importance for each neuron in a layer's MLP (specifically, the intermediate
        neurons corresponding to gate_proj output) is based on the norm of its
        incoming weights in the gate_proj layer.

        Args:
            agg (str): The type of norm to use for aggregation ("l1" or "l2"). Default is "l2".

        Returns:
            dict: A dictionary mapping `layer_idx` (int) to an `importance_tensor` (1D tensor
                  of shape (intermediate_size,)) for that layer. Importance tensors are moved to CPU.
        """
        self.model.eval()
        importance_per_layer = {}
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.model.model.layers):
                gate_proj_weights = layer.mlp.gate_proj.weight.detach().to(self.device)
                # gate_proj_weights shape: (intermediate_size, hidden_size)
                # Norm of each row (dim=1) gives importance per neuron.
                neuron_importance = self._calculate_norm(gate_proj_weights, agg, dim=1)
                importance_per_layer[layer_idx] = neuron_importance.cpu()
        return importance_per_layer

    def estimate_attention_heads(self, agg: str = "l2") -> dict[int, torch.Tensor]:
        """
        Computes weight magnitude-based importance for attention heads.
        Importance for each head is the sum of norms of its associated weights
        in q_proj, k_proj, v_proj, and o_proj.

        Args:
            agg (str): The type of norm to use for aggregation ("l1" or "l2"). Default is "l2".

        Returns:
            dict: A dictionary mapping `layer_idx` (int) to an `importance_tensor` (1D tensor
                  of shape (num_attention_heads,)) for that layer. Importance tensors are moved to CPU.
        """
        self.model.eval()
        importance_per_layer = {}

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.model.model.layers):
                attn = layer.self_attn
                
                # Q-projection weights
                # q_proj.weight shape: (num_attention_heads * head_dim, hidden_size)
                q_w = attn.q_proj.weight.detach().to(self.device)
                q_w_reshaped = q_w.view(self.num_attention_heads, self.head_dim, self.hidden_size)
                norm_q = self._calculate_norm(q_w_reshaped, agg, dim=(1, 2)) # Shape: (num_attention_heads,)

                # K-projection weights
                # k_proj.weight shape: (num_key_value_heads * head_dim, hidden_size)
                k_w = attn.k_proj.weight.detach().to(self.device)
                k_w_reshaped = k_w.view(self.num_key_value_heads, self.head_dim, self.hidden_size)
                norm_k_kv = self._calculate_norm(k_w_reshaped, agg, dim=(1, 2)) # Shape: (num_key_value_heads,)
                
                # V-projection weights
                # v_proj.weight shape: (num_key_value_heads * head_dim, hidden_size)
                v_w = attn.v_proj.weight.detach().to(self.device)
                v_w_reshaped = v_w.view(self.num_key_value_heads, self.head_dim, self.hidden_size)
                norm_v_kv = self._calculate_norm(v_w_reshaped, agg, dim=(1, 2)) # Shape: (num_key_value_heads,)

                if self.num_attention_heads != self.num_key_value_heads: # GQA or MQA
                    repetition_factor = self.num_attention_heads // self.num_key_value_heads
                    norm_k = norm_k_kv.repeat_interleave(repetition_factor)
                    norm_v = norm_v_kv.repeat_interleave(repetition_factor)
                else: # MHA
                    norm_k = norm_k_kv
                    norm_v = norm_v_kv
                
                # O-projection weights
                # o_proj.weight shape: (hidden_size, num_attention_heads * head_dim)
                o_w = attn.o_proj.weight.detach().to(self.device)
                o_w_reshaped = o_w.view(self.hidden_size, self.num_attention_heads, self.head_dim)
                # Permute to (num_attention_heads, hidden_size, head_dim) to isolate head's output transformation
                o_w_permuted = o_w_reshaped.permute(1, 0, 2) 
                norm_o = self._calculate_norm(o_w_permuted, agg, dim=(1, 2)) # Shape: (num_attention_heads,)

                total_head_importance = norm_q + norm_k + norm_v + norm_o
                importance_per_layer[layer_idx] = total_head_importance.cpu()
                
        return importance_per_layer
    
class Llama3WeightMagnitudeEstimator:
    """
    Estimate weight magnitude-based importance for a Llama3 model:
      - Embedding channels: Importance is the norm of weights for the i-th channel
                          in the model.model.embed_tokens.weight matrix.
      - MLP neurons:      Importance is the norm of incoming weights to the i-th neuron
                          in each layer's mlp.gate_proj.weight matrix.
      - Attention heads:  Importance is the sum of norms of weights for the i-th head
                          across q_proj, k_proj, v_proj, and o_proj.
    Aggregation for norm can be "l1" or "l2".
    """
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initializes the estimator.

        Args:
            model (nn.Module): The Qwen2ForCausalLM model instance (or a similar HF model
                               with a .config attribute and standard layer structure).
            device (str): The device to perform calculations on ('cpu' or 'cuda').
                          Weights will be temporarily moved to this device if not already there.
                          Final importance tensors are returned on CPU.
        """
        self.model = model
        self.device = device
        assert isinstance(model, LlamaForCausalLM), "Model must be a LlamaForCausalLM instance"

        if not hasattr(model, 'config'):
            raise ValueError("Model does not have a 'config' attribute. "
                             "Please pass a Hugging Face PreTrainedModel instance.")
        
        config = model.config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        # Fallback for models that might not explicitly have num_key_value_heads (older MHA models)
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_attention_heads)

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        self.head_dim = self.hidden_size // self.num_attention_heads

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) for GQA."
            )

        # Basic structural checks
        if not hasattr(model, 'model'):
            raise ValueError("The provided model does not have a 'model' attribute (e.g., Qwen2ForCausalLM.model).")
        if not hasattr(model.model, 'embed_tokens') or not hasattr(model.model.embed_tokens, 'weight'):
            raise ValueError("Missing model.model.embed_tokens.weight.")
        if self.num_layers > 0:
            if not hasattr(model.model, 'layers') or len(model.model.layers) == 0:
                raise ValueError("model.model.layers is missing or empty.")
            first_layer = model.model.layers[0]
            if not hasattr(first_layer, 'mlp') or not hasattr(first_layer.mlp, 'gate_proj') or \
               not hasattr(first_layer.mlp.gate_proj, 'weight'):
                raise ValueError("Missing model.model.layers[0].mlp.gate_proj.weight.")
            if not hasattr(first_layer, 'self_attn'):
                raise ValueError("Missing model.model.layers[0].self_attn.")
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if not hasattr(first_layer.self_attn, proj_name) or \
                   not hasattr(getattr(first_layer.self_attn, proj_name), 'weight'):
                    raise ValueError(f"Missing model.model.layers[0].self_attn.{proj_name}.weight.")
            
            # Verify K,V projection output features if possible
            # k_proj_out_features = self.num_key_value_heads * self.head_dim
            # actual_k_out = first_layer.self_attn.k_proj.weight.shape[0] # out_features for Linear
            # if actual_k_out != k_proj_out_features:
            #     print(f"Warning: k_proj.weight.shape[0] ({actual_k_out}) does not match "
            #           f"num_key_value_heads * head_dim ({k_proj_out_features}). "
            #           f"This might indicate head_dim for K/V differs from Q, or a specific GQA setup. "
            #           f"The estimator assumes head_dim ({self.head_dim}) is consistent for Q, K, V projections.")


    def _calculate_norm(self, tensor: torch.Tensor, agg: str, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Helper function to calculate L1 or L2 norm."""
        if agg.lower() == "l1":
            return torch.norm(tensor, p=1, dim=dim)
        elif agg.lower() == "l2":
            return torch.norm(tensor, p=2, dim=dim)
        else:
            raise ValueError(f"Unknown aggregation (norm type): {agg}. Choose 'l1' or 'l2'.")

    def estimate_embedding_channels(self, agg: str = "l2") -> dict[str, torch.Tensor]:
        """
        Computes weight magnitude-based importance for embedding channels.
        Importance is the specified norm of weights for each channel across all tokens
        in the model.model.embed_tokens.weight tensor.

        Args:
            agg (str): The type of norm to use for aggregation ("l1" or "l2"). Default is "l2".

        Returns:
            dict: A dictionary `{"embedding_channels": importance_tensor}` where
                  `importance_tensor` is a 1D tensor of shape (hidden_size,)
                  containing the importance score for each embedding channel, moved to CPU.
        """
        self.model.eval()
        with torch.no_grad():
            embedding_weights = self.model.model.embed_tokens.weight.detach().to(self.device)
            importance = self._calculate_norm(embedding_weights, agg, dim=0)
        return {"embedding_channels": importance.cpu()}

    def estimate_mlp_neurons(self, agg: str = "l2") -> dict[int, torch.Tensor]:
        """
        Computes weight magnitude-based importance for MLP intermediate neurons.
        Importance for each neuron in a layer's MLP (specifically, the intermediate
        neurons corresponding to gate_proj output) is based on the norm of its
        incoming weights in the gate_proj layer.

        Args:
            agg (str): The type of norm to use for aggregation ("l1" or "l2"). Default is "l2".

        Returns:
            dict: A dictionary mapping `layer_idx` (int) to an `importance_tensor` (1D tensor
                  of shape (intermediate_size,)) for that layer. Importance tensors are moved to CPU.
        """
        self.model.eval()
        importance_per_layer = {}
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.model.model.layers):
                gate_proj_weights = layer.mlp.gate_proj.weight.detach().to(self.device)
                # gate_proj_weights shape: (intermediate_size, hidden_size)
                # Norm of each row (dim=1) gives importance per neuron.
                neuron_importance = self._calculate_norm(gate_proj_weights, agg, dim=1)
                importance_per_layer[layer_idx] = neuron_importance.cpu()
        return importance_per_layer

    def estimate_attention_heads(self, agg: str = "l2") -> dict[int, torch.Tensor]:
        """
        Computes weight magnitude-based importance for attention heads.
        Importance for each head is the sum of norms of its associated weights
        in q_proj, k_proj, v_proj, and o_proj.

        Args:
            agg (str): The type of norm to use for aggregation ("l1" or "l2"). Default is "l2".

        Returns:
            dict: A dictionary mapping `layer_idx` (int) to an `importance_tensor` (1D tensor
                  of shape (num_attention_heads,)) for that layer. Importance tensors are moved to CPU.
        """
        self.model.eval()
        importance_per_layer = {}

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.model.model.layers):
                attn = layer.self_attn
                
                # Q-projection weights
                # q_proj.weight shape: (num_attention_heads * head_dim, hidden_size)
                q_w = attn.q_proj.weight.detach().to(self.device)
                q_w_reshaped = q_w.view(self.num_attention_heads, self.head_dim, self.hidden_size)
                norm_q = self._calculate_norm(q_w_reshaped, agg, dim=(1, 2)) # Shape: (num_attention_heads,)

                # K-projection weights
                # k_proj.weight shape: (num_key_value_heads * head_dim, hidden_size)
                k_w = attn.k_proj.weight.detach().to(self.device)
                k_w_reshaped = k_w.view(self.num_key_value_heads, self.head_dim, self.hidden_size)
                norm_k_kv = self._calculate_norm(k_w_reshaped, agg, dim=(1, 2)) # Shape: (num_key_value_heads,)
                
                # V-projection weights
                # v_proj.weight shape: (num_key_value_heads * head_dim, hidden_size)
                v_w = attn.v_proj.weight.detach().to(self.device)
                v_w_reshaped = v_w.view(self.num_key_value_heads, self.head_dim, self.hidden_size)
                norm_v_kv = self._calculate_norm(v_w_reshaped, agg, dim=(1, 2)) # Shape: (num_key_value_heads,)

                if self.num_attention_heads != self.num_key_value_heads: # GQA or MQA
                    repetition_factor = self.num_attention_heads // self.num_key_value_heads
                    norm_k = norm_k_kv.repeat_interleave(repetition_factor)
                    norm_v = norm_v_kv.repeat_interleave(repetition_factor)
                else: # MHA
                    norm_k = norm_k_kv
                    norm_v = norm_v_kv
                
                # O-projection weights
                # o_proj.weight shape: (hidden_size, num_attention_heads * head_dim)
                o_w = attn.o_proj.weight.detach().to(self.device)
                o_w_reshaped = o_w.view(self.hidden_size, self.num_attention_heads, self.head_dim)
                # Permute to (num_attention_heads, hidden_size, head_dim) to isolate head's output transformation
                o_w_permuted = o_w_reshaped.permute(1, 0, 2) 
                norm_o = self._calculate_norm(o_w_permuted, agg, dim=(1, 2)) # Shape: (num_attention_heads,)

                total_head_importance = norm_q + norm_k + norm_v + norm_o
                importance_per_layer[layer_idx] = total_head_importance.cpu()
                
        return importance_per_layer