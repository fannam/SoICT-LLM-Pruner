import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import defaultdict
from utils import calculate_importance, AttentionPasser, FeedForwardPasser
import gc

class Llama3SimilarityLayerEstimator:
    """
    Estimates importance scores for Attention and MLP sublayers in a causal LM.

    For each layer l:
      - X_A^l: input to the attention block
      - Y_A^l: X_A^l + Attention(LN(X_A^l))
      - X_M^l: input to the MLP block
      - Y_M^l: X_M^l + MLP(LN(X_M^l))
    Importance is defined as 1 - cos(X, Y).
    """
    def __init__(self, model, device: str = "cuda"):  # model: Qwen2ForCausalLM or similar
        self.model = model.eval().to(device)
        self.device = device

    def _register_hooks(self):
        """
        Hook inputs and outputs of attention and MLP.
        Stores tensors in self._inputs and self._outputs dicts keyed by names.
        """
        self._inputs = {}
        self._outputs = {}
        self._hooks = []

        for idx, layer in enumerate(self.model.model.layers):
            # Attention input: before input_layernorm
            name_in = f"attn_in_{idx}"
            hook_in = layer.input_layernorm.register_forward_hook(
                lambda mod, inp, out, key=name_in: self._inputs.update({key: inp[0].detach().clone()})
            )
            self._hooks.append(hook_in)
            # Attention output: after self_attn but before residual
            name_out = f"attn_out_{idx}"
            hook_out = layer.self_attn.register_forward_hook(
                lambda mod, inp, out, key=name_out: self._outputs.update({key: (out[0] if isinstance(out, tuple) else out).detach().clone()})
            )
            self._hooks.append(hook_out)
            # MLP input: before post_attention_layernorm
            name_in = f"mlp_in_{idx}"
            hook_in2 = layer.post_attention_layernorm.register_forward_hook(
                lambda mod, inp, out, key=name_in: self._inputs.update({key: inp[0].detach().clone()})
            )
            self._hooks.append(hook_in2)
            # MLP output: after mlp block
            name_out = f"mlp_out_{idx}"
            hook_out2 = layer.mlp.register_forward_hook(
                lambda mod, inp, out, key=name_out: self._outputs.update({key: (out[0] if isinstance(out, tuple) else out).detach().clone()})
            )
            self._hooks.append(hook_out2)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    @torch.no_grad()
    def estimate(self, dataloader) -> dict:
        """
        Estimate importance scores over calibration data.

        Returns:
            {
              'attention': [imp0, imp1, ...],
              'mlp': [imp0, imp1, ...]
            }
        """
        num_layers = self.model.config.num_hidden_layers
        attn_scores = defaultdict(list)
        mlp_scores = defaultdict(list)

        # Register hooks to capture
        self._register_hooks()

        for batch in tqdm(dataloader, desc="Estimating layer importance"):
            # Move tensors
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            torch.cuda.empty_cache()
            self._inputs.clear()
            self._outputs.clear()

            # Forward pass
            _ = self.model(**batch)

            # Compute importance per layer
            for i in range(num_layers):
                xi = self._inputs.get(f"attn_in_{i}")
                yi = self._outputs.get(f"attn_out_{i}")
                if xi is not None and yi is not None:
                    # Y = X + attention_out
                    Y = xi + yi
                    attn_scores[i].append(calculate_importance(xi, Y))

                xm = self._inputs.get(f"mlp_in_{i}")
                ym = self._outputs.get(f"mlp_out_{i}")
                if xm is not None and ym is not None:
                    Y = xm + ym
                    mlp_scores[i].append(calculate_importance(xm, Y))

            # Cleanup
            self._inputs.clear()
            self._outputs.clear()
            gc.collect()

        # Remove hooks
        self._remove_hooks()

        # Aggregate
        attention_importance = [float(torch.tensor(attn_scores[i]).mean()) if attn_scores[i] else 0.0
                                for i in range(num_layers)]
        mlp_importance = [float(torch.tensor(mlp_scores[i]).mean()) if mlp_scores[i] else 0.0
                          for i in range(num_layers)]

        return {'attention': attention_importance, 'mlp': mlp_importance}

class Qwen2SimilarityLayerEstimator:
    """
    Estimates importance scores for Attention and MLP sublayers in a causal LM.

    For each layer l:
      - X_A^l: input to the attention block
      - Y_A^l: X_A^l + Attention(LN(X_A^l))
      - X_M^l: input to the MLP block
      - Y_M^l: X_M^l + MLP(LN(X_M^l))
    Importance is defined as 1 - cos(X, Y).
    """
    def __init__(self, model, device: str = "cuda"):  # model: Qwen2ForCausalLM or similar
        self.model = model.eval().to(device)
        self.device = device

    def _register_hooks(self):
        """
        Hook inputs and outputs of attention and MLP.
        Stores tensors in self._inputs and self._outputs dicts keyed by names.
        """
        self._inputs = {}
        self._outputs = {}
        self._hooks = []

        for idx, layer in enumerate(self.model.model.layers):
            # Attention input: before input_layernorm
            name_in = f"attn_in_{idx}"
            hook_in = layer.input_layernorm.register_forward_hook(
                lambda mod, inp, out, key=name_in: self._inputs.update({key: inp[0].detach().clone()})
            )
            self._hooks.append(hook_in)
            # Attention output: after self_attn but before residual
            name_out = f"attn_out_{idx}"
            hook_out = layer.self_attn.register_forward_hook(
                lambda mod, inp, out, key=name_out: self._outputs.update({key: (out[0] if isinstance(out, tuple) else out).detach().clone()})
            )
            self._hooks.append(hook_out)
            # MLP input: before post_attention_layernorm
            name_in = f"mlp_in_{idx}"
            hook_in2 = layer.post_attention_layernorm.register_forward_hook(
                lambda mod, inp, out, key=name_in: self._inputs.update({key: inp[0].detach().clone()})
            )
            self._hooks.append(hook_in2)
            # MLP output: after mlp block
            name_out = f"mlp_out_{idx}"
            hook_out2 = layer.mlp.register_forward_hook(
                lambda mod, inp, out, key=name_out: self._outputs.update({key: (out[0] if isinstance(out, tuple) else out).detach().clone()})
            )
            self._hooks.append(hook_out2)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    @torch.no_grad()
    def estimate(self, dataloader) -> dict:
        """
        Estimate importance scores over calibration data.

        Returns:
            {
              'attention': [imp0, imp1, ...],
              'mlp': [imp0, imp1, ...]
            }
        """
        num_layers = self.model.config.num_hidden_layers
        attn_scores = defaultdict(list)
        mlp_scores = defaultdict(list)

        # Register hooks to capture
        self._register_hooks()

        for batch in tqdm(dataloader, desc="Estimating layer importance"):
            # Move tensors
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            torch.cuda.empty_cache()
            self._inputs.clear()
            self._outputs.clear()

            # Forward pass
            _ = self.model(**batch)

            # Compute importance per layer
            for i in range(num_layers):
                xi = self._inputs.get(f"attn_in_{i}")
                yi = self._outputs.get(f"attn_out_{i}")
                if xi is not None and yi is not None:
                    # Y = X + attention_out
                    Y = xi + yi
                    attn_scores[i].append(calculate_importance(xi, Y))

                xm = self._inputs.get(f"mlp_in_{i}")
                ym = self._outputs.get(f"mlp_out_{i}")
                if xm is not None and ym is not None:
                    Y = xm + ym
                    mlp_scores[i].append(calculate_importance(xm, Y))

            # Cleanup
            self._inputs.clear()
            self._outputs.clear()
            gc.collect()

        # Remove hooks
        self._remove_hooks()

        # Aggregate
        attention_importance = [float(torch.tensor(attn_scores[i]).mean()) if attn_scores[i] else 0.0
                                for i in range(num_layers)]
        mlp_importance = [float(torch.tensor(mlp_scores[i]).mean()) if mlp_scores[i] else 0.0
                          for i in range(num_layers)]

        return {'attention': attention_importance, 'mlp': mlp_importance}