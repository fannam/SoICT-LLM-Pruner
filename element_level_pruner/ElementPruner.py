import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, Qwen2ForCausalLM
import numpy as np
from typing import Dict, List

class ElementPruner:
    def __init__(self, original_model, dtype, device='cuda'):
        # store original model & config, derive dims
        self.model = original_model
        self.device = device
        self.original_config = original_model.config
        self.head_dim = self.original_config.hidden_size//self.original_config.num_attention_heads
        self.original_num_key_value_heads = self.original_config.num_key_value_heads
        self.original_num_attention_heads = self.original_config.num_attention_heads
        self.original_num_layers = self.original_config.num_hidden_layers
        self.original_intermediate_size = self.original_config.intermediate_size
        self.original_hidden_size = self.original_config.hidden_size
        self.dtype = dtype

    def prune_attention_query(self, head_importance, target_num_attention_heads):
        """
        Args:
        - head_importance: Dictionary map int to tensor of shape config.num_attention_heads
        - target_num_attention_heads: New number of query heads in GQA
        """
        self.model.to(self.device)
        if target_num_attention_heads%self.original_num_key_value_heads!=0:
            print(f"Warning: Number of query heads is invalid.")
            return None
        if target_num_attention_heads == self.original_num_attention_heads:
            print("Warning: No head to remove.")
            return None
        if target_num_attention_heads>self.original_num_attention_heads:
            print("Warning: New number of attetion heads must not be larger than the original one, retruning original model.")
            return None
        if target_num_attention_heads<self.original_num_key_value_heads:
            print(f"Warning: number of attention heads must be at least {self.original_num_key_value_heads}.")
            return None
        if isinstance(self.model)!=LlamaForCausalLM and isinstance(self.model)!=Qwen2ForCausalLM:
            print(f"Warning: Not supported model type! Please use LlamaForCausalLM or Qwen2ForCausalLM.")
            return None
        pruned_weights = {}
        print("Calculating pruned weights...")
        for i in range(self.original_num_layers):
            layer_name = i
            if layer_name not in head_importance:
                print(f"Warning: Importance scores for layer {layer_name} not found. Cannot prune layer {i}.")
                continue
            try:
                attention_module = self.model.model.layers[i].self_attn
                orig_q_proj = attention_module.q_proj
                orig_o_proj = attention_module.o_proj
            except (AttributeError, IndexError):
                print(f"Warning: Could not access attention module/projections for layer {i}. Skipping.")
                continue

            importance_scores = head_importance[layer_name].to(self.device)
            indices_to_keep = torch.sort(torch.argsort(importance_scores, descending=True)[:target_num_attention_heads]).values

            head_mask = torch.zeros(self.original_num_attention_heads * self.head_dim, dtype=torch.bool, device=self.device)
            for head_idx in indices_to_keep:
                head_mask[head_idx * self.head_dim : (head_idx + 1) * self.head_dim] = True

            new_q_weight = orig_q_proj.weight.data[head_mask, :]
            new_q_bias = None
            if orig_q_proj.bias is not None:
                new_q_bias = orig_q_proj.bias.data[head_mask]

            new_o_weight = orig_o_proj.weight.data[:, head_mask]
            new_o_bias = None
            if orig_o_proj.bias is not None:
                new_o_bias = orig_o_proj.bias.data 

            pruned_weights[i] = {
                'q_weight': new_q_weight,
                'o_weight': new_o_weight,
                'q_bias': new_q_bias,
                'o_bias': new_o_bias,
            }
        print("Creating new model configuration")
        new_config = copy.deepcopy(self.original_config)
        new_config.num_attention_heads = target_num_attention_heads
        new_model = None
        if isinstance(self.model, LlamaForCausalLM):
            new_model = LlamaForCausalLM(new_config)
            new_model.to(self.device)
            new_model.to(self.dtype)
        elif isinstance(self.model, Qwen2ForCausalLM):
            new_model = Qwen2ForCausalLM(new_config)
            new_model.to(self.device)
            new_model.to(self.dtype)
        else:
            print("Unsupported model type.")
            return None
        print("New model instantiated.")
        print("Loading weights into the new model...")
        original_state_dict = self.model.state_dict()
        new_state_dict = new_model.state_dict()
        loaded_keys = set()

        for key in new_state_dict.keys():
            is_pruned_q = ".self_attn.q_proj." in key
            is_pruned_o = ".self_attn.o_proj." in key
            layer_idx_str = key.split('.')[2] if key.startswith("model.layers.") else None

            layer_idx = -1
            if layer_idx_str is not None and layer_idx_str.isdigit():
                layer_idx = int(layer_idx_str)

            if layer_idx in pruned_weights and (is_pruned_q or is_pruned_o):
                proj_type = 'q' if is_pruned_q else 'o'
                bias_type = 'weight' if 'weight' in key else 'bias'
                pruned_tensor = pruned_weights[layer_idx][f'{proj_type}_{bias_type}']

                if pruned_tensor is not None:
                    if new_state_dict[key].shape == pruned_tensor.shape:
                        new_state_dict[key].copy_(pruned_tensor)
                    else:
                        print(f"ERROR: Shape mismatch for {key}. Expected {new_state_dict[key].shape}, got {pruned_tensor.shape}")
                elif bias_type == 'bias' and new_state_dict[key] is not None:
                    print(f"Warning: Original model had no bias for {key.replace('.bias', '')}, but new model expects one? Check config.")
                elif bias_type == 'bias' and new_state_dict[key] is None:
                    pass
                else:
                    print(f"ERROR: Pruned tensor for {key} is None unexpectedly.")

            elif key in original_state_dict:
                if new_state_dict[key].shape == original_state_dict[key].shape:
                    new_state_dict[key].copy_(original_state_dict[key])
                else:
                    print(f"ERROR: Shape mismatch for {key}. Expected {new_state_dict[key].shape}, got {original_state_dict[key].shape} alaalala")
            else:
                print(f"ERROR: Key {key} not found in original state_dict.")

            loaded_keys.add(key)

        if len(loaded_keys) != len(new_state_dict):
            print(f"Warning: Processed {len(loaded_keys)} keys, but new state dict has {len(new_state_dict)} keys.")
            missing_keys = set(new_state_dict.keys()) - loaded_keys
            print(f"Missing keys potentially: {missing_keys}")

        try:
            new_model.load_state_dict(new_state_dict, strict=True)
            print("Successfully loaded state dict into the new model.")
        except RuntimeError as e:
            print(f"ERROR loading state dict: {e}")
            print("Check for shape mismatches or missing/unexpected keys.")
            raise

        print("--- Pruning and Rebuilding Process Complete ---")
        return new_model

    def prune_attention_group(self, head_importance, target_group):
        if isinstance(self.model)!=LlamaForCausalLM and isinstance(self.model)!=Qwen2ForCausalLM:
            print(f"Warning: Not supported model type! Please use LlamaForCausalLM or Qwen2ForCausalLM")
            return None
        self.model.to(self.device)
        query_head_per_group = self.original_num_attention_heads // self.original_num_key_value_heads
        original_group_count = self.original_num_key_value_heads
        if target_group >= original_group_count:
            print("Warning: target_group is larger or equal original group. No pruning is needed.")
            return None
        elif target_group <= 0:
            print("Warning: target_group must be a positive integer.")
            return None
        new_config = copy.deepcopy(self.original_config)
        new_config.num_attention_heads = target_group * query_head_per_group
        new_config.num_key_value_heads = target_group
        new_model = None
        print("Creating pruned model...")
        if isinstance(self.model, LlamaForCausalLM):
            new_model = LlamaForCausalLM(new_config)
            new_model.to(self.device)
            new_model.to(self.dtype)
        elif isinstance(self.model, Qwen2ForCausalLM):
            new_model = Qwen2ForCausalLM(new_config)
            new_model.to(self.device)
            new_model.to(self.dtype)
        else:
            print("Unsupported model type.")
            return None
        def _compute_group_importance_per_layer(
            head_importance: Dict[int, torch.Tensor],
            orig_groups: int
        ) -> Dict[int, torch.Tensor]:
            """
            Given:
              - head_importance: mapping layer_idx → Tensor of shape (num_q_heads,)
              - orig_groups:      number of KV-head groups per layer
            Returns:
              - layer_idx → Tensor of shape (orig_groups,), where each entry is the
                mean importance of that group’s query heads in that layer.
            """
            per_layer_group_imp: Dict[int, torch.Tensor] = {}
            for layer_idx, imp in head_importance.items():
                grouped = imp.view(orig_groups, query_head_per_group)
                per_layer_group_imp[layer_idx] = grouped.mean(dim=1)
            return per_layer_group_imp
        def _get_keep_indices_group(head_imp):
            per_layer_group_importance = _compute_group_importance_per_layer(head_importance=head_imp, orig_groups=original_group_count)
            keep_indices_group = {}
            for layer_idx in range(self.original_num_layers):
                topk = torch.topk(per_layer_group_importance[layer_idx], k=target_group)
                topk_indices_np = topk.indices.numpy()
                topk_indices_np.sort()
                keep_indices_group[layer_idx] = topk_indices_np
            return keep_indices_group
        print("Getting key value heads indices to keep...")
        keep_key_value_heads_indices = _get_keep_indices_group(head_importance)
        #print(keep_key_value_heads_indices)
        keep_query_heads_indices = {}
        for layer_idx in keep_key_value_heads_indices:
            tmp_keep_query_indices = []
            for group_idx in keep_key_value_heads_indices[layer_idx]:
                start = group_idx * query_head_per_group
                end = start + query_head_per_group
                tmp_keep_query_indices.extend(range(start, end))
            keep_query_heads_indices[layer_idx] = np.array(tmp_keep_query_indices)
        #print(keep_query_heads_indices)
        pruned = {}
        print("Calculating new weight...")
        for layer_idx in range(self.original_num_layers):
            try:
                attention_module = self.model.model.layers[layer_idx].self_attn
            except (AttributeError, IndexError):
                print(f"Warning: Could not access attention module/projections for layer {layer_idx}. Skipping.")
                continue
            old_q_weight, old_q_bias = attention_module.q_proj.weight.data, attention_module.q_proj.bias.data if attention_module.q_proj.bias is not None else None
            old_k_weight, old_k_bias = attention_module.k_proj.weight.data, attention_module.k_proj.bias.data if attention_module.k_proj.bias is not None else None
            old_v_weight, old_v_bias = attention_module.v_proj.weight.data, attention_module.v_proj.bias.data if attention_module.v_proj.bias is not None else None
            old_o_weight, old_o_bias = attention_module.o_proj.weight.data, attention_module.o_proj.bias.data if attention_module.o_proj.bias is not None else None

            group_q_mask = torch.zeros(self.original_num_attention_heads * self.head_dim, dtype=torch.bool, device=self.device)
            group_kv_mask = torch.zeros(original_group_count * self.head_dim, dtype=torch.bool, device=self.device)

            for query_idx in keep_query_heads_indices[layer_idx]:
                group_q_mask[self.head_dim * query_idx : self.head_dim * (query_idx + 1)] = True
            for kv_idx in keep_key_value_heads_indices[layer_idx]:
                group_kv_mask[self.head_dim * kv_idx : self.head_dim * (kv_idx + 1)] = True
            # new_q_weight = old_q_weight[group_q_mask, :]
            # print(old_q_weight[0:64]-new_q_weight[0:64])
            pruned[layer_idx] = {
                'qW': old_q_weight[group_q_mask, :],
                'qb': old_q_bias[group_q_mask] if old_q_bias is not None else None,
                'kW': old_k_weight[group_kv_mask, :],
                'kb': old_k_bias[group_kv_mask] if old_k_bias is not None else None,
                'vW': old_v_weight[group_kv_mask, :],
                'vb': old_v_bias[group_kv_mask] if old_v_bias is not None else None,
                'oW': old_o_weight[:, group_q_mask],
                'ob': old_o_bias,
            }
        new_state_dict = new_model.state_dict()
        original_state_dict = self.model.state_dict()
        print("Transferring weight...")
        for key in new_state_dict.keys():
            parts = key.split('.')
            if parts[:2] == ['model','layers']:
                print("Found supported model!")
                layer_idx = int(parts[2])
                attn = self.model.model.layers[layer_idx].self_attn
                if 'self_attn.q_proj.weight' in key:
                    new_state_dict[key].copy_(pruned[layer_idx]['qW'])
                    continue
                if 'self_attn.q_proj.bias' in key and pruned[layer_idx]['qb'] is not None:
                    new_state_dict[key].copy_(pruned[layer_idx]['qb'])
                    continue
                if 'self_attn.k_proj.weight' in key:
                    new_state_dict[key].copy_(pruned[layer_idx]['kW'])
                    continue
                if 'self_attn.k_proj.bias' in key and pruned[layer_idx]['kb'] is not None:
                    new_state_dict[key].copy_(pruned[layer_idx]['kb'])
                    continue
                if 'self_attn.v_proj.weight' in key:
                    new_state_dict[key].copy_(pruned[layer_idx]['vW'])
                    continue
                if 'self_attn.v_proj.bias' in key and pruned[layer_idx]['vb'] is not None:
                    new_state_dict[key].copy_(pruned[layer_idx]['vb'])
                    continue
                if 'self_attn.o_proj.weight' in key:
                    new_state_dict[key].copy_(pruned[layer_idx]['oW'])
                    continue
                if 'self_attn.o_proj.bias' in key and pruned[layer_idx]['ob'] is not None:
                    new_state_dict[key].copy_(pruned[layer_idx]['ob'])
                    continue
            if key in original_state_dict and new_state_dict[key].shape == original_state_dict[key].shape:
                new_state_dict[key].copy_(original_state_dict[key])

        new_model.load_state_dict(new_state_dict, strict=True)
        print("Pruned successfully!")
        print(new_model)
        return new_model

    def prune_mlp(self, neuron_importance, target_num_neurons):
        if target_num_neurons >= self.original_intermediate_size:
            print(f"Warning: target intermediate size: f{target_num_neurons} is larger or equal original intermediate size: {self.original_intermediate_size}. No prune is needed.")
            return None
        if target_num_neurons <= 0:
            print(f"Warning: target intermediate size must be a positive integer")
            return None
        pruned_model = copy.deepcopy(self.model).to(self.device)
        pruned_model.config.intermediate_size = target_num_neurons
        keep = {}
        print("Checking size...")
        for idx, scores in neuron_importance.items():
            if target_num_neurons > scores.numel():
                raise ValueError(f"new_intermediate_size={target_num_neurons} > layer {idx}")
            topk = torch.topk(scores, target_num_neurons, largest=True).indices.tolist()
            keep[idx] = sorted(topk)
        print("Pruning...")
        for idx, layer in enumerate(pruned_model.model.layers):
            hid_idx = keep.get(idx, list(range(self.original_intermediate_size)))
            if len(hid_idx) == self.original_intermediate_size:
                continue
            mlp = layer.mlp
            for name in ('gate_proj', 'up_proj'):
                proj = getattr(mlp, name)
                W = proj.weight.data[hid_idx].to(self.dtype)
                b = proj.bias.data[hid_idx].to(self.dtype) if proj.bias is not None else None
                new_linear = nn.Linear(
                    in_features=proj.in_features,
                    out_features=target_num_neurons,
                    bias=(b is not None)
                ).to(self.device).to(self.dtype)
                new_linear.weight.data.copy_(W)
                if b is not None:
                    new_linear.bias.data.copy_(b)
                setattr(mlp, name, new_linear)
            
            old_down_proj = mlp.down_proj
            Wd = old_down_proj.weight.data[:, hid_idx].to(self.dtype)
            bd = old_down_proj.bias.data.clone().to(self.dtype) if old_down_proj.bias is not None else None
            new_down = nn.Linear(
                in_features=target_num_neurons,
                out_features=old_down_proj.out_features,
                bias=(b is not None)
            ).to(self.device).to(self.dtype)
            new_down.weight.data.copy_(Wd)
            if bd is not None:
                new_down.bias.data.copy_(bd)
            mlp.down_proj = new_down
        print("Pruned successfully")
        return pruned_model

    def prune_embeddings(self, embedding_importance, target_embedding_size):
        if isinstance(self.model)!=LlamaForCausalLM and isinstance(self.model)!=Qwen2ForCausalLM:
            print(f"Warning: Not supported model type! Please use LlamaForCausalLM or Qwen2ForCausalLM")
            return None
        if target_embedding_size >= self.original_hidden_size:
            print(f"Warning: target embedding size: {target_embedding_size} is larger or equal original embedding size: {self.original_hidden_size}. No prune is needed. Returning original model")
            return None
        if target_embedding_size <= 0:
            print(f"Warning: target embedding size must be a positive integer")
            return None
        topk = torch.topk(embedding_importance, k=target_embedding_size)
        topk_indices_np = topk.indices.numpy()
        pruned_config = copy.deepcopy(self.model.config)
        pruned_config.hidden_size = target_embedding_size
        pruned_config.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        pruned_model = self.model.__class__(pruned_config).to(self.dtype)

        def _get_keep_indices():
            topk = torch.topk(embedding_importance, k=target_embedding_size)
            topk_indices_np = topk.indices.numpy()
            topk_indices_np.sort()
            return topk_indices_np

        def _transfer_embeddings(keep_indices):
            print("Transferring embedding weight...")
            orig_w = self.model.model.embed_tokens.weight.data
            pruned_model.model.embed_tokens.weight.data = orig_w[:, keep_indices].clone()

        def _transfer_input_layernorm(keep_indices):
            print("Transferring input layernorm weight...")
            for i, layer in enumerate(self.model.model.layers):
                orig_w = layer.input_layernorm.weight.data
                pruned_model.model.layers[i].input_layernorm.weight.data = orig_w[keep_indices].clone()

        def _transfer_self_attn(keep_indices):
            print("Transferring self attention weight...")
            for i, (orig_layer, pruned_layer) in enumerate(zip(self.model.model.layers, pruned_model.model.layers)):
                pruned_layer.self_attn.q_proj.weight.data = orig_layer.self_attn.q_proj.weight.data[:, keep_indices].clone()
                pruned_layer.self_attn.k_proj.weight.data = orig_layer.self_attn.k_proj.weight.data[:, keep_indices].clone()
                pruned_layer.self_attn.v_proj.weight.data = orig_layer.self_attn.v_proj.weight.data[:, keep_indices].clone()
                pruned_layer.self_attn.o_proj.weight.data = orig_layer.self_attn.o_proj.weight.data[keep_indices, :].clone()
                if orig_layer.self_attn.o_proj.bias is not None:
                    pruned_layer.self_attn.o_proj.bias.data = orig_layer.self_attn.o_proj.bias.data[keep_indices].clone()

        def _transfer_outside_layernorm(keep_indices):
            print("Transferring outside layernorm weight...")
            orig_w = self.model.model.norm.weight.data
            pruned_model.model.norm.weight.data = orig_w[keep_indices].clone()

        def _transfer_post_attention_norm(keep_indices):
            print("Transferring post attention layernorm weight...")
            for i, layer in enumerate(self.model.model.layers):
                orig_w = layer.post_attention_layernorm.weight.data
                pruned_model.model.layers[i].post_attention_layernorm.weight.data = orig_w[keep_indices].clone()

        def _transfer_mlp(keep_indices):
            print("Transferring MLP weight...")
            for i, (orig_layer, pruned_layer) in enumerate(zip(self.model.model.layers, pruned_model.model.layers)):
                # gate_proj and up_proj: prune columns of input dim
                orig_gate_w = orig_layer.mlp.gate_proj.weight.data
                pruned_layer.mlp.gate_proj.weight.data = orig_gate_w[:, keep_indices].clone()
                if orig_layer.mlp.gate_proj.bias is not None:
                    pruned_layer.mlp.gate_proj.bias.data = orig_layer.mlp.gate_proj.bias.data.clone()

                orig_up_w = orig_layer.mlp.up_proj.weight.data
                pruned_layer.mlp.up_proj.weight.data = orig_up_w[:, keep_indices].clone()
                if orig_layer.mlp.up_proj.bias is not None:
                    pruned_layer.mlp.up_proj.bias.data = orig_layer.mlp.up_proj.bias.data.clone()

                # down_proj: prune rows of output dim
                orig_down_w = orig_layer.mlp.down_proj.weight.data
                pruned_layer.mlp.down_proj.weight.data = orig_down_w[keep_indices, :].clone()
                if orig_layer.mlp.down_proj.bias is not None:
                    pruned_layer.mlp.down_proj.bias.data = orig_layer.mlp.down_proj.bias.data[keep_indices].clone()

        keep_indices = _get_keep_indices()
        _transfer_embeddings(keep_indices)
        _transfer_input_layernorm(keep_indices)
        _transfer_outside_layernorm(keep_indices)
        _transfer_post_attention_norm(keep_indices)
        _transfer_self_attn(keep_indices)
        _transfer_mlp(keep_indices)
        pruned_model.config.hidden_size = target_embedding_size
        print("Pruned successfully!")
        print(pruned_model)
        return pruned_model