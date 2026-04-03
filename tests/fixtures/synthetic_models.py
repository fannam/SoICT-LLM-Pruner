from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SyntheticConfig:
    hidden_size: int = 8
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 6
    vocab_size: int = 16
    head_dim: int = 2
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "SyntheticConfig":
        return cls(**payload)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class SyntheticNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return hidden_states * self.weight


class SyntheticAttention(nn.Module):
    def __init__(self, config: SyntheticConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        if self.num_heads != self.num_key_value_heads:
            k = k.repeat_interleave(self.num_key_value_groups, dim=2)
            v = v.repeat_interleave(self.num_key_value_groups, dim=2)
        context = torch.tanh(q + k + v).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(context)


class SyntheticMLP(nn.Module):
    def __init__(self, config: SyntheticConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class SyntheticLayer(nn.Module):
    def __init__(self, config: SyntheticConfig):
        super().__init__()
        self.input_layernorm = SyntheticNorm(config.hidden_size)
        self.post_attention_layernorm = SyntheticNorm(config.hidden_size)
        self.self_attn = SyntheticAttention(config)
        self.mlp = SyntheticMLP(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class SyntheticBackbone(nn.Module):
    def __init__(self, config: SyntheticConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SyntheticLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = SyntheticNorm(config.hidden_size)


class SyntheticCausalLM(nn.Module):
    def __init__(self, config: SyntheticConfig):
        super().__init__()
        self.config = config
        self.model = SyntheticBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        outputs = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            outputs["loss"] = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
        return outputs
