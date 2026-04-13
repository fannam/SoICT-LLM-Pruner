from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from carve_lm.llm.core import AttentionPasser, FeedForwardPasser
from carve_lm.llm.estimators import LayerPerplexityEstimator, create_estimator


class ToyAttention(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states, None


class ToyMLP(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()


class ToyBackbone(nn.Module):
    def __init__(self, num_hidden_layers: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([ToyLayer() for _ in range(num_hidden_layers)])
        self.norm = nn.Identity()


class TokenLossComponentToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=4,
            vocab_size=8,
            head_dim=4,
        )
        self.model = ToyBackbone(
            num_hidden_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
        )
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None, use_cache=False, **kwargs):
        expected_labels = input_ids.clone()
        if attention_mask is not None:
            expected_labels = expected_labels.masked_fill(attention_mask == 0, -100)
        assert torch.equal(labels, expected_labels)

        loss = 1.0
        if isinstance(self.model.layers[0].self_attn, AttentionPasser):
            loss += 0.5
        if isinstance(self.model.layers[1].self_attn, AttentionPasser):
            loss += 1.5
        if isinstance(self.model.layers[0].mlp, FeedForwardPasser):
            loss += 2.0
        if isinstance(self.model.layers[1].mlp, FeedForwardPasser):
            loss += 4.0

        return SimpleNamespace(
            loss=torch.tensor(loss, dtype=torch.float32, device=input_ids.device)
        )


def test_layer_perplexity_estimator_scores_attention_and_mlp_components():
    model = TokenLossComponentToyModel()
    dataloader = DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
            }
        ],
        batch_size=1,
    )

    estimator = LayerPerplexityEstimator(model, tokenizer=object(), device="cpu")
    importance = estimator.estimate(dataloader, n_samples=1)

    baseline = torch.exp(torch.tensor(1.0))
    expected_attention = torch.tensor(
        [
            torch.exp(torch.tensor(1.5)) - baseline,
            torch.exp(torch.tensor(2.5)) - baseline,
        ]
    )
    expected_mlp = torch.tensor(
        [
            torch.exp(torch.tensor(3.0)) - baseline,
            torch.exp(torch.tensor(5.0)) - baseline,
        ]
    )

    torch.testing.assert_close(torch.tensor(importance["attention"]), expected_attention)
    torch.testing.assert_close(torch.tensor(importance["mlp"]), expected_mlp)


def test_create_estimator_resolves_layer_perplexity():
    estimator = create_estimator(
        "perplexity.layer",
        TokenLossComponentToyModel(),
        tokenizer=object(),
        device="cpu",
    )

    assert isinstance(estimator, LayerPerplexityEstimator)
