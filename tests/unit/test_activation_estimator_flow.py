from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from carve_lm.adapters import AttentionProjectionBundle, BaseModelAdapter, MLPProjectionBundle
from carve_lm.estimators import ActivationElementEstimator


class IdentityNorm(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class SyntheticAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        qkv_hidden = config.head_dim * config.num_attention_heads
        kv_hidden = config.head_dim * config.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, qkv_hidden, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_hidden, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_hidden, bias=False)
        self.o_proj = nn.Linear(qkv_hidden, config.hidden_size, bias=False)

        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(config.hidden_size))

    def forward(self, hidden_states):
        expanded = hidden_states.new_zeros(
            *hidden_states.shape[:-1],
            hidden_states.size(-1) * 2,
        )
        expanded[..., ::2] = hidden_states
        context = expanded[..., ::2]
        return self.o_proj(context)


class SyntheticMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        with torch.no_grad():
            self.down_proj.weight.zero_()
            self.down_proj.weight[0, 0] = 1.0
            self.down_proj.weight[1, 1] = 1.0
            self.down_proj.weight[2, 2] = 1.0

    def forward(self, hidden_states):
        intermediate = hidden_states[..., : self.down_proj.in_features]
        return self.down_proj(intermediate)


class SyntheticLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = IdentityNorm()
        self.post_attention_layernorm = IdentityNorm()
        self.self_attn = SyntheticAttention(config)
        self.mlp = SyntheticMLP(config)


class SyntheticBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SyntheticLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = IdentityNorm()


class SyntheticModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SyntheticBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, **kwargs):
        hidden_states = self.model.embed_tokens(input_ids)

        for layer in self.model.layers:
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.self_attn(hidden_states)
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)

        hidden_states = self.model.norm(hidden_states)
        return {"last_hidden_state": hidden_states, "logits": self.lm_head(hidden_states)}


class SyntheticModelAdapter(BaseModelAdapter):
    def __init__(self):
        super().__init__(name="synthetic-activation", model_cls=SyntheticModel)

    def get_layers(self, model: SyntheticModel):
        self.ensure_supported(model)
        return model.model.layers

    def get_embed_tokens(self, model: SyntheticModel) -> nn.Embedding:
        self.ensure_supported(model)
        return model.model.embed_tokens

    def get_lm_head(self, model: SyntheticModel) -> nn.Module | None:
        self.ensure_supported(model)
        return model.lm_head

    def get_final_norm(self, model: SyntheticModel) -> nn.Module:
        self.ensure_supported(model)
        return model.model.norm

    def get_input_layernorm(self, layer: SyntheticLayer) -> nn.Module:
        return layer.input_layernorm

    def get_post_attention_layernorm(self, layer: SyntheticLayer) -> nn.Module:
        return layer.post_attention_layernorm

    def get_attention_module(self, layer: SyntheticLayer) -> nn.Module:
        return layer.self_attn

    def set_attention_module(self, layer: SyntheticLayer, module: nn.Module) -> None:
        layer.self_attn = module

    def get_mlp_module(self, layer: SyntheticLayer) -> nn.Module:
        return layer.mlp

    def set_mlp_module(self, layer: SyntheticLayer, module: nn.Module) -> None:
        layer.mlp = module

    def get_attention_projections(self, layer: SyntheticLayer) -> AttentionProjectionBundle:
        attention = self.get_attention_module(layer)
        return AttentionProjectionBundle(
            q_proj=attention.q_proj,
            k_proj=attention.k_proj,
            v_proj=attention.v_proj,
            o_proj=attention.o_proj,
        )

    def get_mlp_projections(self, layer: SyntheticLayer) -> MLPProjectionBundle:
        mlp = self.get_mlp_module(layer)
        return MLPProjectionBundle(
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
        )

    def set_attention_projection(
        self,
        layer: SyntheticLayer,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        setattr(layer.self_attn, projection_name, projection)

    def set_mlp_projection(
        self,
        layer: SyntheticLayer,
        projection_name: str,
        projection: nn.Module,
    ) -> None:
        setattr(layer.mlp, projection_name, projection)


def make_config(**overrides):
    config = SimpleNamespace(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=3,
        head_dim=2,
        vocab_size=16,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_model(**config_overrides) -> SyntheticModel:
    return SyntheticModel(make_config(**config_overrides))


def set_token_vectors(model: SyntheticModel, token_vectors: dict[int, list[float]]) -> None:
    with torch.no_grad():
        model.model.embed_tokens.weight.zero_()
        for token_id, values in token_vectors.items():
            model.model.embed_tokens.weight[token_id] = torch.tensor(
                values,
                dtype=model.model.embed_tokens.weight.dtype,
            )


def make_dataloader(token_ids: list[int]) -> DataLoader:
    return DataLoader(
        [{"input_ids": torch.tensor(token_ids, dtype=torch.long)}],
        batch_size=1,
    )


def make_adapter():
    return SyntheticModelAdapter()


def test_activation_mlp_neuron_aggregations_match_global_statistics():
    adapter = make_adapter()
    model = make_model()
    set_token_vectors(
        model,
        {
            0: [1.0, 2.0, 3.0, 4.0],
            1: [5.0, 6.0, 7.0, 8.0],
        },
    )

    dataloader = make_dataloader([0, 1])
    estimator = ActivationElementEstimator(model, device="cpu", model_adapter=adapter)

    expected_by_agg = {
        "sum": torch.tensor([6.0, 8.0, 10.0]),
        "mean": torch.tensor([3.0, 4.0, 5.0]),
        "l2": torch.sqrt(torch.tensor([26.0, 40.0, 58.0])),
        "var": torch.tensor([8.0, 8.0, 8.0]),
    }

    for agg, expected in expected_by_agg.items():
        scores = estimator.estimate_mlp_neurons(dataloader, agg=agg)
        torch.testing.assert_close(scores[0], expected)


def test_activation_attention_heads_support_non_contiguous_o_proj_inputs():
    adapter = make_adapter()
    model = make_model()
    set_token_vectors(
        model,
        {
            0: [3.0, 4.0, 5.0, 12.0],
            1: [8.0, 15.0, 7.0, 24.0],
        },
    )

    estimator = ActivationElementEstimator(model, device="cpu", model_adapter=adapter)
    scores = estimator.estimate_attention_heads(make_dataloader([0, 1]), agg="sum")

    torch.testing.assert_close(scores[0], torch.tensor([22.0, 38.0]))


def test_activation_attention_groups_compute_direct_group_scores():
    adapter = make_adapter()
    model = make_model(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
    )
    set_token_vectors(
        model,
        {
            0: [3.0, 4.0, 6.0, 8.0, 0.0, 5.0, 0.0, 12.0],
            1: [8.0, 15.0, 9.0, 12.0, 7.0, 24.0, 0.0, 0.0],
        },
    )

    estimator = ActivationElementEstimator(model, device="cpu", model_adapter=adapter)
    scores = estimator.estimate_attention_groups(make_dataloader([0, 1]), agg="sum")

    expected = torch.tensor(
        [
            torch.linalg.vector_norm(torch.tensor([3.0, 4.0, 6.0, 8.0]))
            + torch.linalg.vector_norm(torch.tensor([8.0, 15.0, 9.0, 12.0])),
            torch.linalg.vector_norm(torch.tensor([0.0, 5.0, 0.0, 12.0]))
            + torch.linalg.vector_norm(torch.tensor([7.0, 24.0, 0.0, 0.0])),
        ]
    )
    torch.testing.assert_close(scores[0], expected)


def test_activation_embedding_channels_are_reported_per_norm_site():
    adapter = make_adapter()
    model = make_model()
    set_token_vectors(
        model,
        {
            0: [1.0, 2.0, 3.0, 4.0],
            1: [5.0, 6.0, 7.0, 8.0],
        },
    )

    estimator = ActivationElementEstimator(model, device="cpu", model_adapter=adapter)
    scores = estimator.estimate_embedding_channels(make_dataloader([0, 1]), agg="mean")

    assert set(scores) == {"input_layernorm_0", "post_attention_layernorm_0", "final_norm"}
    torch.testing.assert_close(scores["input_layernorm_0"], torch.tensor([3.0, 4.0, 5.0, 6.0]))
    torch.testing.assert_close(scores["post_attention_layernorm_0"], torch.tensor([3.0, 4.0, 5.0, 6.0]))
    torch.testing.assert_close(scores["final_norm"], torch.tensor([3.0, 4.0, 5.0, 0.0]))


def test_activation_variance_is_zero_for_single_observation():
    adapter = make_adapter()
    model = make_model()
    set_token_vectors(
        model,
        {
            0: [1.0, 2.0, 3.0, 4.0],
        },
    )

    estimator = ActivationElementEstimator(model, device="cpu", model_adapter=adapter)
    scores = estimator.estimate_mlp_neurons(make_dataloader([0]), agg="var")

    torch.testing.assert_close(scores[0], torch.zeros(3))
