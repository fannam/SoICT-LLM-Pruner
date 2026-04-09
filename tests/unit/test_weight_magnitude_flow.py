from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from carve_lm.adapters import AttentionProjectionBundle, BaseModelAdapter, MLPProjectionBundle
from carve_lm.estimators import WeightMagnitudeElementEstimator
from carve_lm.pruners import ElementPruner


class WeightOnlyNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))

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


class SyntheticMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)


class SyntheticLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = WeightOnlyNorm(config.hidden_size)
        self.post_attention_layernorm = WeightOnlyNorm(config.hidden_size)
        self.self_attn = SyntheticAttention(config)
        self.mlp = SyntheticMLP(config)


class SyntheticBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SyntheticLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = WeightOnlyNorm(config.hidden_size)


class SyntheticModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SyntheticBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, **kwargs):
        hidden_states = self.model.embed_tokens(input_ids)
        return {"last_hidden_state": hidden_states, "logits": self.lm_head(hidden_states)}


class SyntheticModelAdapter(BaseModelAdapter):
    def __init__(self):
        super().__init__(name="synthetic", model_cls=SyntheticModel)

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
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=3,
        head_dim=2,
        vocab_size=5,
        tie_word_embeddings=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_model(*, tie_word_embeddings: bool = False) -> SyntheticModel:
    return SyntheticModel(make_config(tie_word_embeddings=tie_word_embeddings))


def zero_parameters(model: nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()


def fill_attention_group_markers(model: SyntheticModel) -> None:
    zero_parameters(model)
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        attention.q_proj.weight[0, 0] = 1.0
        attention.q_proj.weight[4, 0] = 2.0
        attention.k_proj.weight[0, 0] = 10.0
        attention.k_proj.weight[2, 0] = 20.0
        attention.v_proj.weight[0, 0] = 100.0
        attention.v_proj.weight[2, 0] = 200.0
        attention.o_proj.weight[0, 0] = 1000.0
        attention.o_proj.weight[0, 4] = 2000.0


def assert_group_one_kept(source_model: SyntheticModel, pruned_model: SyntheticModel) -> None:
    source_attention = source_model.model.layers[0].self_attn
    pruned_attention = pruned_model.model.layers[0].self_attn
    torch.testing.assert_close(pruned_attention.q_proj.weight, source_attention.q_proj.weight[4:8, :])
    torch.testing.assert_close(pruned_attention.k_proj.weight, source_attention.k_proj.weight[2:4, :])
    torch.testing.assert_close(pruned_attention.v_proj.weight, source_attention.v_proj.weight[2:4, :])
    torch.testing.assert_close(pruned_attention.o_proj.weight, source_attention.o_proj.weight[:, 4:8])


@pytest.fixture
def adapter():
    return SyntheticModelAdapter()


def test_estimate_attention_heads_scores_query_slices_only(adapter):
    model = make_model()
    zero_parameters(model)
    attention = model.model.layers[0].self_attn

    with torch.no_grad():
        attention.q_proj.weight[0, 0] = 1.0
        attention.q_proj.weight[1, 0] = 2.0
        attention.q_proj.weight[2, 0] = 4.0
        attention.q_proj.weight[4, 0] = 5.0
        attention.q_proj.weight[6, 0] = 6.0

        attention.k_proj.weight[0, 0] = 1000.0
        attention.v_proj.weight[0, 0] = 2000.0

        attention.o_proj.weight[0, 0] = 10.0
        attention.o_proj.weight[0, 2] = 20.0
        attention.o_proj.weight[0, 4] = 30.0
        attention.o_proj.weight[0, 6] = 40.0

    estimator = WeightMagnitudeElementEstimator(model, device="cpu", model_adapter=adapter)
    scores = estimator.estimate_attention_heads(agg="l1")

    torch.testing.assert_close(scores[0], torch.tensor([13.0, 24.0, 35.0, 46.0]))


def test_estimate_attention_groups_scores_group_slices_only(adapter):
    model = make_model()
    fill_attention_group_markers(model)

    estimator = WeightMagnitudeElementEstimator(model, device="cpu", model_adapter=adapter)
    scores = estimator.estimate_attention_groups(agg="l1")

    assert scores[0].shape == (2,)
    torch.testing.assert_close(scores[0], torch.tensor([1111.0, 2222.0]))


def test_estimate_mlp_neurons_scores_coupled_slices(adapter):
    model = make_model()
    zero_parameters(model)
    mlp = model.model.layers[0].mlp

    with torch.no_grad():
        mlp.gate_proj.weight[0, 0] = 1.0
        mlp.gate_proj.weight[1, 0] = 50.0
        mlp.up_proj.weight[0, 0] = 100.0
        mlp.up_proj.weight[1, 0] = 1.0
        mlp.down_proj.weight[0, 0] = 1000.0
        mlp.down_proj.weight[0, 1] = 1.0

    estimator = WeightMagnitudeElementEstimator(model, device="cpu", model_adapter=adapter)
    scores = estimator.estimate_mlp_neurons(agg="l1")

    torch.testing.assert_close(scores[0], torch.tensor([1101.0, 52.0, 0.0]))
    assert torch.argmax(scores[0]).item() == 0


def test_estimate_embedding_channels_covers_hidden_stream_and_deduplicates_tied_weights(adapter):
    model = make_model(tie_word_embeddings=True)
    zero_parameters(model)
    layer = model.model.layers[0]

    with torch.no_grad():
        model.model.embed_tokens.weight[:, 1] = 1.0
        model.model.norm.weight[0] = 1.0
        layer.input_layernorm.weight[0] = 1.0
        layer.post_attention_layernorm.weight[0] = 1.0

        layer.self_attn.q_proj.weight[0, 0] = 2.0
        layer.self_attn.k_proj.weight[0, 0] = 1.0
        layer.self_attn.v_proj.weight[0, 0] = 1.0
        layer.self_attn.o_proj.weight[0, 0] = 1.0

        layer.mlp.gate_proj.weight[0, 0] = 1.0
        layer.mlp.up_proj.weight[0, 0] = 1.0
        layer.mlp.down_proj.weight[0, 0] = 1.0

    estimator = WeightMagnitudeElementEstimator(model, device="cpu", model_adapter=adapter)
    scores = estimator.estimate_embedding_channels(agg="l1")

    torch.testing.assert_close(
        scores["embedding_channels"],
        torch.tensor([11.0, 5.0, 0.0, 0.0]),
    )


def test_prune_attention_group_accepts_group_importance_from_estimator(adapter):
    model = make_model()
    fill_attention_group_markers(model)

    estimator = WeightMagnitudeElementEstimator(model, device="cpu", model_adapter=adapter)
    group_importance = estimator.estimate_attention_groups(agg="l1")

    pruner = ElementPruner(model, device="cpu", model_adapter=adapter)
    pruned_model = pruner.prune_attention_group(
        group_importance=group_importance,
        target_group=1,
    )

    assert pruned_model.config.num_attention_heads == 2
    assert pruned_model.config.num_key_value_heads == 1
    assert_group_one_kept(model, pruned_model)


def test_prune_attention_group_legacy_head_importance_warns_and_still_prunes(adapter):
    model = make_model()
    fill_attention_group_markers(model)

    pruner = ElementPruner(model, device="cpu", model_adapter=adapter)
    head_importance = {0: torch.tensor([1.0, 1.0, 10.0, 10.0])}

    with pytest.warns(UserWarning, match="head_importance"):
        pruned_model = pruner.prune_attention_group(
            head_importance=head_importance,
            target_group=1,
        )

    assert pruned_model.config.num_attention_heads == 2
    assert pruned_model.config.num_key_value_heads == 1
    assert_group_one_kept(model, pruned_model)


def test_prune_attention_group_requires_exactly_one_score_input(adapter):
    model = make_model()
    pruner = ElementPruner(model, device="cpu", model_adapter=adapter)
    head_importance = {0: torch.tensor([1.0, 1.0, 10.0, 10.0])}
    group_importance = {0: torch.tensor([1.0, 10.0])}

    with pytest.raises(ValueError, match="Exactly one of group_importance or head_importance"):
        pruner.prune_attention_group(target_group=1)

    with pytest.raises(ValueError, match="Exactly one of group_importance or head_importance"):
        pruner.prune_attention_group(
            head_importance=head_importance,
            group_importance=group_importance,
            target_group=1,
        )
