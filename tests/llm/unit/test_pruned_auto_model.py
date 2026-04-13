from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from carve_lm.llm.auto_model import PrunedAutoModelForCausalLM
from carve_lm.llm.core import AttentionPasser, FeedForwardPasser
from carve_lm.llm.pruners import ComponentPruner


def make_tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config)


def test_pruned_auto_model_round_trip_restores_component_pruned_architecture(tmp_path):
    torch.manual_seed(0)
    model = make_tiny_llama()
    pruned_model = ComponentPruner(model, device="cpu").prune(
        importance_scores={
            "attention": [0.0, 1.0],
            "mlp": [1.0, 0.0],
        },
        prune_counts={
            "attention": 1,
            "mlp": 1,
        },
    )

    input_ids = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    pruned_model.eval()
    with torch.no_grad():
        expected_logits = pruned_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    pruned_model.save_pretrained(tmp_path, safe_serialization=False)

    loaded_model = PrunedAutoModelForCausalLM.from_pretrained(tmp_path)
    loaded_model.eval()

    assert loaded_model.config.attention_layer_to_prune == [0]
    assert loaded_model.config.mlp_layer_to_prune == [1]
    assert isinstance(loaded_model.model.layers[0].self_attn, AttentionPasser)
    assert not isinstance(loaded_model.model.layers[1].self_attn, AttentionPasser)
    assert isinstance(loaded_model.model.layers[1].mlp, FeedForwardPasser)
    assert not isinstance(loaded_model.model.layers[0].mlp, FeedForwardPasser)

    with torch.no_grad():
        actual_logits = loaded_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    torch.testing.assert_close(actual_logits, expected_logits)


def test_pruned_auto_model_from_config_applies_identity_components():
    config = make_tiny_llama().config
    config.attention_layer_to_prune = [1]
    config.mlp_layer_to_prune = [0]

    model = PrunedAutoModelForCausalLM.from_config(config)

    assert isinstance(model.model.layers[1].self_attn, AttentionPasser)
    assert isinstance(model.model.layers[0].mlp, FeedForwardPasser)
