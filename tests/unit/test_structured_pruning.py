from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from tests.fixtures.synthetic_models import SyntheticCausalLM, SyntheticConfig
from torch.utils.data import DataLoader

from soict_llm_pruner.estimators import ActivationElementEstimator
from soict_llm_pruner.estimators._shared import _BaseBlockPerplexityEstimator
from soict_llm_pruner.pruners.structured import (
    BlockWiseConfig,
    ChannelWiseConfig,
    DiscoveryContext,
    ImportanceConfig,
    LayerWiseConfig,
    PruningGroup,
    SliceSpec,
    StructuredBlockPruner,
    StructuredChannelPruner,
    StructuredLayerPruner,
    discover_blockwise,
    discover_channelwise,
    estimate_importance,
)
from soict_llm_pruner.pruners.structured.spec import object_path


def make_model(**overrides) -> SyntheticCausalLM:
    config = SyntheticConfig(**overrides)
    return SyntheticCausalLM(config)


def make_dataloader(*samples: dict) -> DataLoader:
    return DataLoader(list(samples), batch_size=1)


def zero_parameters(model: nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()


def fill_linear_markers(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.copy_(
            torch.arange(linear.weight.numel(), dtype=linear.weight.dtype).reshape_as(linear.weight)
        )


def make_score_map(context: DiscoveryContext, keep_ids: set[str], high: float = 10.0) -> dict[str, float]:
    return {
        group.group_id: (high if group.group_id in keep_ids else 0.0)
        for group in context.groups
    }


def test_blockwise_discovery_mha_attention_groups_map_one_head_each():
    model = make_model(num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4, head_dim=2, hidden_size=8)
    context = discover_blockwise(model, StructuredBlockPruner(model, BlockWiseConfig(pruning_ratio=0.5)).spec)

    attention_groups = [group for group in context.groups if group.family == "attention"]
    assert len(attention_groups) == 4
    assert attention_groups[1].metadata["query_head_indices"] == (1,)
    assert attention_groups[1].metadata["query_row_indices"] == (2, 3)
    assert attention_groups[1].metadata["kv_row_indices"] == (2, 3)
    assert attention_groups[1].dependent_slices[-1].axis == 1
    assert attention_groups[1].dependent_slices[-1].indices == (2, 3)


def test_blockwise_discovery_gqa_attention_groups_tie_query_heads_to_shared_kv():
    model = make_model(num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=2, hidden_size=8)
    context = discover_blockwise(model, StructuredBlockPruner(model, BlockWiseConfig(pruning_ratio=0.5)).spec)

    attention_groups = [group for group in context.groups if group.family == "attention"]
    assert len(attention_groups) == 2
    assert attention_groups[0].metadata["query_head_indices"] == (0, 1)
    assert attention_groups[0].metadata["query_row_indices"] == (0, 1, 2, 3)
    assert attention_groups[0].metadata["kv_row_indices"] == (0, 1)
    assert attention_groups[1].metadata["query_row_indices"] == (4, 5, 6, 7)
    assert attention_groups[1].metadata["kv_row_indices"] == (2, 3)


def test_channelwise_discovery_spans_hidden_stream_attention_mlp_and_lm_head():
    model = make_model(num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=2, hidden_size=8)
    context = discover_channelwise(model, StructuredChannelPruner(model, ChannelWiseConfig(pruning_ratio=0.5)).spec)

    assert len(context.groups) == 2
    channel_group = context.groups[0]
    assert channel_group.metadata["residual_indices"] == (0, 2, 4, 6)
    assert channel_group.metadata["kv_indices"] == (0, 2)
    roles = {slice_spec.role for slice_spec in channel_group.dependent_slices}
    assert {
        "embed_tokens_in",
        "final_norm",
        "lm_head_in",
        "input_norm",
        "post_attention_norm",
        "q_proj_out",
        "q_proj_in",
        "k_proj_out",
        "k_proj_in",
        "v_proj_out",
        "v_proj_in",
        "o_proj_out",
        "o_proj_in",
        "gate_proj_in",
        "up_proj_in",
        "down_proj_out",
    }.issubset(roles)


def test_magnitude_importance_supports_l1_and_l2():
    model = make_model(num_hidden_layers=1, intermediate_size=3)
    zero_parameters(model)
    with torch.no_grad():
        mlp = model.model.layers[0].mlp
        mlp.gate_proj.weight[0, 0] = 1.0
        mlp.up_proj.weight[0, 1] = -2.0
        mlp.down_proj.weight[2, 0] = 3.0

    spec = StructuredBlockPruner(model, BlockWiseConfig(pruning_ratio=0.5)).spec
    context = discover_blockwise(model, spec)
    l1_scores = estimate_importance(model, context, ImportanceConfig(kind="l1"), None, device="cpu")
    l2_scores = estimate_importance(model, context, ImportanceConfig(kind="l2"), None, device="cpu")

    assert l1_scores["mlp.layer0.neuron0"] == 6.0
    assert l2_scores["mlp.layer0.neuron0"] == 6.0
    assert l1_scores["mlp.layer0.neuron1"] == 0.0


@dataclass
class TaylorToyConfig:
    vocab_size: int = 3
    hidden_size: int = 2
    num_hidden_layers: int = 1
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    intermediate_size: int = 2
    head_dim: int = 2

    @classmethod
    def from_dict(cls, payload: dict) -> "TaylorToyConfig":
        return cls(**payload)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class TaylorToyLM(nn.Module):
    def __init__(self, config: TaylorToyConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_states = self.embed(input_ids)
        logits = self.linear(hidden_states)
        return {"logits": logits}


def make_taylor_context(model: TaylorToyLM) -> DiscoveryContext:
    group = PruningGroup(
        group_id="linear.row0",
        family="toy",
        layer_idx=None,
        local_idx=0,
        members=("linear",),
        dependent_slices=(
            SliceSpec(
                module_path="linear",
                param_name="weight",
                axis=0,
                indices=(0,),
                role="linear_out",
            ),
        ),
    )
    return DiscoveryContext(
        mode="block",
        family_key="toy",
        model_class_path=object_path(model.__class__),
        config_class_path=object_path(model.config.__class__),
        base_config=model.config.to_dict(),
        groups=(group,),
        layer_metadata=tuple(),
        hidden_size=model.config.hidden_size,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=model.config.head_dim,
        metadata={},
    )


def manual_taylor_score(model: TaylorToyLM, dataloader: DataLoader, variant: str) -> float:
    grad_sum = None
    grad_sq_sum = None
    for batch in dataloader:
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        logits = outputs["logits"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        loss.backward()
        grad = model.linear.weight.grad.detach().clone()
        if grad_sum is None:
            grad_sum = grad
            grad_sq_sum = grad.pow(2)
        else:
            grad_sum.add_(grad)
            grad_sq_sum.add_(grad.pow(2))

    weight = model.linear.weight.detach()[0:1, :]
    first = grad_sum[0:1, :]
    second = grad_sq_sum[0:1, :]
    if variant == "param_first":
        return float((weight * first).abs().sum().item())
    if variant == "param_second":
        return float((weight * second * weight).abs().sum().item())
    if variant == "param_mix":
        return float(((weight * first) - 0.5 * (weight * second * weight)).abs().sum().item())
    if variant == "vectorize":
        return float(torch.abs((weight * first).sum()).item())
    raise AssertionError("Unknown variant {}".format(variant))


@torch.no_grad()
def initialize_taylor_toy(model: TaylorToyLM) -> None:
    model.embed.weight.copy_(torch.tensor([[1.0, 0.5], [0.25, -0.5], [0.75, 1.0]]))
    model.linear.weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 0.25], [-0.75, 1.5]]))


def test_taylor_importance_matches_manual_variants():
    dataloader = make_dataloader(
        {
            "input_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "labels": torch.tensor([0, 1, 2], dtype=torch.long),
        }
    )
    for variant in ("param_first", "param_second", "param_mix", "vectorize"):
        model = TaylorToyLM(TaylorToyConfig())
        initialize_taylor_toy(model)
        context = make_taylor_context(model)
        scores = estimate_importance(
            model,
            context,
            ImportanceConfig(kind="taylor", taylor_variant=variant),
            dataloader,
            device="cpu",
        )
        expected = manual_taylor_score(model, dataloader, variant)
        torch.testing.assert_close(torch.tensor(scores["linear.row0"]), torch.tensor(expected))


def test_block_perplexity_uses_token_weighted_average_and_masks_padding():
    class TokenLossToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_by_first_token = {1: 1.0, 2: 4.0}

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            expected_labels = input_ids.clone()
            if attention_mask is not None:
                expected_labels = expected_labels.masked_fill(attention_mask == 0, -100)
            assert torch.equal(labels, expected_labels)
            return SimpleNamespace(
                loss=torch.tensor(
                    self.loss_by_first_token[int(input_ids[0, 0].item())],
                    dtype=torch.float32,
                    device=input_ids.device,
                )
            )

    dataloader = DataLoader(
        [
            {
                "input_ids": torch.tensor([1, 10, 11], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 0], dtype=torch.long),
                "labels": torch.tensor([1, 10, 11], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([2, 20, 21, 22, 23], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([2, 20, 21, 22, 23], dtype=torch.long),
            },
        ],
        batch_size=1,
    )

    estimator = object.__new__(_BaseBlockPerplexityEstimator)
    estimator.device = "cpu"

    perplexity = estimator._calculate_perplexity(TokenLossToyModel(), dataloader, n_samples=2)
    expected_loss = (1.0 * 1 + 4.0 * 4) / 5.0
    torch.testing.assert_close(torch.tensor(perplexity), torch.exp(torch.tensor(expected_loss)))


def test_blockwise_execution_prunes_mha_attention_groups():
    model = make_model(num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4, head_dim=2, hidden_size=8)
    attention = model.model.layers[0].self_attn
    fill_linear_markers(attention.q_proj)
    fill_linear_markers(attention.k_proj)
    fill_linear_markers(attention.v_proj)
    fill_linear_markers(attention.o_proj)

    pruner = StructuredBlockPruner(
        model,
        BlockWiseConfig(
            pruning_ratio=0.5,
            attention_layers=(0,),
            mlp_layers=(),
            importance=ImportanceConfig(kind="random"),
        ),
    )
    context = pruner.discover()
    scores = make_score_map(context, {"attention.layer0.group1", "attention.layer0.group3"})
    plan = pruner.select(scores)
    result = pruner.apply(plan)
    pruned_attention = result.model.model.layers[0].self_attn

    expected_indices = [2, 3, 6, 7]
    torch.testing.assert_close(pruned_attention.q_proj.weight, attention.q_proj.weight[expected_indices, :])
    torch.testing.assert_close(pruned_attention.k_proj.weight, attention.k_proj.weight[expected_indices, :])
    torch.testing.assert_close(pruned_attention.v_proj.weight, attention.v_proj.weight[expected_indices, :])
    torch.testing.assert_close(pruned_attention.o_proj.weight, attention.o_proj.weight[:, expected_indices])
    assert pruned_attention.num_heads == 2
    assert pruned_attention.num_key_value_heads == 2


def test_blockwise_execution_prunes_gqa_attention_groups():
    model = make_model(num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=2, hidden_size=8)
    attention = model.model.layers[0].self_attn
    fill_linear_markers(attention.q_proj)
    fill_linear_markers(attention.k_proj)
    fill_linear_markers(attention.v_proj)
    fill_linear_markers(attention.o_proj)

    pruner = StructuredBlockPruner(
        model,
        BlockWiseConfig(
            pruning_ratio=0.5,
            attention_layers=(0,),
            mlp_layers=(),
            importance=ImportanceConfig(kind="random"),
        ),
    )
    context = pruner.discover()
    scores = make_score_map(context, {"attention.layer0.group1"})
    plan = pruner.select(scores)
    result = pruner.apply(plan)
    pruned_attention = result.model.model.layers[0].self_attn

    torch.testing.assert_close(pruned_attention.q_proj.weight, attention.q_proj.weight[4:8, :])
    torch.testing.assert_close(pruned_attention.k_proj.weight, attention.k_proj.weight[2:4, :])
    torch.testing.assert_close(pruned_attention.v_proj.weight, attention.v_proj.weight[2:4, :])
    torch.testing.assert_close(pruned_attention.o_proj.weight, attention.o_proj.weight[:, 4:8])
    assert pruned_attention.num_heads == 2
    assert pruned_attention.num_key_value_heads == 1


def test_blockwise_execution_prunes_mlp_neurons():
    model = make_model(num_hidden_layers=1, intermediate_size=4)
    mlp = model.model.layers[0].mlp
    fill_linear_markers(mlp.gate_proj)
    fill_linear_markers(mlp.up_proj)
    fill_linear_markers(mlp.down_proj)

    pruner = StructuredBlockPruner(
        model,
        BlockWiseConfig(
            pruning_ratio=0.5,
            attention_layers=(),
            mlp_layers=(0,),
            importance=ImportanceConfig(kind="random"),
        ),
    )
    context = pruner.discover()
    scores = make_score_map(context, {"mlp.layer0.neuron1", "mlp.layer0.neuron3"})
    plan = pruner.select(scores)
    result = pruner.apply(plan)
    pruned_mlp = result.model.model.layers[0].mlp

    torch.testing.assert_close(pruned_mlp.gate_proj.weight, mlp.gate_proj.weight[[1, 3], :])
    torch.testing.assert_close(pruned_mlp.up_proj.weight, mlp.up_proj.weight[[1, 3], :])
    torch.testing.assert_close(pruned_mlp.down_proj.weight, mlp.down_proj.weight[:, [1, 3]])


def test_blockwise_execution_supports_mixed_attention_and_mlp_pruning():
    model = make_model(
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        hidden_size=8,
        intermediate_size=4,
    )
    layer = model.model.layers[0]
    fill_linear_markers(layer.self_attn.q_proj)
    fill_linear_markers(layer.self_attn.k_proj)
    fill_linear_markers(layer.self_attn.v_proj)
    fill_linear_markers(layer.self_attn.o_proj)
    fill_linear_markers(layer.mlp.gate_proj)
    fill_linear_markers(layer.mlp.up_proj)
    fill_linear_markers(layer.mlp.down_proj)

    pruner = StructuredBlockPruner(
        model,
        BlockWiseConfig(
            pruning_ratio=0.5,
            attention_layers=(0,),
            mlp_layers=(0,),
            importance=ImportanceConfig(kind="random"),
        ),
    )
    context = pruner.discover()
    scores = make_score_map(
        context,
        {"attention.layer0.group1", "mlp.layer0.neuron1", "mlp.layer0.neuron3"},
    )
    result = pruner.apply(pruner.select(scores))
    pruned_layer = result.model.model.layers[0]

    assert pruned_layer.self_attn.q_proj.weight.shape == (4, 8)
    assert pruned_layer.self_attn.k_proj.weight.shape == (2, 8)
    assert pruned_layer.mlp.gate_proj.weight.shape == (2, 8)
    assert pruned_layer.mlp.down_proj.weight.shape == (8, 2)


def test_channelwise_execution_rewrites_hidden_stream_and_keeps_model_runnable():
    model = make_model(num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=2, hidden_size=8)
    fill_linear_markers(model.model.layers[0].self_attn.q_proj)
    fill_linear_markers(model.model.layers[0].self_attn.k_proj)
    fill_linear_markers(model.model.layers[0].self_attn.v_proj)
    fill_linear_markers(model.model.layers[0].self_attn.o_proj)
    fill_linear_markers(model.model.layers[0].mlp.gate_proj)
    fill_linear_markers(model.model.layers[0].mlp.up_proj)
    fill_linear_markers(model.model.layers[0].mlp.down_proj)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(
            torch.arange(model.model.embed_tokens.weight.numel(), dtype=torch.float32).reshape_as(
                model.model.embed_tokens.weight
            )
        )

    pruner = StructuredChannelPruner(
        model,
        ChannelWiseConfig(pruning_ratio=0.5, importance=ImportanceConfig(kind="random")),
    )
    context = pruner.discover()
    scores = make_score_map(context, {"channel.bundle1"})
    plan = pruner.select(scores)
    result = pruner.apply(plan)
    pruned_model = result.model
    pruned_attention = pruned_model.model.layers[0].self_attn

    expected_residual_indices = [1, 3, 5, 7]
    expected_kv_indices = [1, 3]
    assert pruned_model.config.hidden_size == 4
    assert pruned_model.config.head_dim == 1
    assert pruned_attention.q_proj.weight.shape == (4, 4)
    assert pruned_attention.k_proj.weight.shape == (2, 4)
    assert pruned_attention.v_proj.weight.shape == (2, 4)
    assert pruned_attention.o_proj.weight.shape == (4, 4)
    torch.testing.assert_close(
        pruned_model.model.embed_tokens.weight,
        model.model.embed_tokens.weight[:, expected_residual_indices],
    )
    assert plan.metadata["selected_residual_indices"] == expected_residual_indices
    assert plan.metadata["selected_kv_indices"] == expected_kv_indices

    outputs = pruned_model(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert outputs["logits"].shape == (1, 3, model.config.vocab_size)


def test_layerwise_execution_keeps_prefix_layers_and_updates_config():
    model = make_model(num_hidden_layers=3)
    pruner = StructuredLayerPruner(model, LayerWiseConfig(target_num_layers=1))
    result = pruner.run()

    assert len(result.model.model.layers) == 1
    assert result.model.config.num_hidden_layers == 1
    outputs = result.model(input_ids=torch.tensor([[0, 1]], dtype=torch.long))
    assert outputs["logits"].shape == (1, 2, model.config.vocab_size)


def test_blockwise_persistence_roundtrip_recreates_logits_and_manifest(tmp_path: Path):
    model = make_model(num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4, head_dim=2, hidden_size=8)
    fill_linear_markers(model.model.layers[0].self_attn.q_proj)
    fill_linear_markers(model.model.layers[0].self_attn.k_proj)
    fill_linear_markers(model.model.layers[0].self_attn.v_proj)
    fill_linear_markers(model.model.layers[0].self_attn.o_proj)
    pruner = StructuredBlockPruner(
        model,
        BlockWiseConfig(
            pruning_ratio=0.5,
            attention_layers=(0,),
            mlp_layers=(),
            importance=ImportanceConfig(kind="random"),
        ),
    )
    context = pruner.discover()
    result = pruner.apply(
        pruner.select(
            make_score_map(
                context,
                {"attention.layer0.group1", "attention.layer0.group3"},
            )
        )
    )
    sample = {"input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long)}
    expected_logits = result.model(**sample)["logits"]
    save_dir = tmp_path / "block"
    pruner.save_pruned(save_dir, result)

    loaded = StructuredBlockPruner.load_pruned(save_dir, device="cpu")
    actual_logits = loaded.model(**sample)["logits"]
    torch.testing.assert_close(actual_logits, expected_logits)

    manifest = json.loads(
        (save_dir / "llm_pruner_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == 2
    assert manifest["canonical_pruner"] == "width.group"
    assert manifest["pruning_mode"] == "block"
    assert manifest["adapter_name"] == "decoder_layout"
    assert manifest["config_class"].endswith(":WidthGroupConfig")
    assert manifest["config_payload"]["estimator"]["name"] == "random.group"
    assert manifest["plan"]["metadata"]["selected_attention_groups_by_layer"] == {"0": [1, 3]}


def test_channelwise_persistence_roundtrip_recreates_logits_and_indices(tmp_path: Path):
    model = make_model(num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=2, hidden_size=8)
    pruner = StructuredChannelPruner(
        model,
        ChannelWiseConfig(
            pruning_ratio=0.5,
            importance=ImportanceConfig(kind="random"),
        ),
    )
    context = pruner.discover()
    result = pruner.apply(pruner.select(make_score_map(context, {"channel.bundle1"})))
    sample = {"input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long)}
    expected_logits = result.model(**sample)["logits"]
    save_dir = tmp_path / "channel"
    pruner.save_pruned(save_dir, result)

    loaded = StructuredChannelPruner.load_pruned(save_dir, device="cpu")
    actual_logits = loaded.model(**sample)["logits"]
    torch.testing.assert_close(actual_logits, expected_logits)

    manifest = json.loads(
        (save_dir / "llm_pruner_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == 2
    assert manifest["canonical_pruner"] == "width.channel"
    assert manifest["pruning_mode"] == "channel"
    assert manifest["config_class"].endswith(":WidthChannelConfig")
    assert manifest["config_payload"]["estimator"]["name"] == "random.group"
    assert manifest["plan"]["metadata"]["selected_residual_indices"] == [1, 3, 5, 7]


def test_layerwise_persistence_roundtrip_recreates_logits(tmp_path: Path):
    model = make_model(num_hidden_layers=3)
    pruner = StructuredLayerPruner(model, LayerWiseConfig(target_num_layers=1))
    result = pruner.run()
    sample = {"input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long)}
    expected_logits = result.model(**sample)["logits"]
    save_dir = tmp_path / "layer"
    pruner.save_pruned(save_dir, result)

    loaded = StructuredLayerPruner.load_pruned(save_dir, device="cpu")
    actual_logits = loaded.model(**sample)["logits"]
    torch.testing.assert_close(actual_logits, expected_logits)

    manifest = json.loads(
        (save_dir / "llm_pruner_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == 2
    assert manifest["canonical_pruner"] == "depth.layer"
    assert manifest["pruning_mode"] == "layer"
    assert manifest["config_class"].endswith(":DepthLayerConfig")


def test_unified_package_does_not_import_legacy_namespaces():
    package_dir = Path(__file__).resolve().parents[2] / "src" / "soict_llm_pruner"
    forbidden_tokens = (
        "soict_llm_pruner_core",
        "llm_pruner_paper",
        "from estimator",
        "from pruner",
    )
    for path in package_dir.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for forbidden in forbidden_tokens:
            assert forbidden not in content

    estimator = ActivationElementEstimator
    assert estimator is not None
