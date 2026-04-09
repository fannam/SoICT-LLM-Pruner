# CarveLM

Tri-level structured pruning library for decoder-only large language models, published as `carve-lm`.

![Framework Overview](docs/assets/tri-level-framework.png)

## Install

```bash
git clone https://github.com/fannam/CarveLM.git
cd CarveLM
pip install -e .
```

Optional extras:

```bash
pip install -e ".[train]"      # accelerate, datasets, wandb — recovery workflows
pip install -e ".[dev]"        # pytest, ruff, build, twine
pip install -e ".[notebooks]"  # jupyter
```

## Package Map

| Module | Description |
|--------|-------------|
| `carve_lm.adapters` | Model adapter contracts and registry. Concrete adapters: `LlamaModelAdapter`, `Qwen2ModelAdapter`, `MistralModelAdapter`, `GenericDecoderModelAdapter`. |
| `carve_lm.core` | Global registries (`ESTIMATOR_REGISTRY`, `PRUNER_REGISTRY`), identity pass-through layers, and scoring helpers. |
| `carve_lm.estimators` | **Tri-level** importance estimators: `element.*` (activation, magnitude, Taylor, random), `layer.*` (similarity), `block.*` (similarity, perplexity). Factory: `create_estimator`. |
| `carve_lm.pruners` | **Tri-level** structured pruners: `element/` → width/channel pruning, `layer/` → component pruning, `block/` → depth pruning. Config types and `create_pruner` factory. |
| `carve_lm.pruners.structured` | Deprecated compatibility facade — `StructuredBlockPruner`, `StructuredChannelPruner`, `StructuredLayerPruner`. Emits `DeprecationWarning`. |
| `carve_lm.distillation` | Recovery helpers: `LogitsDistiller`, `HybridDistiller`, `HybridOTDistiller`, `TeacherCorrection` (requires `[train]`). |
| `carve_lm.evaluation` | Latency and throughput measurement via `LLMMeasurer`. |

Additional documentation:

- [Architecture](docs/architecture.md)
- [Migration Guide](docs/migration.md)

## Tri-level Framework

Pruning operates at three independent levels:

| Level | Target | Pruners | Estimators |
|-------|--------|---------|------------|
| **Element** (L1) | Attention heads, GQA groups, MLP neurons, embedding channels | `WidthGroupPruner`, `WidthChannelPruner` | `activation.*`, `magnitude.*`, `taylor.*`, `random.*` |
| **Layer** (L2) | Attention or MLP sublayer within a decoder block | `ComponentPruner` | `similarity.layer` |
| **Block** (L3) | Contiguous decoder blocks | `DepthBlockPruner`, `DepthLayerPruner` | `similarity.block`, `perplexity.block` |

## Supported Models

Natively registered adapters:

- **Llama** (`LlamaForCausalLM`) — Llama 2 / 3 family
- **Qwen2** (`Qwen2ForCausalLM`) — Qwen 2 / 2.5 family
- **Mistral** (`MistralForCausalLM`) — Mistral family

Any model that follows the standard HuggingFace decoder layout (`model.model.layers[*].{self_attn, mlp, input_layernorm, post_attention_layernorm}`) is automatically picked up by `GenericDecoderModelAdapter`.

Custom adapters can be registered at runtime:

```python
from carve_lm.adapters import register_model_adapter, DecoderModelAdapter
from transformers import MyModelForCausalLM

class MyModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(name="my_model", model_cls=MyModelForCausalLM)

register_model_adapter(MyModelAdapter())
```

## Quick Start

Classic estimator + pruner flow:

```python
from transformers import AutoModelForCausalLM

from carve_lm.estimators import create_estimator
from carve_lm.pruners import create_pruner

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

estimator = create_estimator("magnitude.element", model, device="cpu")
pruner = create_pruner("width", model, device="cpu")

head_scores = estimator.estimate_attention_heads(agg="l1")
pruned_model = pruner.prune_attention_query(
    head_importance=head_scores,
    target_num_attention_heads=model.config.num_attention_heads // 2,
)
```

Structured pruning flow (GQA-aware):

```python
from carve_lm.pruners import (
    EstimatorSpec,
    WidthGroupConfig,
    WidthGroupPruner,
)

pruner = WidthGroupPruner(
    model,
    WidthGroupConfig(
        pruning_ratio=0.2,
        estimator=EstimatorSpec("magnitude.group", {"norm": "l1"}),
    ),
    device="cpu",
)

context = pruner.discover()
scores = pruner.estimate(dataloader=None)
plan = pruner.select(scores)
result = pruner.apply(plan)

pruner.save_pruned("artifacts/block", result)
reloaded = WidthGroupPruner.load_pruned("artifacts/block", device="cpu")
```

## Canonical Pruning API

```python
from carve_lm.pruners import (
    DepthLayerConfig,
    EstimatorSpec,
    PruningResult,
    WidthChannelConfig,
    WidthChannelPruner,
    WidthGroupConfig,
    WidthGroupPruner,
)
```

Structured block-wise attention groups are GQA-aware:

- **MHA**: one head is one atomic attention group
- **GQA/MQA**: one atomic group = one KV group + all attached query heads + the matching `o_proj` slice

Structured MLP groups are always coupled:

- one `gate_proj` row + one `up_proj` row + one `down_proj` column

## Examples

- [examples/pruning/basic_usage.py](examples/pruning/basic_usage.py)
- [examples/structured/llm_pruner_usage.py](examples/structured/llm_pruner_usage.py)
- [examples/evaluation/measure_latency.py](examples/evaluation/measure_latency.py)

## Recovery Scripts

Post-pruning fine-tuning and knowledge distillation scripts live under `scripts/recovery/`:

- `finetune_llama.py` — SFT fine-tuning for Llama models
- `finetune_qwen.py` — SFT fine-tuning for Qwen models
- `teacher_correction_accelerate.py` — Teacher-correction distillation with Accelerate

## Development

```bash
pip install -e ".[dev,train]"
ruff check .
pytest
```
