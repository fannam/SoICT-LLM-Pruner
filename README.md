# SOICT-LLM-Pruner

Unified pruning library for decoder-only LLMs under one namespace: `soict_llm_pruner`.

![Framework Overview](docs/assets/tri-level-framework.png)

## Install

```bash
git clone https://github.com/fannam/SoICT-LLM-Pruner.git
cd SoICT-LLM-Pruner
pip install -e .
```

Optional extras:

```bash
pip install -e ".[train]"
pip install -e ".[dev]"
pip install -e ".[notebooks]"
```

## Package Map

- `soict_llm_pruner.adapters`: model adapter contracts and registrations
- `soict_llm_pruner.core`: registries, identity layers, scoring helpers
- `soict_llm_pruner.estimators`: method-first estimators such as `activation.*`, `magnitude.*`, `similarity.*`, `perplexity.*`, `taylor.*`
- `soict_llm_pruner.pruners`: canonical width/component/depth pruning APIs plus config types
- `soict_llm_pruner.pruners.structured`: deprecated compatibility facade for legacy structured imports
- `soict_llm_pruner.distillation`: recovery and distillation helpers
- `soict_llm_pruner.evaluation`: latency and throughput measurement

Additional documentation:

- [Architecture](docs/architecture.md)
- [Migration Guide](docs/migration.md)

## Quick Start

Classic estimator + pruner flow:

```python
from transformers import AutoModelForCausalLM

from soict_llm_pruner.estimators import create_estimator
from soict_llm_pruner.pruners import create_pruner

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

estimator = create_estimator("magnitude.element", model, device="cpu")
pruner = create_pruner("width", model, device="cpu")

head_scores = estimator.estimate_attention_heads(agg="l1")
pruned_model = pruner.prune_attention_query(
    head_importance=head_scores,
    target_num_attention_heads=model.config.num_attention_heads // 2,
)
```

Structured pruning flow:

```python
from soict_llm_pruner.pruners import (
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

## Supported Models

- Llama-like decoder-only architectures via adapters
- Structured v1 rules for `Llama3`, `Qwen2`, and `Mistral`

## Canonical Pruning API

```python
from soict_llm_pruner.pruners import (
    DepthLayerConfig,
    EstimatorSpec,
    PruningResult,
    WidthChannelConfig,
    WidthChannelPruner,
    WidthGroupConfig,
    WidthGroupPruner,
)
```

Legacy `soict_llm_pruner.pruners.structured` imports still work for one compatibility release and emit `DeprecationWarning`.

Structured block-wise attention groups are GQA-aware:

- MHA: one head is one atomic attention group
- GQA/MQA: one atomic group is one KV group plus all attached query heads plus the matching `o_proj` slice

Structured MLP groups are coupled:

- one `gate_proj` row
- one `up_proj` row
- one `down_proj` column

## Examples

- [examples/pruning/basic_usage.py](examples/pruning/basic_usage.py)
- [examples/structured/llm_pruner_usage.py](examples/structured/llm_pruner_usage.py)
- [examples/evaluation/measure_latency.py](examples/evaluation/measure_latency.py)

## Development

```bash
ruff check .
pytest
```
