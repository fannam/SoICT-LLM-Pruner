<!-- last_updated: 2026-05-12 -->

# Overview

## Purpose

CarveLM (`carve-lm`) is a tri-level structured pruning library for LLM and VLM transformer models. It targets three independent pruning granularities:

| Level | Target |
|-------|--------|
| Element (L1) | Attention heads, GQA groups, MLP neurons, embedding channels |
| Layer (L2) | Attention or MLP sublayer within a decoder block |
| Block (L3) | Contiguous decoder blocks |

It also ships post-pruning recovery (distillation) and latency/throughput evaluation helpers.

## Target Users

- ML researchers studying structured pruning and compression of HuggingFace decoder models.
- Practitioners running pruning + distillation recovery on Llama / Qwen / Mistral / Qwen2.5-VL / Qwen3-VL.
- Library authors integrating pruning into larger training or serving stacks via the adapter / registry API.

## Development Stage

Alpha (`Development Status :: 3 - Alpha` in `pyproject.toml`). API is breaking between releases. See [docs/migration.md](../migration.md) for the canonical-namespace migration table.

## Tech Stack

Core runtime dependencies (`pyproject.toml`):

| Package | Version |
|---------|---------|
| `torch` | `>=2.0.0` |
| `transformers` | `>=4.38.2` |
| `tqdm` | `>=4.65.0` |

Optional extras:

| Extra | Adds |
|-------|------|
| `train` | `accelerate>=0.30.0`, `datasets>=2.14.0`, `matplotlib>=3.7.0`, `wandb>=0.16.0` |
| `dev` | `build>=1.2.0`, `pytest>=8.0.0`, `ruff>=0.5.0`, `twine>=5.0.0` |
| `notebooks` | `jupyter>=1.0.0` |

Supported Python versions: 3.10, 3.11, 3.12.

## Quick Start

Install (editable):

```bash
git clone https://github.com/fannam/CarveLM.git
cd CarveLM
pip install -e .
```

With optional extras:

```bash
pip install -e ".[train]"
pip install -e ".[dev]"
pip install -e ".[notebooks]"
```

Run lint and tests:

```bash
ruff check .
pytest
```

Minimal usage:

```python
from transformers import AutoModelForCausalLM
from carve_lm.llm.estimators import create_estimator
from carve_lm.llm.pruners import create_pruner

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
estimator = create_estimator("magnitude.element", model, device="cpu")
pruner = create_pruner("width", model, device="cpu")

head_scores = estimator.estimate_attention_heads(agg="l1")
pruned_model = pruner.prune_attention_query(
    head_importance=head_scores,
    target_num_attention_heads=model.config.num_attention_heads // 2,
)
```

Examples:

- [examples/pruning/basic_usage.py](../../examples/pruning/basic_usage.py)
- [examples/structured/llm_pruner_usage.py](../../examples/structured/llm_pruner_usage.py)
- [examples/evaluation/measure_latency.py](../../examples/evaluation/measure_latency.py)

See [architecture.md](architecture.md) for the namespace map.
