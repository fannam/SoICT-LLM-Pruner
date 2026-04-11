# Architecture

## Split Namespace

All shipped code now lives under `src/carve_lm/`.

```text
carve_lm/
├── llm/
│   ├── adapters/
│   ├── core/
│   ├── distillation/
│   ├── estimators/
│   ├── evaluation/
│   └── pruners/
│       └── structured/
├── vlm/
│   ├── adapters/
│   ├── core/
│   ├── distillation/
│   ├── estimators/
│   ├── evaluation/
│   └── pruners/
│       └── structured/
```

There are no compatibility shims for the old root packages. Consumers are expected to import directly from `carve_lm.llm.*` or `carve_lm.vlm.*`.

## Core Concepts

### Adapters

`carve_lm.llm.adapters` and `carve_lm.vlm.adapters` define the model-facing contracts used by the pruning stacks. A `BaseModelAdapter` exposes:

- decoder layers
- attention and MLP projections
- embeddings and LM head
- norms
- mutators for replacing projections or whole submodules

This keeps pruning logic independent from model-family-specific attribute paths.

### Registries

Each domain-local core registry (`carve_lm.llm.core.registry`, `carve_lm.vlm.core.registry`) owns three extension points:

- `ESTIMATOR_REGISTRY`
- `PRUNER_REGISTRY`
- `PRUNING_STRATEGY_REGISTRY`

Classic estimators and pruners register themselves here so external code can use `create_estimator()` and `create_pruner()` without hardcoding classes.

### Estimators

`carve_lm.llm.estimators` and `carve_lm.vlm.estimators` use a method-first taxonomy:

- `activation.element`
- `magnitude.element`
- `similarity.layer`
- `similarity.block`
- `perplexity.block`
- `random.group`
- `magnitude.group`
- `magnitude.channel`
- `taylor.group`

The activation estimator computes importance from calibration activations. For grouped attention it measures the grouped `o_proj` input directly, so one attention-group score corresponds to one KV group plus all attached query heads.

The weight-magnitude estimator is data free. Its group scores are structured sums over exactly the slices that survive pruning:

- query-head score: `q_proj` rows + matching `o_proj` columns
- attention-group score: grouped `q_proj` rows + `k_proj` rows + `v_proj` rows + matching `o_proj` columns
- MLP neuron score: `gate_proj` row + `up_proj` row + `down_proj` column
- embedding/channel score: hidden-stream-aligned slices summed across embeddings, norms, attention, MLP, and LM head

### Pruners

`carve_lm.llm.pruners` and `carve_lm.vlm.pruners` use an effect-first taxonomy:

- `width`
- `width.group`
- `width.channel`
- `component`
- `depth.block`
- `depth.layer`

Canonical public classes:

- `WidthPruner`
- `WidthGroupPruner`
- `WidthChannelPruner`
- `ComponentPruner`
- `DepthBlockPruner`
- `DepthLayerPruner`

Compatibility aliases such as `ElementPruner` and `StructuredBlockPruner` remain for one release and emit `DeprecationWarning`.

Shared flow:

1. `discover()`
2. `estimate()`
3. `select()`
4. `apply()`

Structured block-wise discovery creates only two group families:

- attention groups anchored on `q_proj`
- MLP groups anchored on `gate_proj`

Structured persistence is manifest-based. LLM pruners store `llm_pruner_manifest.json`; VLM pruners store `vlm_pruner_manifest.json`. `load_pruned()` reconstructs the dense base architecture before replaying the structural rewrite. Manifest v2 records canonical pruner names, adapter metadata, and config payloads.

### Distillation And Recovery

`carve_lm.llm.distillation` contains the LLM recovery helpers:

- `HybridDistiller`
- `LogitsDistiller`
- `TeacherCorrection`

`carve_lm.vlm.distillation` mirrors the same API, but preserves full multimodal processor batches and forwards every non-label key through teacher and student calls.

Operational training scripts live in `scripts/recovery/`.

### Evaluation

`carve_lm.llm.evaluation` exposes `LLMMeasurer` for text-generation latency and throughput measurement.

`carve_lm.vlm.evaluation` exposes `VLMMeasurer` for multimodal generation latency and throughput measurement from processor-style batches.

## Repository Layout

- `examples/`: minimal importable usage examples
- `scripts/`: operational entrypoints
- `notebooks/`: interactive demos
- `tests/llm/unit/`: fast LLM behavioral coverage
- `tests/vlm/unit/`: fast VLM behavioral coverage
- `tests/integration/`: cross-module and persistence coverage
