# Architecture

## Unified Namespace

All shipped code now lives under `src/carve_lm/`.

```text
carve_lm/
├── adapters/
├── core/
├── estimators/
├── pruners/
│   └── structured/
├── distillation/
└── evaluation/
```

There are no compatibility shims for the old top-level packages. Consumers are expected to import directly from `carve_lm.*`.

## Core Concepts

### Adapters

`carve_lm.adapters` defines the model-facing contract used by the classic estimator/pruner stack. A `BaseModelAdapter` exposes:

- decoder layers
- attention and MLP projections
- embeddings and LM head
- norms
- mutators for replacing projections or whole submodules

This keeps pruning logic independent from model-family-specific attribute paths.

### Registries

`carve_lm.core.registry` owns three extension points:

- `ESTIMATOR_REGISTRY`
- `PRUNER_REGISTRY`
- `PRUNING_STRATEGY_REGISTRY`

Classic estimators and pruners register themselves here so external code can use `create_estimator()` and `create_pruner()` without hardcoding classes.

### Estimators

`carve_lm.estimators` now uses a method-first taxonomy:

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

`carve_lm.pruners` now uses an effect-first taxonomy:

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

Structured persistence is manifest-based. `save_pruned()` stores weights plus `llm_pruner_manifest.json`, and `load_pruned()` reconstructs the dense base architecture before replaying the structural rewrite. Manifest v2 records canonical pruner names, adapter metadata, and config payloads.

### Distillation And Recovery

`carve_lm.distillation` contains reusable training helpers:

- `HybridDistiller`
- `LogitsDistiller`
- `TeacherCorrection`

Operational training scripts live in `scripts/recovery/`.

### Evaluation

`carve_lm.evaluation` currently exposes `LLMMeasurer` for latency and throughput measurement.

## Repository Layout

- `examples/`: minimal importable usage examples
- `scripts/`: operational entrypoints
- `notebooks/`: interactive demos
- `tests/unit/`: fast behavioral coverage
- `tests/integration/`: cross-module and persistence coverage
