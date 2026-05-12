<!-- last_updated: 2026-05-12 -->

# Architecture

Authoritative architectural reference: [docs/architecture.md](../architecture.md). This file is a status-focused summary mapping directories to responsibilities.

## Top-level Layout

| Path | Responsibility |
|------|----------------|
| [src/carve_lm/](../../src/carve_lm/) | Shipped library code. Split into `llm` and `vlm` namespaces. |
| [tests/](../../tests/) | Unit coverage (`tests/llm/unit/`, `tests/vlm/unit/`). |
| [examples/](../../examples/) | Minimal importable usage scripts. |
| [scripts/](../../scripts/) | Operational entrypoints (recovery / fine-tuning). |
| [docs/](../../docs/) | Architecture, migration, status reports. |
| [Notebook/](../../Notebook/) | Interactive demos. [TBD — exact contents not documented] |
| [.github/workflows/](../../.github/workflows/) | CI pipeline (`ci.yml`). |

## `src/carve_lm/` Namespace Map

```text
carve_lm/
├── llm/
│   ├── adapters/        # BaseModelAdapter + Llama/Qwen2/Qwen3/Mistral/Generic
│   ├── core/            # ESTIMATOR_REGISTRY, PRUNER_REGISTRY, identity layers
│   ├── estimators/      # activation.*, magnitude.*, similarity.*, perplexity.*, random.*, taylor.*
│   ├── pruners/         # WidthPruner, WidthGroupPruner, WidthChannelPruner, ComponentPruner, DepthBlockPruner, DepthLayerPruner
│   ├── distillation/    # LogitsDistiller, HybridDistiller, HybridOTDistiller, TeacherCorrection
│   └── evaluation/      # LLMMeasurer
└── vlm/
    ├── components/
    │   ├── language/    # Active decoder-side stack (Qwen2.5-VL, Qwen3-VL)
    │   ├── vision/      # Estimators only; pruners reserved (no v1 release)
    │   └── merger/      # Estimators only; pruners reserved (no v1 release)
    ├── distillation/    # Multimodal recovery helpers
    └── evaluation/      # VLMMeasurer
```

## Tri-level Framework

| Level | Pruners | Estimators |
|-------|---------|------------|
| Element (L1) | `WidthGroupPruner`, `WidthChannelPruner` | `activation.*`, `magnitude.*`, `taylor.*`, `random.*` |
| Layer (L2) | `ComponentPruner` | `similarity.layer` |
| Block (L3) | `DepthBlockPruner`, `DepthLayerPruner` | `similarity.block`, `perplexity.block` |

## Key Patterns

- **Adapter contract.** `BaseModelAdapter` abstracts decoder layers, attention / MLP projections, embeddings, LM head, norms, and mutators. Pruners never read model-family-specific attribute paths directly.
- **Registries.** `ESTIMATOR_REGISTRY`, `PRUNER_REGISTRY`, `PRUNING_STRATEGY_REGISTRY` per domain (`llm.core`, `vlm.components.language.core`, plus placeholders for `vision` and `merger`). Factories `create_estimator()` / `create_pruner()` route by method-first / effect-first keys.
- **GQA-aware grouping.** One attention group = one KV group + attached query heads + matching `o_proj` slice. MLP groups couple `gate_proj` row + `up_proj` row + `down_proj` column.
- **Manifest persistence.** `save_pruned()` writes `llm_pruner_manifest.json` (or `vlm_pruner_manifest.json`). `load_pruned()` rebuilds the dense base then replays the structural rewrite. Manifest v2 records pruner names, adapter metadata, and config payloads.

## Data / Request Flow

Canonical pruning pipeline (LLM and VLM language component):

```
discover()  →  estimate(dataloader)  →  select(scores)  →  apply(plan)  →  save_pruned(path)
                                                                            load_pruned(path)
```

Recovery flow lives in `*.distillation` (`HybridDistiller`, `LogitsDistiller`, `TeacherCorrection`) and operational scripts under `scripts/recovery/`.

## Notable Design Decisions

- Hard namespace split: no compatibility shims for legacy top-level packages (`estimator`, `pruner`, `distiller`, `measurer`, `carve_lm_core`, `llm_pruner_paper`). Consumers import directly from `carve_lm.llm.*` or `carve_lm.vlm.*`. See [docs/migration.md](../migration.md).
- Method-first taxonomy for estimators (e.g. `magnitude.element`), effect-first taxonomy for pruners (e.g. `width.group`).
- Reserved-namespace pattern for `vision` / `merger`: registries exist so future stacks can register without API churn.

## Infrastructure / Deployment

- Distribution: Python package via `setuptools` (`pyproject.toml`, build backend `setuptools.build_meta`).
- No Dockerfile or deployment manifest in repo — library only.
- CI: GitHub Actions matrix across Python 3.10 / 3.11 / 3.12. See [maintenance.md](maintenance.md).

Cross-references: [overview.md](overview.md) for quick start, [issues.md](issues.md) for areas still under development.
