# Migration Guide

This refactor is intentionally breaking. Old top-level packages such as `estimator`, `pruner`, `distiller`, `measurer`, `carve_lm_core`, and `llm_pruner_paper` are removed.

## Import Mapping

| Old import | New import |
| --- | --- |
| `from estimator import ...` | `from carve_lm.llm.estimators import ...` |
| `from pruner import ...` | `from carve_lm.llm.pruners import ...` |
| `create_estimator("element.activation", ...)` | `create_estimator("activation.element", ...)` |
| `create_estimator("element.weight_magnitude", ...)` | `create_estimator("magnitude.element", ...)` |
| `create_estimator("layer.similarity", ...)` | `create_estimator("similarity.layer", ...)` |
| `create_estimator("block.similarity", ...)` | `create_estimator("similarity.block", ...)` |
| `create_estimator("block.perplexity", ...)` | `create_estimator("perplexity.block", ...)` |
| `create_pruner("element", ...)` | `create_pruner("width", ...)` |
| `create_pruner("layer", ...)` | `create_pruner("component", ...)` |
| `create_pruner("block", ...)` | `create_pruner("depth.block", ...)` |
| `from distiller.hybrid_distiller import HybridDistiller` | `from carve_lm.llm.distillation import HybridDistiller` |
| `from distiller.logits_distiller import LogitsDistiller` | `from carve_lm.llm.distillation import LogitsDistiller` |
| `from measurer.general_measure import LLMMeasurer` | `from carve_lm.llm.evaluation import LLMMeasurer` |
| `from carve_lm_core import BaseModelAdapter` | `from carve_lm.llm.adapters import BaseModelAdapter` |
| `from carve_lm_core import DecoderModelAdapter` | `from carve_lm.llm.adapters import DecoderModelAdapter` |
| `from carve_lm_core import ESTIMATOR_REGISTRY` | `from carve_lm.llm.core import ESTIMATOR_REGISTRY` |
| `from carve_lm_core import PRUNER_REGISTRY` | `from carve_lm.llm.core import PRUNER_REGISTRY` |
| `from carve_lm_core import PRUNING_STRATEGY_REGISTRY` | `from carve_lm.llm.core import PRUNING_STRATEGY_REGISTRY` |
| `from llm_pruner_paper import BlockWiseConfig` | `from carve_lm.llm.pruners.structured import BlockWiseConfig` |
| `from llm_pruner_paper import ChannelWiseConfig` | `from carve_lm.llm.pruners.structured import ChannelWiseConfig` |
| `from llm_pruner_paper import LayerWiseConfig` | `from carve_lm.llm.pruners.structured import LayerWiseConfig` |
| `from llm_pruner_paper import ImportanceConfig` | `from carve_lm.llm.pruners.structured import ImportanceConfig` |
| `from llm_pruner_paper import BlockWiseLLMPruner` | `from carve_lm.llm.pruners.structured import StructuredBlockPruner` |
| `from llm_pruner_paper import ChannelWiseLLMPruner` | `from carve_lm.llm.pruners.structured import StructuredChannelPruner` |
| `from llm_pruner_paper import LayerWiseLLMPruner` | `from carve_lm.llm.pruners.structured import StructuredLayerPruner` |

## Canonical API

Prefer the canonical namespaces for new code:

- `ActivationEstimator`
- `MagnitudeEstimator`
- `SimilarityEstimator`
- `PerplexityEstimator`
- `WidthPruner`
- `WidthGroupPruner`
- `WidthChannelPruner`
- `ComponentPruner`
- `DepthBlockPruner`
- `DepthLayerPruner`

## Behavioral Notes

- Backward-compatibility wrappers exist for one release and emit `DeprecationWarning`.
- Structured pruning compatibility imports still work, but the canonical LLM API now lives under `carve_lm.llm.pruners`.
- `save_pruned()` and `load_pruned()` remain the supported persistence path for structured outputs.
- Recovery and evaluation utilities are domain-local now: use `carve_lm.llm.*` for text-only workflows and `carve_lm.vlm.*` for multimodal workflows.

## Validation Checklist

1. Replace imports using the table above.
2. Reinstall the package with `pip install -e .`.
3. Run `pytest`.
4. Run `ruff check .`.
