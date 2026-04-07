# Migration Guide

This refactor is intentionally breaking. Old top-level packages such as `estimator`, `pruner`, `distiller`, `measurer`, `soict_llm_pruner_core`, and `llm_pruner_paper` are removed.

## Import Mapping

| Old import | New import |
| --- | --- |
| `from estimator import ...` | `from soict_llm_pruner.estimators import ...` |
| `from pruner import ...` | `from soict_llm_pruner.pruners import ...` |
| `create_estimator("element.activation", ...)` | `create_estimator("activation.element", ...)` |
| `create_estimator("element.weight_magnitude", ...)` | `create_estimator("magnitude.element", ...)` |
| `create_estimator("layer.similarity", ...)` | `create_estimator("similarity.layer", ...)` |
| `create_estimator("block.similarity", ...)` | `create_estimator("similarity.block", ...)` |
| `create_estimator("block.perplexity", ...)` | `create_estimator("perplexity.block", ...)` |
| `create_pruner("element", ...)` | `create_pruner("width", ...)` |
| `create_pruner("layer", ...)` | `create_pruner("component", ...)` |
| `create_pruner("block", ...)` | `create_pruner("depth.block", ...)` |
| `from distiller.hybrid_distiller import HybridDistiller` | `from soict_llm_pruner.distillation import HybridDistiller` |
| `from distiller.logits_distiller import LogitsDistiller` | `from soict_llm_pruner.distillation import LogitsDistiller` |
| `from measurer.general_measure import LLMMeasurer` | `from soict_llm_pruner.evaluation import LLMMeasurer` |
| `from soict_llm_pruner_core import BaseModelAdapter` | `from soict_llm_pruner.adapters import BaseModelAdapter` |
| `from soict_llm_pruner_core import DecoderModelAdapter` | `from soict_llm_pruner.adapters import DecoderModelAdapter` |
| `from soict_llm_pruner_core import ESTIMATOR_REGISTRY` | `from soict_llm_pruner.core import ESTIMATOR_REGISTRY` |
| `from soict_llm_pruner_core import PRUNER_REGISTRY` | `from soict_llm_pruner.core import PRUNER_REGISTRY` |
| `from soict_llm_pruner_core import PRUNING_STRATEGY_REGISTRY` | `from soict_llm_pruner.core import PRUNING_STRATEGY_REGISTRY` |
| `from llm_pruner_paper import BlockWiseConfig` | `from soict_llm_pruner.pruners.structured import BlockWiseConfig` |
| `from llm_pruner_paper import ChannelWiseConfig` | `from soict_llm_pruner.pruners.structured import ChannelWiseConfig` |
| `from llm_pruner_paper import LayerWiseConfig` | `from soict_llm_pruner.pruners.structured import LayerWiseConfig` |
| `from llm_pruner_paper import ImportanceConfig` | `from soict_llm_pruner.pruners.structured import ImportanceConfig` |
| `from llm_pruner_paper import BlockWiseLLMPruner` | `from soict_llm_pruner.pruners.structured import StructuredBlockPruner` |
| `from llm_pruner_paper import ChannelWiseLLMPruner` | `from soict_llm_pruner.pruners.structured import StructuredChannelPruner` |
| `from llm_pruner_paper import LayerWiseLLMPruner` | `from soict_llm_pruner.pruners.structured import StructuredLayerPruner` |

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
- Structured pruning compatibility imports still work, but the canonical API now lives under `soict_llm_pruner.pruners`.
- `save_pruned()` and `load_pruned()` remain the supported persistence path for structured outputs.
- Recovery and distillation utilities are still separate from structured prune execution.

## Validation Checklist

1. Replace imports using the table above.
2. Reinstall the package with `pip install -e .`.
3. Run `pytest`.
4. Run `ruff check .`.
