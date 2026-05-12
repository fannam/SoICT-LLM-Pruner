<!-- last_updated: 2026-05-12 -->

# Issues & Gaps

Concrete known issues, technical debt, and coverage gaps observable from the repository state. Severity reflects impact, not urgency.

## Severity Legend

- 🔴 **High** — broken or absent functionality that blocks a stated capability.
- ⚠️ **Medium** — works but with explicit caveats or missing validation.
- 🟡 **Low** — minor inconsistency, naming drift, or documentation lag.

## Active Bugs / Risks

No active high-severity bugs are known after the CI, synthetic VLM validation, documentation-layout, and integration-test fixes.

Resolved in the current maintenance pass:

| # | Previous Severity | Area | Resolution |
|---|-------------------|------|------------|
| 1 | 🔴 | CI | `.github/workflows/ci.yml` now runs on `push` to `master` and uses `uv` with the locked environment. |
| 2 | ⚠️ | VLM / Qwen2.5-VL | Synthetic Qwen2.5-VL end-to-end pruning, forward, and persistence coverage is recorded under `tests/integration/`. |
| 3 | ⚠️ | VLM / Qwen3-VL | Synthetic Qwen3-VL end-to-end pruning, forward, and persistence coverage is recorded under `tests/integration/`. |
| 4 | ⚠️ | Docs vs. layout | `tests/integration/` now exists and backs the architecture claim. |
| 5 | ⚠️ | Docs vs. layout | `docs/architecture.md` now references `Notebook/`, matching the repository layout. |

## Limitations

- Real-model Qwen2.5-VL / Qwen3-VL validation is not part of CI. Current coverage uses synthetic model layouts to avoid network, large downloads, and GPU requirements.
- **Compatibility aliases** (`ElementPruner`, `StructuredBlockPruner`, etc.) remain for one release and emit `DeprecationWarning`. They will be removed without further notice.

## Technical Debt

- Dual `_compat.py` modules exist in both `carve_lm.llm.estimators` / `carve_lm.llm.pruners` and the VLM mirror. Schedule removal alongside the next breaking release.
- Some distillation code remains domain-specific because VLM forwarding must preserve multimodal batch keys. Shared OT and wrapper utilities live under an internal shared module.
- `examples/` and `scripts/recovery/` have import-smoke coverage, but their full training/download workflows remain manual.

## Security Notes

- License is MIT. No security policy file (`SECURITY.md`) exists.
- `SECURITY.md` exists. No dependency-pinning lockfile is required for end users (`uv.lock` is enforced for repo development and CI only). Runtime deps use minimum-version bounds — supply-chain audit is the user's responsibility.
- Recovery scripts in `scripts/recovery/` use `wandb`, `datasets`, and `accelerate`. They run user-supplied training configs; review before executing in shared environments.

## Test Coverage Gaps

Observed test files:

- `tests/llm/unit/`: `test_structured_pruning.py`, `test_distillation.py`, `test_activation_estimator_flow.py`, `test_weight_magnitude_flow.py`, `test_layer_perplexity.py`, `test_pruned_auto_model.py`, `test_namespace_smoke.py`.
- `tests/vlm/unit/`: `test_qwen3_vl_support.py`, `test_vlm_distillation.py`, `test_evaluation.py`, `test_qwen2_5_vl_support.py`.

Gaps:

- Integration tests cover synthetic Qwen2.5-VL and Qwen3-VL pruning / persistence flows.
- VLM `save_pruned` / `load_pruned` round trips are covered for language, vision, and merger components.
- Vision and merger registries are covered by namespace smoke tests.
- `LLMMeasurer` and `VLMMeasurer` have synthetic smoke coverage for latency / throughput APIs.

See [roadmap.md](roadmap.md) for prioritized fixes.
