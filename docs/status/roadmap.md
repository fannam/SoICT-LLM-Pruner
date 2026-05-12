<!-- last_updated: 2026-05-12 -->

# Roadmap

Priorities are inferred from recent commits, remaining validation scope, and release-readiness gaps. No formal roadmap document exists in the repo — items below should be confirmed with the maintainer before treating them as committed work.

## Priority Legend

- **P0** — blocking for next release / Alpha → Beta readiness.
- **P1** — important, target this iteration.
- **P2** — nice to have / longer horizon.

## Active Milestones

| Priority | Item | Source / signal |
|----------|------|-----------------|
| P0 | Run and record manually gated real-model Qwen2.5-VL validation for decoder / vision / merger pruning and persistence. | `scripts/validation/validate_real_qwen_vlm.py` exists; results remain environment-dependent and outside CI. |
| P0 | Run and record manually gated real-model Qwen3-VL validation for decoder / vision / merger pruning and persistence. | `scripts/validation/validate_real_qwen_vlm.py` exists; results remain environment-dependent and outside CI. |
| P1 | Harden vision-component pruners against additional real model variants. | Current coverage targets Qwen-style synthetic layouts. |
| P1 | Harden merger-component pruners against additional real model variants. | Current coverage targets Qwen-style synthetic layouts. |
| P1 | Maintain `tests/integration/` coverage. | Integration suite now exists and should grow with new persistence or cross-component behavior. |
| P1 | Add release workflow or keep a documented PyPI release command current. | Manual release checklist exists in `docs/release.md`; no publish workflow is configured. |
| P1 | Maintain an Alpha to Beta readiness checklist. | Beta requires recorded manual real-model VLM validation and release-doc updates. |

## Completed Milestones

| Item | Resolution |
|------|------------|
| Synthetic Qwen2.5-VL decoder / vision / merger pruning validation. | Covered by `tests/integration/`. |
| Synthetic Qwen3-VL decoder / vision / merger pruning validation. | Covered by `tests/integration/`. |
| CI trigger branch alignment. | `.github/workflows/ci.yml` now targets push to `master`. |
| Docs-layout drift. | Status and architecture docs now reference `Notebook/` and the existing `tests/integration/` directory. |
| Manually gated real-model validation entrypoint. | Added `scripts/validation/validate_real_qwen_vlm.py`; execution remains opt-in. |

## Short-term Goals

- Keep synthetic VLM regression tests fast enough for CI and document real-model validation as a manual or opt-in workflow.
- Promote `carve-lm` from Alpha to Beta after manual real-model VLM validation. Requires updating the `Development Status` classifier in `pyproject.toml`.
- Publish first release to PyPI. Manual commands are documented in [docs/release.md](../release.md); no release artifacts in `.github/workflows/` yet.

## Medium-term Goals

- Extend component-scoped VLM pruning beyond the current Qwen-style synthetic validation set.
- Keep distillation parity between LLM and VLM paths as shared internals evolve.
- Add evaluation beyond latency (e.g. quality metrics) — currently only `LLMMeasurer` / `VLMMeasurer` for latency / throughput.

## Long-term / Stretch

- Additional model families beyond Llama / Qwen / Mistral via the `GenericDecoderModelAdapter` extension point.
- Public documentation site (currently only Markdown under `docs/`).

See [issues.md](issues.md) for blocking gaps, [maintenance.md](maintenance.md) for contribution rules.
