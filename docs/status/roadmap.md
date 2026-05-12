<!-- last_updated: 2026-05-12 -->

# Roadmap

Priorities are inferred from recent commits, reserved namespaces, and `[TBD]`-marked surface area in code. No formal roadmap document exists in the repo — items below should be confirmed with the maintainer before treating them as committed work.

## Priority Legend

- **P0** — blocking for next release / Alpha → Beta readiness.
- **P1** — important, target this iteration.
- **P2** — nice to have / longer horizon.

## Active Milestones

| Priority | Item | Source / signal |
|----------|------|-----------------|
| P0 | Validate Qwen2.5-VL decoder / vision / merger pruning synthetically. | Covered by synthetic integration tests; real-model validation remains manual. |
| P0 | Validate Qwen3-VL decoder / vision / merger pruning synthetically. | Covered by synthetic integration tests; real-model validation remains manual. |
| P0 | Align CI trigger branch (`main`) with active branch (`master`). | Resolved in `.github/workflows/ci.yml`. |
| P1 | Harden vision-component pruners against additional real model variants. | Current coverage targets Qwen-style synthetic layouts. |
| P1 | Harden merger-component pruners against additional real model variants. | Current coverage targets Qwen-style synthetic layouts. |
| P1 | Maintain `tests/integration/` coverage. | Integration suite now exists and should grow with new persistence or cross-component behavior. |

## Short-term Goals

- Keep synthetic VLM regression tests fast enough for CI and document any real-model validation separately.
- Promote `carve-lm` from Alpha to Beta after VLM validation. Requires updating the `Development Status` classifier in `pyproject.toml`.
- Publish first release to PyPI (`twine` already in `[dev]` extras — no release artifacts in `.github/workflows/` yet).

## Medium-term Goals

- Component-scoped pruning stacks for the VLM `vision` and `merger` registries, including manifest persistence parity with the language component.
- Distillation parity between LLM and VLM paths for `HybridOTDistiller` (`hybrid_ot.py` exists in both namespaces — verify behavioral parity).
- Add evaluation beyond latency (e.g. quality metrics) — currently only `LLMMeasurer` / `VLMMeasurer` for latency / throughput.

## Long-term / Stretch

- Additional model families beyond Llama / Qwen / Mistral via the `GenericDecoderModelAdapter` extension point.
- Public documentation site (currently only Markdown under `docs/`).

See [issues.md](issues.md) for blocking gaps, [maintenance.md](maintenance.md) for contribution rules.
