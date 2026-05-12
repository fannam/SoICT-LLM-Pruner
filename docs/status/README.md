<!-- last_updated: 2026-05-12 -->

# CarveLM — Status Dashboard

Project status index. Each entry links to a focused report.

## Snapshot

| Field | Value |
|-------|-------|
| Package | `carve-lm` |
| Version | `0.1.0` |
| Development stage | Alpha (`Development Status :: 3 - Alpha`) |
| License | MIT |
| Primary language | Python (`>=3.10`) |
| Maintainer | Phan Hoang Nam (`phanhoangnam234@gmail.com`) |
| Default branch | `master` |
| Last updated | 2026-05-12 |

## Sub-reports

| File | Status | Summary |
|------|--------|---------|
| [overview.md](overview.md) | ✅ | Purpose, audience, stack, quick start commands. |
| [architecture.md](architecture.md) | ✅ | Tri-level framework, namespace layout, data flow. |
| [roadmap.md](roadmap.md) | ✅ | Active milestones and remaining real-model validation follow-ups. |
| [issues.md](issues.md) | ✅ | Resolved CI, synthetic VLM validation, integration-test, and docs-layout gaps. |
| [maintenance.md](maintenance.md) | ✅ | Solo maintainer, MIT, GitHub Actions matrix CI. |

Legend: ✅ healthy · ⚠️ has known gaps · 🔴 blocking issue.

## Overall Health

- Core LLM pruning stack (Llama / Qwen2 / Qwen3 / Mistral) is registered and unit-tested.
- VLM adapters for Qwen2.5-VL and Qwen3-VL have synthetic decoder, vision, merger, and persistence coverage.
- VLM `vision` and `merger` namespaces ship public estimators and pruners for Qwen-style components.
- CI workflow targets push to branch `master` and runs through the locked `uv` environment.

Cross-references: see [architecture.md](architecture.md) for component map, [issues.md](issues.md) for known gaps.
