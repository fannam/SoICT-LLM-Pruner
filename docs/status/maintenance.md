<!-- last_updated: 2026-05-12 -->

# Maintenance

## Maintainers

| Name | Email | Role |
|------|-------|------|
| Phan Hoang Nam | `phanhoangnam234@gmail.com` | Author / sole maintainer (per `pyproject.toml` and `LICENSE` copyright) |

GitHub handle: `fannam` (from `https://github.com/fannam/CarveLM.git` in `README.md`).

## Contribution Workflow

Source: [CONTRIBUTING.md](../../CONTRIBUTING.md).

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,train]"
```

Rules:

1. Library code lives under `src/carve_lm/`.
2. Runnable examples go under `examples/`; operational scripts go under `scripts/`.
3. Add or update tests for any behavior change.
4. Run `ruff check .` and `pytest` before opening a PR.

## Pull Request Guidelines

Per [CONTRIBUTING.md](../../CONTRIBUTING.md):

1. Explain motivation and user-visible impact.
2. Call out any breaking API changes.
3. Include validation details such as `pytest` and `ruff` output.

Recent commit-message style is short and intent-oriented (`refactor: adapters`, `add support for Qwen2.5-VL, not test yet`). Match it.

## CI/CD Status

Pipeline file: [.github/workflows/ci.yml](../../.github/workflows/ci.yml).

| Stage | Tool | Notes |
|-------|------|-------|
| Lint | `ruff check .` | Selectors: `E`, `F`, `I`. `target-version = "py310"`, line length 120. |
| Test | `pytest` | `addopts = "-q"`, `testpaths = ["tests"]`. |

| Field | Value |
|-------|-------|
| Trigger — push | branch `master` only |
| Trigger — PR | all PRs |
| Python matrix | 3.10, 3.11, 3.12 |
| OS | `ubuntu-latest` |
| Install | `uv sync --locked --extra dev --extra train` |

CI uses `uv run ruff check .` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest` so host-level pytest plugins do not affect repository tests.

No release / publish workflow is configured. `twine` is available in the `[dev]` extra for manual PyPI uploads.

## Licensing

- License: MIT.
- Copyright: `Copyright (c) 2026 Phan Hoang Nam`. See [LICENSE](../../LICENSE).
- License classifier in `pyproject.toml`: `License :: OSI Approved :: MIT License`.

## How External Contributors Can Help

Highest-leverage areas (derived from [issues.md](issues.md) and [roadmap.md](roadmap.md)):

- Add optional, manually gated real-model validation for Qwen2.5-VL and Qwen3-VL pruning + persistence round trips.
- Harden pruners for `carve_lm.vlm.components.vision` and `carve_lm.vlm.components.merger` against additional real model variants.
- Expand examples / scripts import-smoke coverage when new entrypoints are added.

Cross-references: [overview.md](overview.md) for stack and quick start, [architecture.md](architecture.md) for the namespace map.
