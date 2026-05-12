# Release Checklist

This project currently publishes manually. There is no GitHub Actions publish workflow.

## Preflight

```bash
uv sync --locked --extra dev --extra train
uv run ruff check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

Run manually gated real-model VLM validation before promoting the project out of Alpha:

```bash
uv sync --locked --extra dev --extra train --extra validation
CARVE_LM_RUN_REAL_VLM_VALIDATION=1 uv run python scripts/validation/validate_real_qwen_vlm.py --device cuda
```

On Kaggle, suppress cross-filesystem uv copy warnings and install the extra runtime packages before running a smaller first slice:

```bash
export UV_LINK_MODE=copy
uv sync --locked --extra dev --extra train --extra validation
CARVE_LM_RUN_REAL_VLM_VALIDATION=1 uv run python scripts/validation/validate_real_qwen_vlm.py \
  --models qwen2_5_vl \
  --components bridge \
  --device cuda \
  --dtype float16 \
  --keep-artifacts
```

## Build

```bash
uv run python -m build
uv run twine check dist/*
```

## Publish

Use scoped PyPI tokens from the maintainer account.

```bash
uv run twine upload dist/*
```

## Beta Readiness

Before changing `Development Status :: 3 - Alpha` to Beta in `pyproject.toml`:

1. Record successful manually gated Qwen2.5-VL and Qwen3-VL validation.
2. Confirm public compatibility aliases are either intentionally retained or removed in a breaking release.
3. Confirm release artifacts pass `twine check`.
4. Update `README.md`, `docs/status/README.md`, and `docs/status/roadmap.md` with the released version and validation scope.
