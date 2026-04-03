# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,train]"
```

## Development Workflow

1. Keep library code under `src/soict_llm_pruner/`.
2. Put runnable examples in `examples/` and operational scripts in `scripts/`.
3. Add or update tests for any behavior change.
4. Run lint and tests before opening a PR.

```bash
ruff check .
pytest
```

## Package Layout

- `src/soict_llm_pruner/`: shipped library code
- `tests/`: unit and integration coverage
- `docs/`: architecture and migration notes
- `examples/`: minimal usage samples
- `scripts/`: operational training and recovery entrypoints

## Pull Requests

1. Explain the motivation and the user-visible impact.
2. Call out any breaking API changes.
3. Include validation details such as `pytest` and `ruff` output.
