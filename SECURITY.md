# Security Policy

## Supported Versions

CarveLM is pre-1.0. Security fixes target the current `master` branch and the latest published package version when one exists.

## Reporting a Vulnerability

Report security issues privately to Phan Hoang Nam at `phanhoangnam234@gmail.com`.

Please include:

- affected version or commit
- a minimal reproduction when possible
- expected and observed impact

## Dependency Policy

The repository keeps `uv.lock` for reproducible development and CI. End-user installs keep flexible minimum-version runtime dependencies in `pyproject.toml`; users who need a strict supply-chain audit should pin and audit dependencies in their own deployment environment.
