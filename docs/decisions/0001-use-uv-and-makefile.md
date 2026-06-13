# 0001 — Use uv and a Makefile

## Context

This is a learning repo maintained by a senior engineer who wants standard,
reproducible project hygiene. Python tooling options include bare pip + venv,
Poetry, and uv. Commands tend to sprawl across READMEs and shell history.

## Decision

- Use **uv** as the sole package and project manager. The project is defined by
  `pyproject.toml` + `uv.lock`. Use `uv add`, `uv sync`, `uv run`, `uv venv`.
  torch is pinned to the CPU wheel index in `pyproject.toml`.
- Use a **Makefile** as the single entry point for every repeatable command.
  `make help` is the discovery surface.

## Consequences

- Reproducible installs via the lockfile; fast resolution.
- One obvious place to look for "how do I run X" — the Makefile.
- Contributors must have uv installed; bare pip/poetry workflows are out.
