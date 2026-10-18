# ADR 0003: Conda + pip instead of uv; pnpm for the front end

Status: accepted

## Context
The design doc assumes `uv` for Python. The owner provided a conda environment named
`hft_forecast` and asked that all Python packages be installed into it with `pip`.

## Decision
- Python: `pip install -e ".[dev]"` inside the `hft_forecast` conda env. `pyproject.toml` remains
  the single source of dependency truth, so switching to `uv` later needs no code change.
- Front end: `pnpm` as the design doc recommends (installed globally via npm).
- The Makefile uses `$CONDA_PREFIX/bin/python` when a conda env is active.

## Consequences
No lock file for Python in v1; reproducibility of results relies on run metadata recording
package versions (see run store).
