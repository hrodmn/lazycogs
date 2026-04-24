# Contributing to lazycogs

## Development setup

lazycogs uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it first if you haven't already.

Clone the repo and install the dev dependencies:

```bash
git clone https://github.com/hrodmn/lazycogs.git
cd lazycogs
uv sync
```

Install the pre-commit hooks so that ruff runs automatically before each commit:

```bash
uv run pre-commit install
```

## Running tests

```bash
uv run pytest
```

The test suite uses synthetic COG fixtures generated at runtime — no external data required.

### Benchmarks

Benchmarks live in `tests/benchmarks/` and are excluded from the default test run. They require a local Sentinel-2 dataset that you download once before running:

```bash
uv run python scripts/prepare_benchmark_data.py
```

This queries the Element84 Earth Search STAC API, downloads a small set of COG assets to `.benchmark_data/`, and writes local parquet index files. Pass `--overwrite` to re-download if needed.

Once the data is in place:

```bash
uv run pytest tests/benchmarks --benchmark-enable
```

## Building docs

Install the docs dependencies and serve the site locally:

```bash
uv run --group docs mkdocs serve
```
