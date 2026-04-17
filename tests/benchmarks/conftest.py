"""Fixtures for end-to-end benchmarks.

Run ``uv run python scripts/prepare_benchmark_data.py`` before using these fixtures.
"""

from pathlib import Path

import pytest

_DATA_DIR = Path(__file__).parents[2] / ".benchmark_data"
_PARQUET = _DATA_DIR / "benchmark_items.parquet"
_EXPANDED_PARQUET = _DATA_DIR / "expanded_benchmark_items.parquet"

# Small area within the benchmark dataset bbox (western Colorado, EPSG:5070)
# ~30km x 30km centred within the STAC query bbox [-108.5, 37.5, -107.5, 38.5]
BENCHMARK_BBOX = (-1_056_282.0, 1_713_715.0, -1_026_282.0, 1_743_715.0)
BENCHMARK_CRS = "EPSG:5070"

# Band names present in the benchmark dataset (Sentinel-2 red + narrow NIR).
# Single-band tests use just red; multi-band tests use both so the shared
# warp-map path in MultiBandStacBackendArray is exercised.
BENCHMARK_SINGLE_BAND: list[str] = ["red"]
BENCHMARK_MULTIBAND: list[str] = ["red", "nir08"]


@pytest.fixture(scope="session")
def benchmark_parquet() -> str:
    """Path to the local benchmark parquet file.

    Skips the test if the benchmark data has not been downloaded yet.
    """
    if not _PARQUET.exists():
        pytest.skip(
            "Benchmark data not found. "
            "Run `uv run python scripts/prepare_benchmark_data.py` first."
        )
    return str(_PARQUET)


@pytest.fixture(scope="session")
def expanded_benchmark_parquet() -> str:
    """Path to the expanded benchmark parquet with 24 synthetic time steps.

    Skips the test if the expanded data has not been generated yet.
    """
    if not _EXPANDED_PARQUET.exists():
        pytest.skip(
            "Expanded benchmark data not found. "
            "Run `uv run python scripts/prepare_benchmark_data.py` first."
        )
    return str(_EXPANDED_PARQUET)
