"""Fixtures for end-to-end benchmarks.

Run ``uv run python scripts/prepare_benchmark_data.py`` before using these fixtures.
"""

from pathlib import Path

import pytest

_DATA_DIR = Path(__file__).parents[2] / ".benchmark_data"
_PARQUET = _DATA_DIR / "benchmark_items.parquet"

# Small area within the benchmark dataset bbox (western Colorado, EPSG:5070)
# ~30km x 30km centred within the STAC query bbox [-108.5, 37.5, -107.5, 38.5]
BENCHMARK_BBOX = (-1_056_282.0, 1_713_715.0, -1_026_282.0, 1_743_715.0)
BENCHMARK_CRS = "EPSG:5070"


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
