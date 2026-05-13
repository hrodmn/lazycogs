"""Fixtures for end-to-end benchmarks.

Run ``uv run python scripts/prepare_benchmark_data.py`` before using these fixtures.
"""

from pathlib import Path
from urllib.parse import urlparse

import pytest
from rustac import DuckdbClient

_DATA_DIR = Path(__file__).parents[2] / ".benchmark_data"
_COG_DIR = _DATA_DIR / "cogs"
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

# Native CRS/resolution for the benchmark dataset (Sentinel-2 UTM Zone 12N).
# Used to benchmark the no-reprojection path.
BENCHMARK_NATIVE_CRS = "EPSG:32612"
# 30 km x 30 km area fully within the source COG footprint, snapped to the
# native 10 m pixel grid so col_off and row_off are whole numbers.
# Source transform origin: (699960, 4200000), pixel size: (10, -10).
# col_off=1000, row_off=200 from the top-left corner.
BENCHMARK_NATIVE_BBOX = (709960.0, 4_168_000.0, 739960.0, 4_198_000.0)
BENCHMARK_NATIVE_RESOLUTION = 10.0  # red band native pixel size


def _benchmark_path_from_href(href: str) -> str:
    """Return the local file path for a benchmark asset HREF.

    Benchmark parquet files may outlive a local checkout move, leaving absolute
    ``file://`` HREFs that still point at an older workspace root. When that
    happens, remap the asset path back into this checkout's ``.benchmark_data``
    directory using the stable ``cogs/<item>/<band>.tif`` suffix.
    """
    parsed = urlparse(href)
    if parsed.scheme != "file":
        return parsed.path.lstrip("/")

    path = Path(parsed.path)
    if path.exists():
        return str(path)

    try:
        cogs_index = path.parts.index("cogs")
    except ValueError:
        return str(path)

    remapped = _DATA_DIR.joinpath(*path.parts[cogs_index:])
    return str(remapped)


def _assert_benchmark_assets_available(parquet_path: Path) -> None:
    """Skip benchmark tests when the local benchmark assets are unavailable."""
    client = DuckdbClient()
    items = client.search(str(parquet_path), max_items=1, include=["assets"])
    if not items:
        pytest.skip(f"Benchmark parquet {parquet_path} contains no STAC items.")

    assets = items[0].get("assets", {})
    for band in BENCHMARK_MULTIBAND:
        href = assets.get(band, {}).get("href")
        if not href:
            continue
        if Path(_benchmark_path_from_href(href)).exists():
            return

    pytest.skip(
        "Benchmark parquet references local COG paths that are unavailable in "
        "this checkout. Re-run `uv run python scripts/prepare_benchmark_data.py` "
        "to refresh the benchmark dataset.",
    )


@pytest.fixture(scope="session")
def benchmark_parquet() -> str:
    """Path to the local benchmark parquet file.

    Skips the test if the benchmark data has not been downloaded yet.
    """
    if not _PARQUET.exists():
        pytest.skip(
            "Benchmark data not found. "
            "Run `uv run python scripts/prepare_benchmark_data.py` first.",
        )
    _assert_benchmark_assets_available(_PARQUET)
    return str(_PARQUET)


@pytest.fixture(scope="session")
def expanded_benchmark_parquet() -> str:
    """Path to the expanded benchmark parquet with 24 synthetic time steps.

    Skips the test if the expanded data has not been generated yet.
    """
    if not _EXPANDED_PARQUET.exists():
        pytest.skip(
            "Expanded benchmark data not found. "
            "Run `uv run python scripts/prepare_benchmark_data.py` first.",
        )
    _assert_benchmark_assets_available(_EXPANDED_PARQUET)
    return str(_EXPANDED_PARQUET)


@pytest.fixture(scope="session")
def benchmark_open_kwargs() -> dict[str, object]:
    """Common ``lazycogs.open()`` kwargs for local benchmark datasets."""
    return {"path_from_href": _benchmark_path_from_href}
