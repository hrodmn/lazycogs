"""End-to-end benchmarks using the public lazycogs.open() API.

These benchmarks require local benchmark data. See scripts/prepare_benchmark_data.py.

Run with:
    uv run pytest tests/benchmarks/ --benchmark-enable
    uv run pytest tests/benchmarks/ --benchmark-enable --benchmark-save=<name>
"""

import pytest

import lazycogs
from lazycogs import FirstMethod, MedianMethod, MosaicMethodBase, set_reproject_workers

from .conftest import (
    BENCHMARK_BBOX,
    BENCHMARK_CRS,
    BENCHMARK_MULTIBAND,
    BENCHMARK_SINGLE_BAND,
)


@pytest.mark.benchmark
def test_open_overhead(benchmark, benchmark_parquet: str) -> None:
    """Phase 0: time the open() call without triggering any COG reads.

    Measures parquet queries, band discovery, time-step building, and grid
    computation.
    """
    benchmark(
        lazycogs.open,
        benchmark_parquet,
        bbox=BENCHMARK_BBOX,
        crs=BENCHMARK_CRS,
        resolution=60.0,
    )


@pytest.mark.benchmark
def test_full_compute(benchmark, benchmark_parquet: str) -> None:
    """Full pipeline: open + .compute() including local COG I/O."""

    def run() -> object:
        da = lazycogs.open(
            benchmark_parquet,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
        )
        return da.compute()

    benchmark(run)


@pytest.mark.benchmark
@pytest.mark.parametrize("method", [FirstMethod, MedianMethod], ids=["first", "median"])
def test_mosaic_method(
    benchmark, benchmark_parquet: str, method: type[MosaicMethodBase]
) -> None:
    """Compare mosaic strategy cost end-to-end."""

    def run() -> object:
        da = lazycogs.open(
            benchmark_parquet,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
            time_period="P1M",
            mosaic_method=method,
        )
        return da.compute()

    benchmark(run)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
def test_reproject_workers(
    benchmark, expanded_benchmark_parquet: str, n_workers: int
) -> None:
    """Measure throughput as reprojection thread count varies.

    Uses the expanded 24-time-step dataset with ``chunks={"time": 1}`` so dask
    dispatches many concurrent tasks, putting real pressure on the per-chunk
    thread pool.  Validates the claim that memory-bandwidth saturation causes
    diminishing returns above 4 threads.
    """
    set_reproject_workers(n_workers)

    def run() -> object:
        da = lazycogs.open(
            expanded_benchmark_parquet,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
            time_period="P1M",
            chunks={"time": 1},
        )
        return da.compute()

    try:
        benchmark(run)
    finally:
        # Reset to default so other benchmarks are not affected.
        set_reproject_workers(min(__import__("os").cpu_count() or 4, 4))


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "bands",
    [BENCHMARK_SINGLE_BAND, BENCHMARK_MULTIBAND],
    ids=["single_band", "multi_band"],
)
def test_band_access_pattern(
    benchmark, expanded_benchmark_parquet: str, bands: list[str]
) -> None:
    """Compare single-band vs multi-band compute cost.

    Uses the expanded 24-time-step dataset with ``chunks={"time": 1}`` so each
    time step is a concurrent dask task.  Multi-band reads share a single
    ``rustac.search_sync`` query and reuse reprojection warp maps across bands;
    this benchmark quantifies that gain under concurrent load.
    """

    def run() -> object:
        da = lazycogs.open(
            expanded_benchmark_parquet,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
            time_period="P1M",
            bands=bands,
            chunks={"time": 1},
        )
        return da.compute()

    benchmark(run)
