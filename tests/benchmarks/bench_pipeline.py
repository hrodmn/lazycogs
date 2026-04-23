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
    BENCHMARK_NATIVE_BBOX,
    BENCHMARK_NATIVE_CRS,
    BENCHMARK_NATIVE_RESOLUTION,
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
@pytest.mark.parametrize("n_workers", [1, 4])
def test_reproject_workers(
    benchmark, expanded_benchmark_parquet: str, n_workers: int
) -> None:
    """Measure throughput as reprojection thread count varies.

    Uses the expanded 12-time-step dataset with ``chunks={"time": 1}`` so dask
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
def test_native_crs_resolution(benchmark, benchmark_parquet: str) -> None:
    """Full pipeline using the assets' native CRS and resolution.

    Requests data in EPSG:32612 at 10 m — exactly the source COG projection and
    pixel size — so reprojection should be a no-op.  Compared against
    ``test_full_compute`` (which reprojects to EPSG:5070 at 60 m) to quantify
    the overhead of the warp path when it is not needed.
    """

    def run() -> object:
        da = lazycogs.open(
            benchmark_parquet,
            bbox=BENCHMARK_NATIVE_BBOX,
            crs=BENCHMARK_NATIVE_CRS,
            resolution=BENCHMARK_NATIVE_RESOLUTION,
        )
        return da.compute()

    benchmark(run)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "chunks",
    [None, {"time": 1}],
    ids=["no_dask", "dask_time_1"],
)
def test_time_step_parallelism(
    benchmark, expanded_benchmark_parquet: str, chunks: dict | None
) -> None:
    """Compare native time-step thread pool vs Dask across 24 time steps.

    ``no_dask`` exercises the per-chunk ``ThreadPoolExecutor`` introduced in
    ``_raw_getitem``; ``dask_time_1`` dispatches one Dask task per time step.
    Both paths read the same data — the result shows relative overhead of Dask
    scheduling vs the built-in thread pool for this workload.
    """

    def run() -> object:
        da = lazycogs.open(
            expanded_benchmark_parquet,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
            time_period="P1M",
            chunks=chunks,
        )
        return da.compute()

    benchmark(run)


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

    Uses the expanded 12-time-step dataset with ``chunks={"time": 1}`` so each
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
