"""End-to-end and micro benchmarks for lazycogs reprojection.

These benchmarks require local benchmark data only for the public ``open()``
benchmarks. The direct reprojection micro-benchmarks use synthetic arrays and
run without any prepared dataset.

Run with:
    uv run pytest tests/benchmarks/ --benchmark-enable
    uv run pytest tests/benchmarks/ --benchmark-enable --benchmark-save=<name>
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj import CRS

import lazycogs
from lazycogs import FirstMethod, MedianMethod, MosaicMethodBase, set_reproject_workers
from lazycogs._warp import ReprojectRequest, ResamplingMethod, reproject_tile

from ._profiling import add_resource_profile
from .conftest import (
    BENCHMARK_BBOX,
    BENCHMARK_CRS,
    BENCHMARK_MULTIBAND,
    BENCHMARK_NATIVE_BBOX,
    BENCHMARK_NATIVE_CRS,
    BENCHMARK_NATIVE_RESOLUTION,
    BENCHMARK_SINGLE_BAND,
)


def _benchmark_request(
    *,
    same_grid: bool = False,
    dst_resolution: float = 20.0,
    bands: int = 3,
    size: int = 64,
) -> ReprojectRequest:
    """Build a representative small-window reprojection request."""
    src_crs = CRS.from_epsg(32632)
    src_transform = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 5_600_000.0)
    data = np.arange(bands * size * size, dtype=np.float32).reshape(bands, size, size)

    if same_grid:
        dst_transform = src_transform
        dst_width = size
        dst_height = size
    else:
        dst_transform = Affine(
            dst_resolution,
            0.0,
            500_320.0,
            0.0,
            -dst_resolution,
            5_599_680.0,
        )
        dst_width = size
        dst_height = size

    return ReprojectRequest(
        data=data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        dst_width=dst_width,
        dst_height=dst_height,
        nodata=-9999.0,
    )


def _profile_then_benchmark(benchmark, run):
    """Attach one-shot resource info, then run the pytest benchmark."""
    add_resource_profile(benchmark, run)
    benchmark(run)


@pytest.mark.benchmark
def test_small_window_nearest_reprojection(benchmark) -> None:
    """Benchmark the representative nearest-neighbor rust-warp path."""
    request = _benchmark_request(dst_resolution=20.0)
    _profile_then_benchmark(benchmark, lambda: reproject_tile(request))


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("label", "reproject_request", "resampling"),
    [
        pytest.param(
            "same_grid_noop",
            _benchmark_request(same_grid=True),
            ResamplingMethod.NEAREST,
            id="same_grid_noop",
        ),
        pytest.param(
            "nearest",
            _benchmark_request(dst_resolution=20.0),
            ResamplingMethod.NEAREST,
            id="nearest",
        ),
        pytest.param(
            "bilinear",
            _benchmark_request(dst_resolution=15.0),
            ResamplingMethod.BILINEAR,
            id="bilinear",
        ),
        pytest.param(
            "cubic",
            _benchmark_request(dst_resolution=15.0),
            ResamplingMethod.CUBIC,
            id="cubic",
        ),
    ],
)
def test_small_window_reprojection_modes(
    benchmark,
    label: str,
    reproject_request: ReprojectRequest,
    resampling: ResamplingMethod,
) -> None:
    """Show the cost gap between no-op, nearest, and interpolating modes."""
    benchmark.extra_info["mode"] = label

    def run() -> np.ndarray:
        return reproject_tile(
            ReprojectRequest(
                data=reproject_request.data,
                src_transform=reproject_request.src_transform,
                src_crs=reproject_request.src_crs,
                dst_transform=reproject_request.dst_transform,
                dst_crs=reproject_request.dst_crs,
                dst_width=reproject_request.dst_width,
                dst_height=reproject_request.dst_height,
                nodata=reproject_request.nodata,
                resampling=resampling,
            ),
        )

    _profile_then_benchmark(benchmark, run)


@pytest.mark.benchmark
def test_open_overhead(
    benchmark,
    benchmark_parquet: str,
    benchmark_open_kwargs: dict[str, object],
) -> None:
    """Phase 0: time the open() call without triggering any COG reads.

    Measures parquet queries, band discovery, time-step building, and grid
    computation.
    """

    def run() -> object:
        return lazycogs.open(
            benchmark_parquet,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
            **benchmark_open_kwargs,
        )

    _profile_then_benchmark(benchmark, run)


@pytest.mark.benchmark
def test_full_compute(
    benchmark,
    benchmark_parquet: str,
    benchmark_open_kwargs: dict[str, object],
) -> None:
    """Full pipeline: open + .compute() including local COG I/O."""

    def run() -> object:
        da = lazycogs.open(
            benchmark_parquet,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
            **benchmark_open_kwargs,
        )
        return da.compute()

    _profile_then_benchmark(benchmark, run)


@pytest.mark.benchmark
@pytest.mark.parametrize("method", [FirstMethod, MedianMethod], ids=["first", "median"])
def test_mosaic_method(
    benchmark,
    benchmark_parquet: str,
    benchmark_open_kwargs: dict[str, object],
    method: type[MosaicMethodBase],
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
            **benchmark_open_kwargs,
        )
        return da.compute()

    _profile_then_benchmark(benchmark, run)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_workers", [1, 4])
def test_reproject_workers(
    benchmark,
    expanded_benchmark_parquet: str,
    benchmark_open_kwargs: dict[str, object],
    n_workers: int,
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
            **benchmark_open_kwargs,
        )
        return da.compute()

    try:
        _profile_then_benchmark(benchmark, run)
    finally:
        # Reset to default so other benchmarks are not affected.
        set_reproject_workers(min(__import__("os").cpu_count() or 4, 4))


@pytest.mark.benchmark
def test_native_crs_resolution(
    benchmark,
    benchmark_parquet: str,
    benchmark_open_kwargs: dict[str, object],
) -> None:
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
            **benchmark_open_kwargs,
        )
        return da.compute()

    _profile_then_benchmark(benchmark, run)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "chunks",
    [None, {"time": 1}],
    ids=["no_dask", "dask_time_1"],
)
def test_time_step_parallelism(
    benchmark,
    expanded_benchmark_parquet: str,
    benchmark_open_kwargs: dict[str, object],
    chunks: dict | None,
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
            **benchmark_open_kwargs,
        )
        return da.compute()

    _profile_then_benchmark(benchmark, run)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "bands",
    [BENCHMARK_SINGLE_BAND, BENCHMARK_MULTIBAND],
    ids=["single_band", "multi_band"],
)
def test_band_access_pattern(
    benchmark,
    expanded_benchmark_parquet: str,
    benchmark_open_kwargs: dict[str, object],
    bands: list[str],
) -> None:
    """Compare single-band vs multi-band compute cost.

    Uses the expanded 12-time-step dataset with ``chunks={"time": 1}`` so each
    time step is a concurrent dask task. Multi-band reads share a single
    ``rustac.search_sync`` query and reproject all requested bands in one pass;
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
            **benchmark_open_kwargs,
        )
        return da.compute()

    _profile_then_benchmark(benchmark, run)
