"""Benchmark DuckDB's share of per-date chunk wall time.

These benchmarks reuse the local fixtures from ``tests/benchmarks/conftest.py``.
They answer the U4 follow-up question from the concurrency refactor plan:
should lazycogs add a per-thread DuckDB client pool for true parallel query
execution, or is the current single-worker bounded executor already good enough?

Run with:
    uv run pytest tests/benchmarks/bench_duckdb_share.py -s
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pytest
from pyproj import Transformer

import lazycogs
import lazycogs._backend as backend

from .conftest import BENCHMARK_BBOX, BENCHMARK_CRS

_FULL_QUERY_BBOX_4326 = (-108.5, 37.5, -107.5, 38.5)
_WARMUP_RUNS = 1
_MEASURED_RUNS = 5


@dataclass
class _WorkloadSummary:
    name: str
    wall_mean_s: float
    wall_p95_s: float
    wall_p99_s: float
    duck_share_mean_pct: float
    duck_query_mean_ms: float
    duck_query_p95_ms: float
    duck_query_p99_ms: float
    date_total_mean_ms: float
    date_total_p95_ms: float
    date_total_p99_ms: float

    def format(self) -> str:
        return (
            f"{self.name}: wall mean={self.wall_mean_s:.3f}s "
            f"p95={self.wall_p95_s:.3f}s p99={self.wall_p99_s:.3f}s | "
            f"DuckDB share mean={self.duck_share_mean_pct:.1f}% | "
            f"duck query mean={self.duck_query_mean_ms:.1f}ms "
            f"p95={self.duck_query_p95_ms:.1f}ms p99={self.duck_query_p99_ms:.1f}ms | "
            f"per-date chunk mean={self.date_total_mean_ms:.1f}ms "
            f"p95={self.date_total_p95_ms:.1f}ms p99={self.date_total_p99_ms:.1f}ms"
        )


@contextmanager
def _collect_duckdb_share():
    duck_times: list[float] = []
    date_totals: list[float] = []
    orig_search = backend._search_items_sync
    orig_run_one_date = backend._run_one_date

    def wrapped_search(plan, date):
        t0 = time.perf_counter()
        try:
            return orig_search(plan, date)
        finally:
            duck_times.append(time.perf_counter() - t0)

    async def wrapped_run_one_date(t_idx, plan):
        t0 = time.perf_counter()
        try:
            return await orig_run_one_date(t_idx, plan)
        finally:
            date_totals.append(time.perf_counter() - t0)

    backend._search_items_sync = wrapped_search
    backend._run_one_date = wrapped_run_one_date
    try:
        yield duck_times, date_totals
    finally:
        backend._search_items_sync = orig_search
        backend._run_one_date = orig_run_one_date


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def _large_bbox_5070() -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs("EPSG:4326", BENCHMARK_CRS, always_xy=True)
    return transformer.transform_bounds(*_FULL_QUERY_BBOX_4326)


def _measure_workload(
    name: str,
    parquet_path: str,
    *,
    bbox: tuple[float, float, float, float],
    crs: str,
    resolution: float,
    time_period: str,
) -> _WorkloadSummary:
    runs: list[tuple[float, float, list[float], list[float]]] = []

    for _ in range(_WARMUP_RUNS + _MEASURED_RUNS):
        with _collect_duckdb_share() as (duck_times, date_totals):
            da = lazycogs.open(
                parquet_path,
                bbox=bbox,
                crs=crs,
                resolution=resolution,
                time_period=time_period,
                chunks={"time": 1},
            )
            t0 = time.perf_counter()
            da.compute()
            wall_s = time.perf_counter() - t0

        duck_share = sum(duck_times) / sum(date_totals)
        runs.append((wall_s, duck_share, list(duck_times), list(date_totals)))

    measured = runs[_WARMUP_RUNS:]
    wall_s = [run[0] for run in measured]
    duck_share_pct = [run[1] * 100 for run in measured]
    duck_queries_ms = [value * 1000 for run in measured for value in run[2]]
    date_totals_ms = [value * 1000 for run in measured for value in run[3]]

    return _WorkloadSummary(
        name=name,
        wall_mean_s=float(np.mean(wall_s)),
        wall_p95_s=_percentile(wall_s, 95),
        wall_p99_s=_percentile(wall_s, 99),
        duck_share_mean_pct=float(np.mean(duck_share_pct)),
        duck_query_mean_ms=float(np.mean(duck_queries_ms)),
        duck_query_p95_ms=_percentile(duck_queries_ms, 95),
        duck_query_p99_ms=_percentile(duck_queries_ms, 99),
        date_total_mean_ms=float(np.mean(date_totals_ms)),
        date_total_p95_ms=_percentile(date_totals_ms, 95),
        date_total_p99_ms=_percentile(date_totals_ms, 99),
    )


@pytest.mark.benchmark
def test_duckdb_share_small_bbox_many_time_steps(
    expanded_benchmark_parquet: str,
) -> None:
    """Small bbox, many monthly time steps from the expanded benchmark fixture."""
    summary = _measure_workload(
        "small bbox / many time steps",
        expanded_benchmark_parquet,
        bbox=BENCHMARK_BBOX,
        crs=BENCHMARK_CRS,
        resolution=60.0,
        time_period="P1M",
    )
    print(summary.format())


@pytest.mark.benchmark
def test_duckdb_share_large_bbox_few_time_steps(benchmark_parquet: str) -> None:
    """Large bbox, few daily time steps from the original benchmark fixture."""
    summary = _measure_workload(
        "large bbox / few time steps",
        benchmark_parquet,
        bbox=_large_bbox_5070(),
        crs=BENCHMARK_CRS,
        resolution=60.0,
        time_period="P1D",
    )
    print(summary.format())
