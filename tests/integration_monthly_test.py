import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import resource
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import rustac
from async_geotiff import GeoTIFF, Overview
from pyproj import Transformer

import lazycogs
from lazycogs import _backend, _chunk_reader

logging.basicConfig(level="WARN")
logging.getLogger("lazycogs").setLevel("DEBUG")
logger = logging.getLogger(__name__)


@dataclass
class PhaseStats:
    """Aggregate timing and count information for one phase."""

    calls: int = 0
    total_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        """Record one timed call."""
        self.calls += 1
        self.total_s += elapsed_s


@dataclass
class LoadBreakdown:
    """Timing breakdown for the monthly load path."""

    duckdb_search: PhaseStats
    geotiff_open: PhaseStats
    geotiff_read: PhaseStats
    overview_read: PhaseStats
    reproject: PhaseStats

    def log_summary(self, total_load_s: float) -> None:
        """Log a compact summary of where load time was spent."""
        parts = [
            ("duckdb_search", self.duckdb_search),
            ("geotiff_open", self.geotiff_open),
            ("geotiff_read", self.geotiff_read),
            ("overview_read", self.overview_read),
            ("reproject", self.reproject),
        ]

        logger.warning("[monthly load] wall_time=%.2fs", total_load_s)
        logger.warning(
            "[monthly load] phase timings below are cumulative across "
            "concurrent calls, so they can exceed wall time.",
        )
        for name, stats in parts:
            avg_ms = 1000.0 * stats.total_s / stats.calls if stats.calls else 0.0
            logger.warning(
                "[monthly load] %-14s cumulative=%.2fs calls=%d avg=%.2fms",
                name,
                stats.total_s,
                stats.calls,
                avg_ms,
            )


@dataclass
class ResourceProfile:
    """Resource profile captured during the monthly load."""

    wall_time_s: float
    cpu_user_s: float
    cpu_system_s: float
    cpu_total_s: float
    peak_rss_mb: float
    rss_before_mb: float
    rss_after_mb: float
    max_cpu_pct_of_wall: float
    sample_count: int

    def log_summary(self) -> None:
        """Log CPU and memory usage observed during the load."""
        logger.warning(
            "[monthly load resources] wall=%.2fs cpu_user=%.2fs cpu_system=%.2fs "
            "cpu_total=%.2fs cpu/wall=%.2fx peak_rss=%.0fMB rss_before=%.0fMB "
            "rss_after=%.0fMB peak_cpu_pct_of_wall=%.0f%% samples=%d",
            self.wall_time_s,
            self.cpu_user_s,
            self.cpu_system_s,
            self.cpu_total_s,
            self.cpu_total_s / self.wall_time_s if self.wall_time_s else 0.0,
            self.peak_rss_mb,
            self.rss_before_mb,
            self.rss_after_mb,
            self.max_cpu_pct_of_wall,
            self.sample_count,
        )


class ResourceSampler:
    """Sample process RSS and CPU usage while a block runs."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._peak_rss_mb = 0.0
        self._max_cpu_pct = 0.0
        self._sample_count = 0
        self._rss_before_mb = 0.0
        self._rss_after_mb = 0.0
        self._wall_start = 0.0
        self._wall_end = 0.0
        self._cpu_start: os.times_result | None = None
        self._cpu_end: os.times_result | None = None

    def __enter__(self) -> Self:
        self._rss_before_mb = _rss_mb()
        self._wall_start = time.perf_counter()
        self._cpu_start = os.times()
        self._peak_rss_mb = self._rss_before_mb
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join()
        self._cpu_end = os.times()
        self._wall_end = time.perf_counter()
        self._rss_after_mb = _rss_mb()
        self._peak_rss_mb = max(self._peak_rss_mb, self._rss_after_mb, _peak_rss_mb())

    def _run(self) -> None:
        last_wall = time.perf_counter()
        last_cpu = os.times()
        while not self._stop.wait(self.interval_s):
            now_wall = time.perf_counter()
            now_cpu = os.times()
            self._sample_count += 1
            self._peak_rss_mb = max(self._peak_rss_mb, _rss_mb(), _peak_rss_mb())
            wall_delta = now_wall - last_wall
            cpu_delta = (now_cpu.user - last_cpu.user) + (
                now_cpu.system - last_cpu.system
            )
            if wall_delta > 0:
                self._max_cpu_pct = max(
                    self._max_cpu_pct,
                    100.0 * cpu_delta / wall_delta,
                )
            last_wall = now_wall
            last_cpu = now_cpu

    def profile(self) -> ResourceProfile:
        """Return the aggregated resource profile."""
        if self._cpu_start is None or self._cpu_end is None:
            raise RuntimeError(
                "ResourceSampler.profile() called before sampling finished.",
            )
        cpu_user_s = self._cpu_end.user - self._cpu_start.user
        cpu_system_s = self._cpu_end.system - self._cpu_start.system
        wall_time_s = self._wall_end - self._wall_start
        return ResourceProfile(
            wall_time_s=wall_time_s,
            cpu_user_s=cpu_user_s,
            cpu_system_s=cpu_system_s,
            cpu_total_s=cpu_user_s + cpu_system_s,
            peak_rss_mb=self._peak_rss_mb,
            rss_before_mb=self._rss_before_mb,
            rss_after_mb=self._rss_after_mb,
            max_cpu_pct_of_wall=self._max_cpu_pct,
            sample_count=self._sample_count,
        )


def _parquet_path(
    href: str,
    collections: list[str],
    datetime: str,
    bbox: list[float],
    limit: int,
) -> Path:
    """Return a cache path for a STAC search derived from its parameters."""
    params = {
        "href": href,
        "collections": sorted(collections),
        "datetime": datetime,
        "bbox": [round(v, 6) for v in bbox],
        "limit": limit,
    }
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    return Path(f"/tmp/stac_{digest}.parquet")


def _rss_mb() -> float:
    """Return current RSS of this process in MB on Linux."""
    with Path("/proc/self/status").open() as file_handle:
        for line in file_handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _peak_rss_mb() -> float:
    """Return peak RSS of this process in MB on Linux."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024


@contextlib.contextmanager
def measure(label: str):
    """Log wall time and RSS change for a block."""
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    rss_after = _rss_mb()
    logger.warning(
        "[%s] time=%.2fs rss_before=%.0fMB rss_after=%.0fMB delta=%+.0fMB",
        label,
        elapsed,
        rss_before,
        rss_after,
        rss_after - rss_before,
    )


@contextlib.contextmanager
def instrument_monthly_load() -> LoadBreakdown:
    """Patch internal helpers to measure monthly load sub-phases."""
    search_stats = PhaseStats()
    geotiff_open_stats = PhaseStats()
    geotiff_read_stats = PhaseStats()
    overview_read_stats = PhaseStats()
    reproject_stats = PhaseStats()

    original_search = _backend._search_items_sync
    reproject_helper_name = (
        "_reproject_bands"
        if hasattr(_chunk_reader, "_reproject_bands")
        else "_apply_bands_with_warp_cache"
    )
    original_reproject = getattr(_chunk_reader, reproject_helper_name)
    original_geotiff_open = GeoTIFF.open
    original_geotiff_read = GeoTIFF.read
    original_overview_read = Overview.read

    def timed_search(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_search(*args, **kwargs)
        search_stats.add(time.perf_counter() - t0)
        return result

    def timed_reproject(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_reproject(*args, **kwargs)
        reproject_stats.add(time.perf_counter() - t0)
        return result

    async def timed_geotiff_open(cls, *args, **kwargs):
        t0 = time.perf_counter()
        result = await original_geotiff_open(*args, **kwargs)
        geotiff_open_stats.add(time.perf_counter() - t0)
        return result

    async def timed_geotiff_read(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = await original_geotiff_read(self, *args, **kwargs)
        geotiff_read_stats.add(time.perf_counter() - t0)
        return result

    async def timed_overview_read(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = await original_overview_read(self, *args, **kwargs)
        overview_read_stats.add(time.perf_counter() - t0)
        return result

    _backend._search_items_sync = timed_search
    setattr(_chunk_reader, reproject_helper_name, timed_reproject)
    GeoTIFF.open = classmethod(timed_geotiff_open)
    GeoTIFF.read = timed_geotiff_read
    Overview.read = timed_overview_read

    try:
        yield LoadBreakdown(
            duckdb_search=search_stats,
            geotiff_open=geotiff_open_stats,
            geotiff_read=geotiff_read_stats,
            overview_read=overview_read_stats,
            reproject=reproject_stats,
        )
    finally:
        _backend._search_items_sync = original_search
        setattr(_chunk_reader, reproject_helper_name, original_reproject)
        GeoTIFF.open = original_geotiff_open
        GeoTIFF.read = original_geotiff_read
        Overview.read = original_overview_read


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for the integration script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-explain",
        action="store_true",
        help="Skip the explain() dry run before loading data.",
    )
    return parser.parse_args()


async def run(*, skip_explain: bool = False) -> None:
    """Run the monthly southwest scenario outside Jupyter."""
    dst_crs = "epsg:3310"
    dst_bbox = (-444_000, -609_000, 681_000, 500_000)

    stac_href = "https://earth-search.aws.element84.com/v1"
    collections = ["sentinel-2-c1-l2a"]
    datetime = "2025-03-01/2025-06-30"
    limit = 100

    transformer = Transformer.from_crs(dst_crs, "epsg:4326", always_xy=True)
    bbox_4326 = list(transformer.transform_bounds(*dst_bbox))

    items_parquet = _parquet_path(
        href=stac_href,
        collections=collections,
        datetime=datetime,
        bbox=bbox_4326,
        limit=limit,
    )
    logger.warning("cache: %s", items_parquet)

    if not items_parquet.exists():
        with measure("search_to parquet cache"):
            await rustac.search_to(
                str(items_parquet),
                href=stac_href,
                collections=collections,
                datetime=datetime,
                bbox=bbox_4326,
                limit=limit,
            )

    store = lazycogs.store_for(str(items_parquet), skip_signature=True)

    with measure("monthly open"):
        ca_monthly = lazycogs.open(
            str(items_parquet),
            crs=dst_crs,
            bbox=dst_bbox,
            resolution=300,
            time_period="P1M",
            bands=["red", "green", "blue"],
            dtype="int16",
            filter="eo:cloud_cover < 50",
            sortby="eo:cloud_cover",
            store=store,
        )
    logger.warning("monthly array: %s", ca_monthly)

    ca_may = ca_monthly.chunk(x=1024, y=1024).sel(time="2025-05-01")
    logger.warning("monthly may chunked array: %s", ca_may)

    if not skip_explain:
        with measure("monthly explain"):
            plan = ca_may.lazycogs.explain(fetch_headers=True)
        logger.warning("\n%s", plan.summary())

    with instrument_monthly_load() as breakdown, ResourceSampler() as resource_sampler:
        t0 = time.perf_counter()
        loaded = await ca_may.load_async()
        total_load_s = time.perf_counter() - t0

    logger.warning("loaded shape: %s", loaded.shape)
    breakdown.log_summary(total_load_s)
    resource_sampler.profile().log_summary()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run(skip_explain=args.skip_explain))
