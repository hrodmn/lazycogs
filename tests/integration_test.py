import asyncio
import contextlib
import hashlib
import json
import logging
import time
from pathlib import Path

import rustac
from pyproj import Transformer

import lazycogs

logging.basicConfig(level="WARN")
logging.getLogger("lazycogs").setLevel("INFO")


def _parquet_path(
    href: str,
    collections: list[str],
    datetime: str,
    bbox: list[float],
    limit: int,
) -> Path:
    """Return a cache path for a STAC search derived from its parameters.

    The filename encodes a short hash of the search parameters so that
    different searches never collide and the right cached file is always used.

    Args:
        href: STAC API endpoint URL.
        collections: Collection IDs to search.
        datetime: ISO 8601 datetime or interval string.
        bbox: Bounding box as ``[minx, miny, maxx, maxy]`` in EPSG:4326.
        limit: Maximum number of items to return.

    Returns:
        Path under ``/tmp`` of the form ``stac_<12-char-hash>.parquet``.

    """
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
    """Return current RSS of this process in MB (Linux only)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


@contextlib.contextmanager
def measure(label: str):
    """Log wall time and RSS change for a block."""
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    rss_after = _rss_mb()
    print(
        f"[{label}] "
        f"time={elapsed:.2f}s  "
        f"rss_before={rss_before:.0f}MB  "
        f"rss_after={rss_after:.0f}MB  "
        f"delta={rss_after - rss_before:+.0f}MB"
    )


async def run():
    dst_crs = "epsg:5070"
    dst_bbox = (-700_000, 2_220_000, 600_000, 2_930_000)

    stac_href = "https://earth-search.aws.element84.com/v1"
    collections = ["sentinel-2-c1-l2a"]
    datetime = "2025-06-01/2025-06-30"
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
    print(f"cache: {items_parquet}")

    if not items_parquet.exists():
        await rustac.search_to(
            str(items_parquet),
            href=stac_href,
            collections=collections,
            datetime=datetime,
            bbox=bbox_4326,
            limit=limit,
        )

    # --- daily time steps ---
    da = await lazycogs.open_async(
        str(items_parquet),
        crs=dst_crs,
        bbox=dst_bbox,
        resolution=100,
        time_period="P1D",
        bands=["red", "green", "blue"],
        dtype="int16",
    )
    print(f"\ndaily array: {da}")

    with measure("daily point (chunked)"):
        _ = da.chunk(time=1).sel(x=299965, y=2653947, method="nearest").compute()

    subset = da.sel(
        x=slice(100_000, 400_000),
        y=slice(2_600_000, 2_800_000),
    )
    with measure("daily spatial subset isel(time=1)"):
        _ = subset.isel(time=1).load()

    # --- monthly composite ---
    # max_concurrent_reads is lowered here because each reprojected array for
    # the full extent is ~20 MB (int16, 4333×2367 px). The default of 32 would
    # put ~650 MB in-flight per band task; with 3 bands running in parallel via
    # dask that is ~2 GB just for in-flight reads. 8 keeps it under ~500 MB.
    da_monthly = await lazycogs.open_async(
        str(items_parquet),
        crs=dst_crs,
        bbox=dst_bbox,
        resolution=300,
        time_period="P1M",
        bands=["red", "green", "blue"],
        dtype="int16",
        sortby="eo:cloud_cover",
        filter="eo:cloud_cover < 50",
        max_concurrent_reads=8,
    )
    print(f"\nmonthly array: {da_monthly}")

    # Add spatial chunks so dask breaks the full extent into smaller tasks.
    # Without them each task holds the full 4333×2367 array in memory.
    with measure("monthly full extent (chunked time=1, band=1, x=1024, y=1024)"):
        _ = da_monthly.chunk(time=1, band=1, x=1024, y=1024).compute()

    with measure("monthly spatial subset"):
        monthly_subset = da_monthly.sel(
            x=slice(100_000, 400_000),
            y=slice(2_600_000, 2_800_000),
        )
        _ = monthly_subset.load()


if __name__ == "__main__":
    asyncio.run(run())
