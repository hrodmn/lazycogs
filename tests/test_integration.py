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
logging.getLogger("lazycogs").setLevel("DEBUG")


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
    with Path("/proc/self/status").open() as f:
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
        f"delta={rss_after - rss_before:+.0f}MB",
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
    store = lazycogs.store_for(str(items_parquet), skip_signature=True)
    da = lazycogs.open(
        str(items_parquet),
        crs=dst_crs,
        bbox=dst_bbox,
        resolution=100,
        time_period="P1D",
        bands=["red", "green", "blue"],
        dtype="int16",
        store=store,
    )
    print(f"\ndaily array: {da}")

    with measure("daily point (chunked)"):
        _ = da.chunk(time=1).sel(x=299965, y=2653947, method="nearest").compute()

    subset = da.sel(
        x=slice(100_000, 400_000),
        y=slice(2_800_000, 2_600_000),
    )
    with measure("daily spatial subset isel(time=1)"):
        _ = subset.isel(time=1).load()


if __name__ == "__main__":
    asyncio.run(run())
