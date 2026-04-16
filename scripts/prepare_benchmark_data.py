#!/usr/bin/env python3
"""Download a small Sentinel-2 dataset for offline benchmarks.

Queries the Element84 Earth Search STAC API for Sentinel-2 items over western
Colorado, downloads the selected band assets to a local directory, then writes
a new parquet file with hrefs pointing to the local files.

Creates .benchmark_data/ (gitignored) with:
  cogs/{item_id}/{band}.tif   downloaded COG files
  benchmark_items.parquet     parquet index with file:// hrefs

Usage:
    uv run python scripts/prepare_benchmark_data.py
    uv run python scripts/prepare_benchmark_data.py --overwrite
"""

import argparse
import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse

import rustac
from obstore.store import from_url

STAC_HREF = "https://earth-search.aws.element84.com/v1"
COLLECTIONS = ["sentinel-2-c1-l2a"]
# Small region over western Colorado, in EPSG:4326 for STAC query
BBOX_4326 = [-108.5, 37.5, -107.5, 38.5]
DATETIME = "2025-07-02/2025-07-05"
# Red (10m) + Narrow NIR (20m) — sufficient for NDVI benchmarks
BANDS = ["red", "nir08"]
LIMIT = 10

DATA_DIR = Path(__file__).parents[1] / ".benchmark_data"


def _download(href: str, dest: Path) -> None:
    """Download a cloud object to a local file using obstore."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(href)
    root_url = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.lstrip("/")
    kwargs = {"skip_signature": True}
    store = from_url(root_url, **kwargs)
    logging.info("Downloading %s", href)
    result = store.get(path)
    dest.write_bytes(result.bytes())
    logging.info("Wrote %s (%.1f MB)", dest, dest.stat().st_size / 1_048_576)


async def main(overwrite: bool = False) -> None:
    """Run the benchmark data preparation pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    DATA_DIR.mkdir(exist_ok=True)
    cog_dir = DATA_DIR / "cogs"

    raw_parquet = DATA_DIR / "raw_items.parquet"
    if overwrite or not raw_parquet.exists():
        logging.info("Querying STAC API (%s)...", STAC_HREF)
        await rustac.search_to(
            str(raw_parquet),
            href=STAC_HREF,
            collections=COLLECTIONS,
            datetime=DATETIME,
            bbox=BBOX_4326,
            limit=LIMIT,
        )

    items: list[dict] = rustac.search_sync(str(raw_parquet), use_duckdb=True)
    logging.info("Found %d items", len(items))

    local_items = []
    for item in items:
        item_id = item["id"]
        local_assets = {}
        for band in BANDS:
            if band not in item.get("assets", {}):
                logging.warning("Item %s has no asset %r; skipping.", item_id, band)
                continue
            href = item["assets"][band]["href"]
            local_path = cog_dir / item_id / f"{band}.tif"
            if overwrite or not local_path.exists():
                _download(href, local_path)
            local_assets[band] = {
                **item["assets"][band],
                "href": local_path.as_uri(),
            }
        local_items.append({**item, "assets": local_assets})

    out_parquet = DATA_DIR / "benchmark_items.parquet"
    rustac.write_sync(str(out_parquet), local_items)
    logging.info("Wrote benchmark parquet: %s", out_parquet)
    logging.info(
        "Run benchmarks with: uv run pytest tests/benchmarks/ --benchmark-enable"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download assets even if they already exist locally.",
    )
    args = parser.parse_args()
    asyncio.run(main(overwrite=args.overwrite))
