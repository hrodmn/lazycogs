<p align="center">
  <img src="https://raw.githubusercontent.com/developmentseed/lazycogs/main/lazycogs.png" alt="lazycogs">
</p>

[![CI](https://github.com/developmentseed/lazycogs/actions/workflows/ci.yml/badge.svg)](https://github.com/developmentseed/lazycogs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lazycogs)](https://pypi.org/project/lazycogs/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lazycogs)](https://pypi.org/project/lazycogs/)
[![License](https://img.shields.io/github/license/developmentseed/lazycogs)](https://github.com/developmentseed/lazycogs/blob/main/LICENSE)

Open a lazy `(band, time, y, x)` xarray DataArray from thousands of cloud-optimized geotiffs (COGs). No GDAL required.

## What is lazycogs?

[stackstac](https://stackstac.readthedocs.io) and [odc-stac](https://odc-stac.readthedocs.io) established the pattern that lazycogs builds on: take a STAC item collection and expose it as a spatially-aligned xarray DataArray ready for dask-parallel computation. Both are excellent tools that cover most satellite imagery workflows well. They rely on the trusty combination of rasterio and GDAL for data i/o and warping operations.

lazycogs takes the same approach but replaces GDAL and rasterio with a Rust-native stack: [rustac](https://stac-utils.github.io/rustac-py) for STAC queries over stac-geoparquet files, [async-geotiff](https://developmentseed/async-geotiff) for COG i/o, and [obstore](https://developmentseed.org/obstore) as the default cloud storage integration.

The result is a tool that can instantly expose a lazy xarray DataArray view of massive STAC item archives in any CRS and resolution. Each array operation triggers a targeted spatial query on the stac-geoparquet file to find only the assets needed for that specific chunk — no upfront scan of every item required.

One constraint worth naming: lazycogs only reads Cloud Optimized GeoTIFFs. If your assets are in another format, this is not the right tool.

| Task | Library |
|---|---|
| STAC search + spatial indexing | `rustac` (DuckDB + geoparquet) |
| COG I/O | `async-geotiff` (Rust, no GDAL) |
| Cloud storage | `obstore` by default; any `async_geotiff.Store` when passed via `store=` |
| Reprojection | `pyproj` + numpy |
| Lazy dataset construction | xarray `BackendEntrypoint` + `LazilyIndexedArray` |

## Installation

```bash
pip install lazycogs
```

## Coordinate convention

`lazycogs.open()` returns a DataArray whose `y` coordinates follow the standard
north-up raster convention with the origin in the top left (not bottom left). That is, `y` coordinates are **descending** from north to south.  In other words,
y label `0` is the northernmost pixel and `y[-1]` is the southernmost.  This
matches the affine transform and is consistent with `odc-stac`, `rioxarray`, and
GDAL.

Use ``sel(y=slice(north, south))`` (high to low) for spatial subsetting.

`x` and `y` keep their `RasterIndex`-based spatial selection behavior, but the
coordinate variables themselves are materialized eagerly so chunked nearest-neighbor
spatial selections compute cleanly.

## Example

```python
import rustac
import lazycogs
from pyproj import Transformer

# set a target CRS and extent
dst_crs = "EPSG:32615"
dst_bbox = (380000.0, 4928000.0, 420000.0, 4984000.0)

# transform to 4326 for STAC search
transformer = Transformer.from_crs(dst_crs, "epsg:4326", always_xy=True)
bbox_4326 = transformer.transform_bounds(*dst_bbox)

# Search a STAC API and cache results to a local stac-geoparquet file.
await rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-08-31",
    bbox=bbox_4326,
)

# Open a fully lazy (band, time, y, x) DataArray. No pixel data is read yet.
# lazycogs does perform a small storage-access smoketest here so auth or
# object-store misconfiguration fails early instead of on the first chunk read.
da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
)
```

### Async loading

When you are already inside an async context (for example, a Jupyter
notebook running on an asyncio loop), you can trigger chunk reads
without blocking the event loop:

```python
# Fetch data asynchronously and load into memory in-place.
subset = await da.isel(x=slice(0, 10), y=slice(0, 10), time=slice(0, 10)).load_async()
```

`load_async` uses xarray's async protocol, which dispatches through
`MultiBandStacBackendArray.async_getitem` and stays on the caller's
event loop. Multiple concurrent chunk reads overlap naturally, so the
async path can be faster than the synchronous `da.compute()` when
reading many chunks inside an already-running loop.

## Custom stores

`lazycogs.open(..., store=...)` accepts any store object that satisfies `async_geotiff.Store`.
For most users, the recommended path is still obstore: leave `store=None` to auto-resolve per-asset stores, or call `lazycogs.store_for()` to build one explicitly.

### Concurrency notes

- Sync callers submit work to one shared persistent lazycogs event loop.
- CPU-bound reprojection runs on one bounded shared thread pool. Set
  `LAZYCOGS_REPROJECT_WORKERS` before first use to change the default
  `min(os.cpu_count() or 1, 4)` limit. The value is read when the pool is
  first created; changes after that are ignored for the life of the process.
- DuckDB queries yield the event loop by running on a small explicit
  executor instead of on the loop thread. On the local benchmark fixture,
  DuckDB stayed under 2% of per-date chunk wall time, so there is no
  separate per-thread DuckDB client pool today.
- If you need to construct a loop-bound resource for lazycogs internals,
  use `lazycogs.run_on_loop(...)`.
- Low-level callers should use `await lazycogs.read_chunk_async(...)`.

## Documentation

- [Home](https://developmentseed.github.io/lazycogs/) — quickstart and full usage guide
- [Example: Midwest US daily Sentinel-2 array](https://developmentseed.org/lazycogs/notebooks/demo_midwest_daily/)
- [Example: Southwest US monthly low-cloud Sentinel-2 array](https://developmentseed.org/lazycogs/notebooks/demo_southwest_monthly/)
- [Architecture](https://developmentseed.org/lazycogs/architecture/)
- [Contributing](CONTRIBUTING.md)
