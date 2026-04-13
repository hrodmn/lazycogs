# stac-cog-xarray

Open a geoparquet STAC item collection as a lazy `(time, band, y, x)` xarray DataArray backed by Cloud Optimized GeoTIFFs. No GDAL required.

**[Documentation](https://hrodmn.github.io/stac-cog-xarray/)**

## Why

Most tools that combine STAC and xarray (stackstac, odc-stac, rioxarray's GTI backend) depend on GDAL for spatial indexing, COG I/O, and reprojection. GDAL works, but it introduces a large build-time dependency that is difficult to distribute as a standard wheel and requires custom Docker images for anything beyond basic rasterio functionality.

This package replaces GDAL with a set of modern, Rust-backed libraries that ship as standard Python wheels:

| Task | Library |
|---|---|
| STAC search + spatial indexing | `rustac` (DuckDB + geoparquet) |
| COG I/O | `async-geotiff` (Rust, no GDAL) |
| Cloud storage | `obstore` |
| Reprojection | `pyproj` + numpy |
| Lazy dataset construction | xarray `BackendEntrypoint` + `LazilyIndexedArray` |

## How it works

### Opening a collection

`open()` takes a path to a local geoparquet file containing STAC items (produced by `rustac.search_to()`), not a STAC API URL. This is intentional: when dask computes a large DataArray it may run hundreds of chunk tasks in parallel, and each task needs to query which COGs overlap its spatial footprint. Sending those queries directly to a remote STAC API would fire hundreds of concurrent HTTP requests at it, which is likely to hit rate limits or get you banned. Instead, `rustac.search_to()` downloads the matching items once into a local geoparquet file, and per-chunk spatial filtering is then a fast local DuckDB query against that file with no network traffic.

`open()` does minimal upfront work: band and date discovery from the parquet index, and output grid calculation. No pixel data is read, and no dask task graph is constructed.

```python
import rustac
import stac_cog_xarray

# Step 1: search a STAC API and write results to a local geoparquet file
rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-08-31",
    bbox=[-93.5, 44.5, -93.0, 45.0],
)

# Step 2: open the parquet file as a lazy DataArray
# bbox here is in the target CRS (EPSG:32615)
da = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    bands=["red", "green", "blue", "nir"],
    chunks={"time": 1, "y": 2048, "x": 2048},
)
# da is a lazy (time, band, y, x) DataArray — no COGs have been read yet
```

The geoparquet file path is serialized into each dask task. COG reads happen only when you compute.

### Chunk materialization

When a chunk is computed, `StacBackendArray.__getitem__` runs inside the dask worker:

1. The chunk's bounding box is derived from the output affine transform and the slice indices.
2. `rustac.search_sync` runs a DuckDB spatial query against the local parquet file to find only the COGs that overlap this specific chunk and date. Chunks with no coverage return a nodata array immediately.
3. All overlapping COGs are opened concurrently with `async-geotiff`. The correct overview level is selected automatically based on the target resolution, minimizing data transfer.
4. Each COG tile is reprojected to the destination grid using `pyproj.Transformer` for coordinate transformation and numpy fancy indexing for nearest-neighbor resampling.
5. Reprojected arrays are composited using the selected mosaic method.

The DuckDB query runs per chunk at compute time, not at open time. Only the parquet path is stored in each task node, so the task graph stays small regardless of how many scenes are in the collection.

### Lazy evaluation

xarray's `LazilyIndexedArray` protocol means the dask task graph is built only when you call `.chunk()`, and only for the region you have already selected. Slicing before chunking is free:

```python
# Select a region of interest before building the task graph
subset = da.sel(
    time="2023-07-15",
    x=slice(390000, 410000),
    y=slice(4975000, 4945000),
)
# Now chunk and compute — the task graph covers only the selected region
result = subset.chunk({"y": 512, "x": 512}).compute()
```

This avoids the "unwieldy task graph" problem common to tools that build the full graph at open time.

## Installation

```bash
pip install stac-cog-xarray
```

## Usage

### Basic

```python
import rustac
import stac_cog_xarray

rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-08-31",
    bbox=[-93.5, 44.5, -93.0, 45.0],
)

da = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
)
```

The returned DataArray has dimensions `(time, band, y, x)` with coordinate arrays for each dimension. `x` and `y` are in the requested CRS; `time` is `datetime64[D]`.

### Async contexts (Jupyter notebooks)

In async contexts such as Jupyter notebooks, use `open_async` to avoid a `RuntimeError` from a running event loop:

```python
da = await stac_cog_xarray.open_async(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
)
```

### Filtering items

Pass a CQL2 filter expression to restrict which items are used:

```python
da = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    datetime="2023-06-01/2023-08-31",
    filter="eo:cloud_cover < 20",
)
```

### Custom object store

By default, each asset HREF is parsed to create a per-thread cached obstore `Store`. Pass a pre-configured `ObjectStore` when you need custom credentials, non-default endpoints, or other store options:

```python
from obstore.store import S3Store

store = S3Store.from_env(
    bucket="my-bucket",
    config={"AWS_REGION": "us-west-2", "AWS_ACCESS_KEY_ID": "...", "AWS_SECRET_ACCESS_KEY": "..."},
)

da = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    store=store,
)
```

The provided store is used for all asset reads. The path component of each asset HREF is extracted and used as the object key within that store.

### Mosaic methods

When multiple scenes cover the same pixel, the mosaic method controls how they are composited. The default is `FirstMethod`, which takes the first valid pixel in the sort order.

```python
da = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    mosaic_method=stac_cog_xarray.MedianMethod,
    sort_by=["-properties.datetime"],  # newest first
)
```

Available methods: `FirstMethod`, `HighestMethod`, `LowestMethod`, `MeanMethod`, `MedianMethod`, `StdevMethod`, `CountMethod`.

### Temporal grouping

By default, each unique calendar day in the parquet file becomes one time step (`time_period="P1D"`). Use `time_period` to group items into coarser buckets:

```python
da = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    time_period="P1M",   # one step per calendar month
)
```

Supported ISO 8601 duration forms: `PnD` (n-day windows), `P1W` (ISO calendar week), `P1M` (calendar month), `P1Y` (calendar year). Multi-day windows (e.g. `"P16D"`) are aligned to an epoch of 2000-01-01.

### Chunking

Pass `chunks` to get a dask-backed DataArray directly from `open()`:

```python
da = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    chunks={"time": 1, "y": 2048, "x": 2048},
)
```

Omit `chunks` to get a truly lazy DataArray backed by `LazilyIndexedArray`. Slices like `da.isel(time=0, x=0, y=0)` fetch only the requested pixels — no unnecessary I/O. Call `.chunk(...)` later when you want dask parallelism.

### Output dtype and nodata

```python
da = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    dtype="uint16",
    nodata=0,
)
```

The default output dtype is `float32`.

## API reference

### `open(href, *, bbox, crs, resolution, ...)`

Synchronous entry point. Raises `RuntimeError` if called from a running event loop (e.g. a Jupyter notebook cell) — use `open_async` there instead.

| Parameter | Type | Description |
|---|---|---|
| `href` | `str` | Path to a local `.parquet` or `.geoparquet` file |
| `bbox` | `tuple[float, float, float, float]` | `(minx, miny, maxx, maxy)` in the target `crs` |
| `crs` | `str \| CRS` | Target output CRS |
| `resolution` | `float` | Output pixel size in `crs` units |
| `datetime` | `str \| None` | RFC 3339 datetime or range to pre-filter items |
| `bands` | `list[str] \| None` | Asset keys to include; auto-detected if `None` |
| `filter` | `str \| dict \| None` | CQL2 filter expression forwarded to DuckDB queries |
| `ids` | `list[str] \| None` | STAC item IDs to restrict results to |
| `chunks` | `dict[str, int] \| None` | Chunk sizes for `DataArray.chunk()`; if `None`, returns a `LazilyIndexedArray`-backed DataArray where only requested pixels are fetched |
| `sort_by` | `list[str] \| None` | Sort keys forwarded to rustac DuckDB queries |
| `mosaic_method` | `type[MosaicMethodBase] \| None` | Mosaic method class; defaults to `FirstMethod` |
| `time_period` | `str` | ISO 8601 duration for temporal grouping; defaults to `"P1D"` |
| `dtype` | `str \| dtype \| None` | Output array dtype; defaults to `float32` |
| `nodata` | `float \| None` | No-data fill value |
| `store` | `ObjectStore \| None` | Pre-configured obstore `ObjectStore` for all asset reads; auto-resolved from each HREF when `None` |

Returns a lazy `xr.DataArray` with dimensions `(time, band, y, x)`.

### `open_async(...)`

Same signature as `open()`. Use with `await` in async contexts.

## Dependencies

- [`rustac`](https://github.com/stac-utils/rustac-py): Rust-backed STAC client with DuckDB integration for fast geoparquet queries
- [`async-geotiff`](https://github.com/geospatial-jeff/async-tiff): Async COG reader backed by Rust; handles HTTP range requests, tile decoding, and overview selection
- [`obstore`](https://github.com/developmentseed/obstore): Cloud object store abstraction (S3, GCS, Azure, HTTP) used by async-geotiff
- [`pyproj`](https://pyproj4.github.io/pyproj/): CRS transformations for reprojection and bbox conversion
- [`xarray`](https://xarray.dev/) + [`dask`](https://dask.org/): Lazy dataset construction and parallel computation
