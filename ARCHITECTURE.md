# Architecture: stac-cog-xarray

stac-cog-xarray turns a geoparquet STAC item index into a lazy `(time, band, y, x)` xarray DataArray backed by Cloud-Optimized GeoTIFFs. It requires no GDAL. All raster I/O is done through `async-geotiff` (Rust-backed), spatial queries go through DuckDB via `rustac`, and reprojection is pure `pyproj` + numpy.

## Why parquet, not a STAC API URL

`open()` accepts a path to a local geoparquet file, not a STAC API endpoint. This is intentional. When dask computes a large DataArray, it may execute hundreds of chunk tasks in parallel. If each task queried a remote STAC API directly, that would fire hundreds of concurrent HTTP requests at the API, which is both impolite and likely to trigger rate limiting or outright bans.

Instead, the expected workflow is to run one `rustac.search_to("items.parquet", api_url, ...)` call upfront to download the matching items into a local geoparquet file, and then pass that file to `open()`. Per-chunk spatial filtering is then a fast local DuckDB query against that file, with no network traffic and no API involvement.

## Two-phase execution model

Work is split sharply into two phases.

**Phase 0 — open time** runs at `open()` / `open_async()` call time. It does the minimum needed to build a fully-described lazy DataArray: one DuckDB query to discover bands, one DuckDB query to build the time axis, and a grid computation. No pixel I/O happens. No dask task graph is built. Each band becomes a `StacBackendArray` wrapped in an xarray `LazilyIndexedArray`.

**Phase 1 — compute time** runs inside a dask worker when a chunk is actually computed. `StacBackendArray.__getitem__` receives the exact `(time, y, x)` index for the chunk, derives the chunk's spatial footprint, queries DuckDB for only the COGs that overlap that footprint and time step, reads and reprojects them concurrently with `asyncio`, and mosaics the results.

This split means that `open()` is nearly instant even for large queries, and the DuckDB spatial filter runs once per chunk rather than over the entire bbox at open time.

## Module overview

```
src/stac_cog_xarray/
  _core.py           Entry point. open() / open_async(), band discovery, time-step building.
  _backend.py        StacBackendArray (per-band) and MultiBandStacBackendArray (4-D wrapper) — xarray BackendArray implementations that bridge xarray indexing to chunk reads.
  _chunk_reader.py   Async mosaic logic: open COGs, select overviews, read windows, reproject, mosaic.
  _grid.py           Compute output affine transform and coordinate arrays from bbox + resolution.
  _reproject.py      Nearest-neighbor reprojection using pyproj Transformer + numpy fancy indexing.
  _store.py          Parse cloud HREFs into thread-local obstore Store instances; extract object paths when a store is provided externally.
  _temporal.py       Temporal grouping strategies (day, week, month, year, fixed-day-count).
  _mosaic_methods.py Pixel-selection strategies (First, Highest, Lowest, Mean, Median, Stdev, Count).
```

## Phase 0 in detail

`open_async()` in `_core.py`:

1. Validates that `href` is a `.parquet` or `.geoparquet` file.
2. Parses `time_period` into a `_TemporalGrouper` (see `_temporal.py`).
3. Converts `bbox` from the target CRS to EPSG:4326 using `pyproj.Transformer`.
4. Calls `_discover_bands()`: queries the parquet file via `rustac.search_sync(..., use_duckdb=True, max_items=1)` to find asset keys. Assets with role `"data"` or media type `"image/tiff"` are returned first.
5. Calls `_build_time_steps()`: queries the parquet for all matching items, extracts their datetimes, buckets them with the `_TemporalGrouper`, deduplicates, and returns sorted `(filter_strings, time_coords)` pairs. Only groups with at least one item produce a time step.
6. Calls `compute_output_grid()` to get the output affine transform and x/y coordinate arrays.
7. For each band, creates a `StacBackendArray` (a dataclass) holding all the parameters needed to materialise any chunk later.
8. Wraps all per-band arrays in a single `MultiBandStacBackendArray` with shape `(band, time, y, x)`, then wraps that in one `xarray.core.indexing.LazilyIndexedArray`. This avoids `xr.concat` (used internally by `ds.to_array()`), which would eagerly load `LazilyIndexedArray`-backed objects.
9. Constructs the `xr.DataArray` directly from the 4-D variable. If `chunks` is provided, calls `.chunk(chunks)` to convert to a dask-backed array; otherwise the `LazilyIndexedArray` remains in play so narrow slices (e.g. a single pixel) translate to minimal I/O.

`open()` is a thin synchronous wrapper that calls `asyncio.run(open_async(...))`.

## Phase 1 in detail

`MultiBandStacBackendArray.__getitem__` in `_backend.py` (the 4-D entry point):

1. xarray calls `__getitem__` with an `ExplicitIndexer`. The call is forwarded through `indexing.explicit_indexing_adapter` to `_raw_getitem` with a basic `(int | slice, int | slice, int | slice, int | slice)` key for `(band, time, y, x)`.
2. The band key is resolved to a list of integer band indices. If it was an integer the band dimension is squeezed in the output.
3. For each selected band, `StacBackendArray._raw_getitem((time_key, y_key, x_key))` is called. If the per-band results are not band-squeezed they are stacked with `np.stack(..., axis=0)`.

`StacBackendArray._raw_getitem` in `_backend.py` (per-band):

1. The time key is resolved to a list of integer positions. Integer keys squeeze the time dimension.
2. Integer y or x keys are normalised to size-1 slices; the dimension is squeezed before returning.
3. Logical y-indices (ascending, south-to-north) are converted to physical row indices (descending, north-to-south) to match the affine transform origin.
4. The chunk's affine transform is derived: `chunk_affine = dst_affine * Affine.translation(x_start, y_start_physical)`.
5. The chunk's EPSG:4326 bounding box is computed from the four corners of the chunk using `pyproj.Transformer`.
6. For each time step:
   a. `rustac.search_sync(parquet_path, bbox=chunk_bbox_4326, datetime=date, use_duckdb=True)` returns only items whose geometry intersects this specific chunk for this date. Empty results short-circuit to nodata immediately.
   b. `_run_coroutine(async_mosaic_chunk(...))` materialises the chunk. This helper uses `asyncio.run` normally, but falls back to a `ThreadPoolExecutor` worker when called from inside a running event loop (e.g. a Jupyter kernel) to avoid the "asyncio.run() cannot be called from a running event loop" error.
7. The result array is flipped vertically (`result[:, ::-1, :]`) to restore ascending y-order before squeezing and returning.

`async_mosaic_chunk` in `_chunk_reader.py`:

1. Launches `_read_item_band()` concurrently for all items via `asyncio.gather`.
2. For each item:
   a. If a `store` was supplied by the caller, `path_from_href(href)` extracts just the object path within that store. Otherwise, `store_from_href(href)` returns a thread-local `obstore` store and an object path.
   b. `await GeoTIFF.open(path, store=store)` opens the COG header.
   c. `_select_overview()` picks the finest overview whose pixel size is at least as coarse as the target resolution.
   d. `_native_window()` computes the pixel window in source space that covers the chunk bbox, clamped to the image extent.
   e. `await reader.read(window=window)` fetches the tile data.
   f. `reproject_array()` warps the read data to the destination chunk grid.
3. Each reprojected array is fed to the `MosaicMethodBase` instance. If `mosaic_method.is_done` becomes true (e.g. `FirstMethod` with no nodata pixels remaining), the loop breaks early.
4. Returns `mosaic_method.data` as a `(bands, chunk_height, chunk_width)` array.

## Grid and coordinate convention

`compute_output_grid()` in `_grid.py` produces:

- An affine transform with origin at the top-left corner of the top-left pixel and a negative y-scale (standard north-up raster convention).
- x-coordinates and y-coordinates as 1-D arrays of pixel-centre values, with y ascending (south to north) to match xarray label-based slicing conventions.

The ascending-y / top-down-affine duality is resolved in `_raw_getitem` by converting logical y-indices to physical row indices before computing `chunk_affine`, and flipping the result array before returning.

## Reprojection

`reproject_array()` in `_reproject.py` performs nearest-neighbor reprojection without GDAL:

1. Build a full meshgrid of destination pixel-centre coordinates.
2. Transform all coordinates from `dst_crs` to `src_crs` in one vectorised `Transformer.transform()` call.
3. Apply the inverse source affine transform to convert coordinates to fractional source pixel indices.
4. Use numpy fancy indexing to sample the source array. Out-of-bounds pixels get the nodata fill value.

This processes every destination pixel in a single PROJ call, which is more PROJ calls than GDAL's approximation-grid approach but requires no approximation logic and no external dependencies beyond pyproj and numpy.

## Temporal grouping

`_temporal.py` defines the `_TemporalGrouper` protocol and five concrete implementations:

| Class | `time_period` | Bucket boundary |
|---|---|---|
| `_DayGrouper` | `P1D` | Calendar day |
| `_WeekGrouper` | `P1W` | ISO 8601 calendar week (Monday anchor) |
| `_MonthGrouper` | `P1M` | Calendar month |
| `_YearGrouper` | `P1Y` | Calendar year |
| `_FixedDayGrouper` | `PnD` (n>1) or `PnW` (n>1) | Fixed-length windows aligned to 2000-01-01 |

Each grouper provides three methods: `group_key()` maps a datetime string to a sortable label, `datetime_filter()` converts a label to a `rustac`-compatible datetime range string, and `to_datetime64()` converts a label to a `numpy.datetime64[D]` coordinate value.

## Mosaic methods

`_mosaic_methods.py` contains pure-numpy pixel-selection strategies operating on `numpy.ma.MaskedArray`:

- `FirstMethod`: Returns the first valid pixel seen. Short-circuits when all pixels are filled.
- `HighestMethod` / `LowestMethod`: Tracks the running max / min across items.
- `MeanMethod` / `MedianMethod` / `StdevMethod`: Accumulates all arrays and reduces at the end.
- `CountMethod`: Counts valid (non-masked) observations per pixel.

These are copied from `rio-tiler` (MIT licence, zero GDAL imports) to avoid pulling in rasterio as a transitive dependency.

## Store caching and the `store` parameter

By default, `store_from_href()` in `_store.py` maintains a thread-local `dict[str, Store]` keyed by root URL (`scheme://netloc`). Because dask tasks run in threads, this avoids repeated connection setup within a single task while remaining safe across concurrent tasks.

When the caller supplies a pre-configured `ObjectStore` via the `store` parameter to `open()` / `open_async()`, automatic resolution is skipped entirely. The store is stored on `StacBackendArray` and forwarded through `async_mosaic_chunk` and `_read_item_band` to `GeoTIFF.open()`. Only the object path is extracted from each asset HREF via `path_from_href()`. This path is explicit — it supports custom credentials, non-default endpoints, and private buckets without relying on environment-variable detection.

## Key dependencies

| Package | Role |
|---|---|
| `rustac` | STAC search against local geoparquet files via DuckDB |
| `async-geotiff` | Async COG header reads and windowed tile reads (Rust, no GDAL) |
| `obstore` | Cloud object store abstraction layer for async-geotiff |
| `pyproj` | CRS transforms: bbox reprojection, warp map generation |
| `xarray` | DataArray / Dataset assembly, `BackendArray` / `LazilyIndexedArray` protocol |
| `dask` | Parallel chunk execution |
| `affine` | Affine transform arithmetic |
| `numpy` | Array operations throughout |

## Documentation

The documentation site is built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages via GitHub Actions (`.github/workflows/docs.yml`). It includes:

- **Introduction**: rendered from `README.md` (symlinked as `docs/index.md`)
- **Demo**: the Jupyter notebook at `docs/demo.ipynb`, rendered by `mkdocs-jupyter`
- **API Reference**: auto-generated from docstrings by `mkdocstrings[python]`

To preview locally: `uv run mkdocs serve`
