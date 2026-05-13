# Architecture: lazycogs

lazycogs turns a geoparquet STAC item index into a lazy `(band, time, y, x)` xarray DataArray backed by Cloud-Optimized GeoTIFFs. It requires no GDAL. All raster I/O is done through `async-geotiff` (Rust-backed), spatial queries go through DuckDB via `rustac`, and reprojection now flows through a backend-neutral dispatcher in `_reproject.py`. Today that dispatcher still defaults to the legacy `pyproj` + numpy nearest-neighbor engine while the rust-warp adapter is being integrated.

## Why parquet, not a STAC API URL

`open()` accepts a path to a geoparquet file or directory, not a STAC API endpoint. This is intentional. When dask computes a large DataArray, it may execute hundreds of chunk tasks in parallel. If each task queried a remote STAC API directly, that would fire hundreds of concurrent HTTP requests at the API, which is both impolite and likely to trigger rate limiting or outright bans.

Instead, the expected workflow is to run one `rustac.search_to("items.parquet", api_url, ...)` call upfront to download the matching items into a local geoparquet file, and then pass that file to `open()`. Per-chunk spatial filtering is then a fast DuckDB query, with no network traffic and no API involvement.

For large pre-existing STAC archives stored as hive-partitioned parquet directories (e.g. `year=2023/month=01/...`), pass a `DuckdbClient(use_hive_partitioning=True)` via the `duckdb_client` parameter to enable DuckDB partition pruning. All internal queries in `open()` use `duckdb_client.search()`. When `duckdb_client` is `None`, a plain `DuckdbClient()` is created automatically.

## Two-phase execution model

Work is split sharply into two phases.

**Phase 0 — open time** runs at `open()` call time. It does the minimum needed to build a fully-described lazy DataArray: one DuckDB query to discover bands, one DuckDB query to build the time axis, and a grid computation. No pixel I/O happens. No dask task graph is built. A single `MultiBandStacBackendArray` is wrapped in an xarray `LazilyIndexedArray`.

**Phase 1 — compute time** runs inside a dask worker when a chunk is actually computed. `MultiBandStacBackendArray.__getitem__` receives the exact `(band, time, y, x)` index for the chunk, derives the chunk's spatial footprint, queries DuckDB for only the COGs that overlap that footprint and time step, reads and reprojects all selected bands concurrently with `asyncio`, and mosaics the results.

This split means that `open()` is nearly instant even for large queries, and the DuckDB spatial filter runs once per chunk rather than over the entire bbox at open time.

## Module overview

```
src/lazycogs/
  _core.py           Entry point. open(), band discovery, time-step building.
  _backend.py        MultiBandStacBackendArray — xarray BackendArray implementation that bridges xarray indexing to chunk reads.
  _chunk_reader.py   Async mosaic logic: open COGs, select overviews, read windows, reproject, mosaic.
  _executor.py       Per-chunk reprojection thread pool configuration. Exposes set_reproject_workers() and get_max_workers(); the actual pool is created per event loop in _backend.py.
  _explain.py        Dry-run read estimator. Registers the da.lazycogs.explain() xarray accessor.
  _grid.py           Compute output affine transform and dimensions from bbox + resolution.
  _reproject.py      Backend-neutral reprojection dispatcher; legacy nearest-neighbor backend still uses pyproj Transformer + numpy fancy indexing.
  _storage_ext.py    STAC Storage Extension metadata parsing (version detection, kwargs extraction for v1 and v2).
  _store.py          Resolve cloud HREFs into obstore Store instances (or route through a user-supplied store) with a thread-local cache; store_for() factory for constructing stores from parquet STAC files.
  _temporal.py       Temporal grouping strategies (day, week, month, year, fixed-day-count).
  _mosaic_methods.py Pixel-selection strategies (First, Highest, Lowest, Mean, Median, Stdev, Count).
```

## Phase 0 in detail

`open()` in `_core.py`:

1. Resolves `duckdb_client`: if not provided, creates a plain `DuckdbClient()`. Validates that `href` ends in `.parquet`/`.geoparquet` when no client is supplied (a directory path is accepted when a custom client is passed).
2. Parses `time_period` into a `_TemporalGrouper` (see `_temporal.py`).
3. Converts `bbox` from the target CRS to EPSG:4326 using `pyproj.Transformer`.
4. Calls `_discover_bands()`: queries the parquet source via `duckdb_client.search(..., max_items=1)` to find asset keys. Assets with role `"data"` or media type `"image/tiff"` are returned first.
5. Calls `_smoketest_store()`: fetches one sample item from the parquet, resolves the object store for a representative data asset HREF, and calls `head()` to confirm access. Raises `RuntimeError` immediately with a clear message if the store cannot reach the asset, so misconfiguration is surfaced at `open()` time rather than deferred to the first chunk read.
6. Calls `_build_time_steps()`: queries the parquet source via `duckdb_client.search_to_arrow(...)` to obtain an Arrow table containing only the `datetime` and `start_datetime` columns (plus any filter/sort fields). Extracts timestamps from the Arrow columns without Python-level dict walking, buckets them with the `_TemporalGrouper`, deduplicates, and returns sorted `(filter_strings, time_coords)` pairs. Only groups with at least one item produce a time step.
7. Calls `compute_output_grid()` to get the output affine transform and dimensions (width, height). No eager coordinate arrays are produced.
8. Creates a single `MultiBandStacBackendArray` (a dataclass) with shape `(band, time, y, x)` holding all the parameters needed to materialise any chunk later, then wraps it in one `xarray.core.indexing.LazilyIndexedArray`. This avoids `xr.concat` (used internally by `ds.to_array()`), which would eagerly load `LazilyIndexedArray`-backed objects.
9. Uses `rasterix.RasterIndex` to define the x/y coordinates (lazily).
10. Constructs the `xr.DataArray` directly from the 4-D variable. If `chunks` is provided, calls `.chunk(chunks)` to convert to a dask-backed array; otherwise the `LazilyIndexedArray` remains in play so narrow slices (e.g. a single pixel) translate to minimal I/O.
11. Stores `_stac_backend` (the `MultiBandStacBackendArray` instance) and `_stac_time_coords` (the full time coordinate array) in `da.attrs` so that `da.lazycogs.explain()` can reconstruct the explain plan without re-specifying `open()` parameters.

## Explain: dry-run read estimator

`da.lazycogs.explain()` (registered in `_explain.py` as an xarray accessor) provides an `EXPLAIN ANALYZE`-style dry run. It issues one DuckDB spatial query per `(time step, spatial chunk)` combination — not per band, because the query result is band-independent. Query results are fanned across all active bands in Python, and execution stops before any pixel I/O. All `(time × spatial tile)` queries are dispatched concurrently via `asyncio.gather`; DuckDB itself serialises access on a single connection.

```python
plan = da.lazycogs.explain()          # DuckDB queries only, no pixel reads
plan = da.lazycogs.explain(fetch_headers=True)  # also reads COG IFD headers
print(plan.summary())
df = plan.to_dataframe()
```

The accessor reads `_stac_backend` and `_stac_time_coords` from `da.attrs` and respects the DataArray's current extent and chunk sizes, so explaining a sliced DataArray (`da.isel(time=0).lazycogs.explain()`) queries only the reads needed for that slice.

`ExplainPlan` exposes:
- `total_chunk_reads` — number of `(band, time, spatial tile)` combinations
- `total_cog_reads` — total COG files matched across all chunks
- `empty_chunk_count` — chunks with no matching COG files (useful for diagnosing sparse series)
- `summary()` — multi-line human-readable report with COG-read distribution histogram
- `to_dataframe()` — one row per `(chunk, COG file)` for further analysis in pandas

With `fetch_headers=True`, each matched COG header is fetched (a small HTTP range request to read the IFD block) and `CogRead.overview_level`, `window_col_off`, `window_row_off`, `window_width`, and `window_height` are populated.

## Phase 1 in detail

`MultiBandStacBackendArray.__getitem__` in `_backend.py`:

1. xarray calls `__getitem__` with an `ExplicitIndexer`. The call is forwarded through `indexing.explicit_indexing_adapter` to `_sync_getitem`, which calls `_run_coroutine(self._async_getitem(key))` with a basic `(int | slice, int | slice, int | slice, int | slice)` key for `(band, time, y, x)`.
   When the caller is already inside an async event loop (for example, an interactive map running in Jupyter), xarray can dispatch reads via `async_getitem` instead. `async_getitem` calls `indexing.async_explicit_indexing_adapter` with the same `_async_getitem` method, so the sync and async paths share a single source of truth and produce identical results. The only difference is that `async_getitem` stays on the caller's loop and avoids the background-thread overhead of `_run_coroutine`.
2. The band key is resolved to a list of integer band indices. If it was an integer the band dimension is squeezed in the output.
3. The time key is resolved to a list of integer positions. Integer keys squeeze the time dimension.
4. Integer y or x keys are normalised to size-1 slices; the dimension is squeezed before returning.
5. Physical top-down y-indices are used directly; no conversion is needed.
6. The chunk's affine transform is derived: `chunk_affine = dst_affine * Affine.translation(x_start, y_start_physical)`.
7. The chunk's EPSG:4326 bounding box is computed from the four corners of the chunk using `_dst_to_4326`, a `pyproj.Transformer` cached on the `MultiBandStacBackendArray` instance at construction time (or `None` when `dst_crs` is already EPSG:4326).
8. `_async_getitem` drives all time steps via `await _read_chunk_all_dates(...)`. Inside `_read_chunk_all_dates`, an `asyncio.gather` fans out one `_run_one_date` coroutine per time step, so COG reads overlap across dates:
   a. Each `_run_one_date` calls `await _search_items_async(plan, date)`, which dispatches the DuckDB query to `_DUCKDB_EXECUTOR` via `run_in_executor`. DuckDB serialises access internally, so concurrent queries on the same client are safe but not parallel. Empty results short-circuit to nodata immediately.
   b. `read_chunk_async(...)` materialises all selected bands for the time step concurrently.
9. The result is returned in the same top-down orientation as the COG data.

## Grid and coordinate convention

`compute_output_grid()` in `_grid.py` produces:

- An affine transform with origin at the top-left corner of the top-left pixel and a negative y-scale (standard north-up raster convention).
- x-coordinates and y-coordinates as 1-D arrays of pixel-centre values, with y descending (north to south) to match the standard raster convention used by GDAL, `odc-stac`, and `rioxarray`.

The y coordinates are derived directly from the affine transform: `y[i] = f + e * (0.5 + i)`. Because `e < 0`, `y[0]` is the northernmost pixel centre and the array is strictly decreasing. Spatial selection uses `sel(y=slice(north, south))` (high to low).

A `rasterix.RasterIndex` is attached to every DataArray returned by `open()`. The index replaces explicit x/y coordinate arrays with lazy coordinate generation from the affine transform, provides CRS discoverability via `da.proj.crs`, and enables spatial alignment with other RasterIndex-backed arrays.

## Per-chunk read and resample pipeline

Each call to `_read_item_band()` in `_chunk_reader.py` follows a four-step pipeline to turn a remote COG into a correctly-sized, correctly-projected numpy array for one destination chunk.

### 1. Overview selection

Before fetching any pixels, `_select_overview()` picks the right level of the COG's built-in pyramid. The target resolution is estimated in the source image's native CRS: if `dst_crs` differs from the COG's CRS, a 1-pixel offset at the chunk centre is transformed with pyproj to approximate the pixel size in source units. The overview list (ordered finest → coarsest) is walked to find the coarsest overview whose pixel size is still ≤ the target — i.e. the finest source data that avoids upsampling. This preserves as much spatial detail as the output scale warrants without reading unnecessarily fine data. If no overview satisfies the condition (target falls between native and the finest overview), or no overviews exist, full-resolution is used.

This is the primary resampling control: the pyramid level is chosen to match the output resolution, so the reprojection step works on roughly the right amount of data.

**Why we don't use the STAC projection extension.** The [projection extension](https://github.com/stac-extensions/projection) fields (`proj:epsg`, `proj:transform`, `proj:shape`) could theoretically substitute for the CRS, affine transform, and full-resolution dimensions read from the COG header. But overview structure — the per-level pixel sizes, transforms, and dimensions needed by `_select_overview()` — is not part of any STAC extension. We would still need `GeoTIFF.open()` to read the IFD chain to get that. Beyond overviews, the IFD also holds the tile index (byte offsets per tile per level), which is required to issue the actual range requests; there is no way to skip the header open regardless of how complete the STAC metadata is. Using projection extension fields would add conditional logic and trust questions (not all catalogs populate them correctly) for zero I/O savings.

### 2. Source window computation

The destination chunk's bounding box (in `dst_crs`) is reprojected to the source CRS via pyproj. The inverse of the source affine transform maps those four corners to fractional pixel coordinates. `floor`/`ceil` and clamping to image bounds gives the `Window` that covers exactly the source pixels needed. Only that window is fetched from the COG, minimising I/O.

If the chunk bbox falls entirely outside the source image after clamping, `_native_window()` returns `None` and the item is skipped.

### 3. Tile read

`await reader.read(window=window)` fetches the windowed pixel data from the selected overview level (or full-res). The result is a `(bands, window_h, window_w)` array in the source CRS/grid.

### 4. Reprojection dispatch and current legacy backend

`_chunk_reader.py` now builds a `ReprojectRequest` and calls `reproject_tile()` in `_reproject.py` rather than reaching directly into warp-map helpers. That gives lazycogs a clean seam for swapping reprojection engines while keeping chunk orchestration unchanged.

The dispatcher currently short-circuits exact same-grid reads and otherwise routes to the legacy nearest-neighbor backend, which still warps the source tile onto the destination chunk grid without GDAL:

1. Build a meshgrid of destination pixel-centre coordinates.
2. Transform all coordinates from `dst_crs` to `src_crs` in one vectorised `Transformer.transform()` call.
3. Apply the inverse source affine transform to get fractional source pixel indices.
4. `np.floor` rounds to the nearest-neighbor sample; numpy fancy indexing populates the output array.
5. Out-of-bounds pixels get the nodata fill value.

Nearest-neighbor is still the only active resampling method in production code. The `rust-warp` dependency is present for development and adapter work, but it is not the default engine yet.

## Concurrency model

There are four nested layers of concurrency in a chunk read.

**Dask (chunk level).** When a dask-backed DataArray is computed, dask dispatches each chunk task to a worker thread. Each worker thread calls `_sync_getitem()` in `_backend.py`, which calls `_run_coroutine(_async_getitem(...))` to drive all time steps from the persistent per-thread background loop. Worker threads are independent — they share no state except the thread-local store cache in `_store.py` and the thread-local DuckDB clients in `_backend.py`.

**asyncio (time + item level).** A single event loop call (via `_run_coroutine`) handles the entire chunk. Inside `_async_getitem`, `asyncio.gather` fans out one `_run_one_date` coroutine per time step, so all time steps are in flight concurrently within the same event loop. DuckDB queries run on `_DUCKDB_EXECUTOR` (a dedicated two-thread pool) via `run_in_executor`, yielding the event loop during each query. DuckDB's internal mutex serialises actual DB access, so queries are safe but not parallel on a single `DuckdbClient`. Once a query returns, its mosaic coroutine proceeds immediately and COG reads for all time steps overlap in the event loop's I/O layer. Because all time steps share a single event loop and therefore a single bounded reprojection executor, the reprojection thread count stays at `get_max_workers()` regardless of how many time steps are in the chunk (no thread explosion). The `warp_cache` is shared across coroutines: `compute_warp_map` is deterministic, so concurrent writes are safe.

**asyncio (item level).** Inside a time step's event loop, `read_chunk_async` launches one `_read_item_band()` task per overlapping item up front, with an `asyncio.Semaphore(max_concurrent_reads)` (configurable via `open()`, default 32) capping how many reads run concurrently. Tasks complete in I/O arrival order, but results are buffered by their original list index and drained into the mosaic in source-list order. This preserves the sort contract for `FirstMethod` — items are fed strictly in the order returned by DuckDB (i.e. the caller's `sortby` order) regardless of which COGs arrive first over the network, while all concurrent I/O remains in flight. COG header reads and tile fetches from `async-geotiff` are all awaitable, so the event loop multiplexes them without blocking. Early exit is preserved: once the mosaic method signals completion, remaining tasks are cancelled in a `finally` block, and items still waiting on the semaphore never start.

**Thread pool (CPU work per item).** `_apply_bands_with_warp_cache` is synchronous CPU-bound work that processes all bands for one item together. `_read_item_band` dispatches it via `loop.run_in_executor(None, ...)` — one executor call per item — so the event loop stays free to process other items' tile reads while reprojections run on threads. Because the call is coarse-grained (all bands per item) and GIL-releasing (`pyproj` and numpy both release during heavy inner loops), offloading to the thread pool gives real CPU parallelism without excessive submission overhead.

**Why threads, not a process pool.** `pyproj.Transformer.transform()` and numpy's fancy-indexing both release the GIL during their heavy inner loops. Threads therefore give real CPU parallelism here — not just interleaving — without the overhead of process spawning and array pickling that a `ProcessPoolExecutor` would require.

**Why reprojection is memory-bandwidth-bound, not compute-bound.** `compute_warp_map` builds two meshgrids the size of the output chunk, transforms all coordinates in one vectorised call, and produces large index arrays. `apply_warp_map` samples the source array with random-access fancy indexing (`out[:, valid] = data[:, row_idx[valid], col_idx[valid]]`), which produces near-constant cache misses. Both phases are dominated by memory latency and bandwidth rather than arithmetic. In practice this means CPU utilisation is low (threads stall waiting for memory), and adding more than 4 concurrent reprojection threads provides no throughput benefit — they saturate the memory bus instead.

**Bounded per-loop executor.** Rather than using Python's default `min(32, cpu_count + 4)` thread count, `_get_or_create_background_loop()` installs a bounded `ThreadPoolExecutor` (default `min(os.cpu_count(), 4)`) as the default executor on each background loop it creates, before any coroutines run. This caps thread count per loop while preserving per-loop isolation: each dask task has its own independent pool and does not queue behind other tasks. The executor is shut down when the background loop thread exits. Call `lazycogs.set_reproject_workers(n)` to change the per-loop bound (see `_executor.py`).

**Persistent background loop.** `_run_coroutine()` submits the coroutine to a persistent per-thread background loop via `asyncio.run_coroutine_threadsafe`. The background loop is created by `_get_or_create_background_loop` on the first call from a given thread and reused for all subsequent calls on that same thread. This is the same path whether the caller is a regular Python script, a Jupyter kernel, or a dask worker. The persistent loop is required because `async-geotiff` and `obstore` may fire callbacks after the awaited coroutine returns; a fresh per-call loop would tear down before those callbacks land. Per-thread isolation is preserved: each dask worker thread gets its own independent background loop and executor. One consequence: credential providers that hold event-loop-bound resources (such as `NasaEarthdataAsyncCredentialProvider`, which creates an `aiohttp` session at construction time) fail in this path because the session is bound to a different loop. Use the synchronous credential provider equivalents (e.g. `NasaEarthdataCredentialProvider`) instead.

## Chunking strategy and throughput tradeoffs

The `chunks` argument to `open()` controls whether the returned DataArray is backed by dask. Choosing the wrong chunking strategy — particularly adding spatial chunks — can significantly reduce throughput compared to leaving the array unchunked.

### Why spatial chunks hurt

Without spatial chunks, xarray calls `MultiBandStacBackendArray.__getitem__` once per time step for the full spatial extent. That single call fires one `asyncio.gather` that reads every overlapping COG for that time step concurrently. This is the maximum possible I/O parallelism for a time step: all tile fetches are in flight simultaneously in a single event loop.

With spatial chunks (e.g. `chunks={"x": 512, "y": 512}`), dask splits the extent into N tasks. Each task:

- Runs a separate DuckDB query (via `_search_items_async` on `_DUCKDB_EXECUTOR`) to find overlapping items.
- Fires a smaller `asyncio.gather` over only the COGs that overlap its sub-region.

The total number of COG reads is the same, but they are spread across N smaller gathers rather than one large one. Dask workers do provide some task-level parallelism, but the overhead of N DuckDB queries and N smaller event loop gathers typically outweighs the benefit, especially for small chunk sizes. A COG that spans multiple spatial chunks is also opened once per overlapping chunk rather than once per time step.

The async layer already handles spatial I/O concurrency. Dask spatial chunks add overhead without adding concurrency.

### Where dask helps

**The time dimension.** Without dask, `_async_getitem()` already parallelises all time steps concurrently within a single chunk read via `asyncio.gather`. Dask adds a second level of time parallelism when the array has more time steps than fit in one chunk: `chunks={"time": N}` lets dask run multiple chunks in parallel across worker threads, each chunk running its own internal gather. For most use cases without dask, the built-in time-step parallelism is sufficient and avoids dask scheduling overhead entirely.

**Band dimension chunking does not help.** Within a single time step, all bands are read together by `_read_item_bands`. Splitting bands into separate dask tasks (`chunks={"band": 1}`) creates the same per-task overhead as spatial chunks (separate DuckDB queries, event loop creation, executor instantiation) without a meaningful parallelism benefit. Keep all bands in a single chunk.

**Memory pressure** is the other legitimate reason to add spatial chunks. If the full array does not fit in memory, spatial chunking limits how much is materialised at once even if it costs throughput.

### Recommended defaults

| Goal | Suggested `chunks` |
|---|---|
| Maximum throughput, array fits in memory | omit `chunks` (or `chunks={}`) |
| Array too large to fit in memory | `{"time": 1, "x": N, "y": N}` with N as large as memory allows |

When spatial chunks are necessary for memory reasons, making them as large as possible minimises per-chunk overhead and keeps the `asyncio.gather` fan-out wide. An alternative to adding spatial chunks is reducing `max_concurrent_reads` on `open()`: this limits peak in-flight memory per chunk without the overhead of smaller dask tasks or additional DuckDB queries.

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

## Temporal compositing

Combining `time_period` with a mosaic method is the idiomatic way to produce
temporal composites. Setting `time_period="P1W"` groups every STAC item within
the same ISO calendar week into a single time step. When a chunk is read,
`async_mosaic_chunk` feeds all items for that week to the mosaic method in
order. With `FirstMethod` (the default), reading stops as soon as every output
pixel has a valid (non-nodata) value — the remaining items in the week are
never fetched.

Note that mosaic methods operate on nodata masks only; they have no awareness
of semantic content like clouds. Pixels that are not masked as nodata in the
source COG will be treated as valid regardless of what they represent. If
cloud masking is desired, it must be applied to the source data before
lazycogs reads it (for example, by using scene-level cloud-mask bands
to set nodata at the COG level).

This approach is substantially more efficient than applying temporal reductions
as post-processing steps on a daily array. Operations like `ffill(dim="time")`
or `max(dim="time")` over a dask-backed DataArray force xarray to materialise
every time step before reducing, because each step's result depends on the
previous one (for `ffill`) or all steps at once (for `max`). A weekly composite
opened with `time_period="P1W"` reads at most one week's worth of COGs per
output time step, with early exit when the mosaic is complete.

The general rule: push compositing logic into `time_period` and `mosaic_method`
at `open()` time. Reach for post-hoc xarray reductions only for operations that
cannot be expressed as a per-time-step mosaic (for example, a multi-week
mean that requires all weeks to be present before reducing).

## Store caching and the `store` parameter

`resolve()` in `_store.py` defers to `obstore.store.from_url` for scheme detection — including the special-case HTTPS routing for `amazonaws.com`, `r2.cloudflarestorage.com`, and Azure hosts — rather than maintaining its own list of known object-store domains. The constructed store is cached per thread in a `dict[str, ObjectStore]` keyed by root URL (`scheme://netloc`). Because dask tasks run in threads, this avoids repeated connection setup within a single task while remaining safe across concurrent tasks.

No credential defaults are applied; stores are constructed with obstore's own environment-based credential discovery (environment variables, instance metadata, config files, etc.). For public buckets that do not require signed requests, callers pass `skip_signature=True` explicitly. For authenticated access or any non-default configuration, the caller is expected to construct an `ObjectStore` and pass it via the `store=` parameter to `open()`; `resolve()` then returns it unchanged and only extracts the object path from each HREF. No introspection is done on a user-supplied store — the caller is responsible for ensuring it is rooted at the same `scheme://netloc` the HREFs point to.

`store_for(href, *, asset=None, **kwargs)` is a public convenience factory that automates this construction. It reads one sample item from the geoparquet file, extracts a data asset HREF, and calls `from_url` using obstore's environment-based credential discovery. If the item carries STAC Storage Extension metadata (v1.0.0 flat fields or v2.0.0 `storage:schemes`/`storage:refs`), `region` and `requester_pays` are also extracted and forwarded. Caller `kwargs` override all inferred values (e.g. `skip_signature=True` for public buckets). The returned store is not cached — the caller owns its lifetime and passes it to `open()` via `store=`.

When the store root does not align with the URL structure of the asset HREFs — for example, an Azure Blob Storage store rooted at a container while the HREFs include the container name in the path — the caller can provide a `path_from_href` callable to `open()`. The callable takes the full HREF string and returns the object path to use with the store. When supplied, it replaces the default `urlparse`-based extraction in `resolve()`.

## Key dependencies

| Package | Role |
|---|---|
| `rustac[arrow]` | STAC search against local geoparquet files via DuckDB; Arrow output via `arro3-core` |
| `arro3-core` | Zero-copy Arrow table output from DuckDB queries (installed via `rustac[arrow]`) |
| `async-geotiff` | Async COG header reads and windowed tile reads (Rust, no GDAL) |
| `obstore` | Cloud object store abstraction layer for async-geotiff |
| `rust-warp` | Experimental reprojection backend dependency, currently sourced from GitHub during integration work |
| `pyproj` | CRS transforms: bbox reprojection, target-resolution estimation, legacy warp backend |
| `xarray` | DataArray / Dataset assembly, `BackendArray` / `LazilyIndexedArray` protocol |
| `rasterix` | CRS-aware `RasterIndex` for lazy spatial coordinates |
| `xproj` | CRS accessor and alignment for xarray Flexible Indexes |
| `dask` | Parallel chunk execution |
| `affine` | Affine transform arithmetic |
| `numpy` | Array operations throughout |

## Documentation

The documentation site is built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages via GitHub Actions (`.github/workflows/docs.yml`). It includes:

- **Introduction**: rendered from `README.md` (symlinked as `docs/index.md`)
- **Examples**: two Jupyter notebooks rendered by `mkdocs-jupyter` — `docs/demo_midwest_daily.ipynb` (daily Sentinel-2 array over the Midwest US) and `docs/demo_southwest_monthly.ipynb` (monthly low-cloud composite over the US Southwest)
- **API Reference**: auto-generated from docstrings by `mkdocstrings[python]`

To preview locally: `uv run mkdocs serve`
