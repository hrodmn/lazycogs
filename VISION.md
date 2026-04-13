# Vision: stac-gti-xarray Without GDAL

## Context

stac-gti-xarray currently delegates nearly everything — spatial indexing, temporal filtering, COG I/O, reprojection, mosaicking, and lazy dataset construction — to GDAL's GTI driver via a single `rioxarray.open_rasterio()` call. The problem is that the GTI driver with OGR Parquet/Arrow support is not available in standard PyPI rasterio wheels. It requires a custom GDAL source build (CMake flags, GDAL master branch, libproj, libcurl), a bespoke Docker image, and ongoing maintenance as upstream GDAL changes. This is a lot of infrastructure tax for what is ultimately a fairly narrow I/O pattern.

The goal: replace GDAL with a composable set of modern, Rust-backed libraries that can be distributed as standard Python wheels, delivering the same `(time, band, y, x)` lazy DataArray API.

## What GDAL Does Today (and What Replaces It)

| GDAL operation | Replacement |
|---|---|
| OGR Parquet/Arrow reads geoparquet tile index | `rustac` + DuckDB — already used for the initial STAC query; extend to per-chunk bbox queries |
| OGR SQL `FILTER` for temporal slicing | `rustac.search_sync(..., datetime=day_str, use_duckdb=True)` |
| Spatial intersection (which COGs overlap this chunk's bbox?) | DuckDB spatial query via `rustac.search_sync(..., use_duckdb=True)` — runs inside each task |
| COG header reads + tile I/O | `async-geotiff` (`GeoTIFF.open()` + `geotiff.read(window=...)`) |
| On-the-fly reprojection | `pyproj.Transformer.transform()` (vectorized) + numpy index sampling |
| Overview selection | Extracted from `rio-tiler`'s `AsyncReader._get_overview_level()` logic — pure Python math on `geotiff.overviews` |
| Mosaicking / pixel selection | Mosaic method classes from `rio_tiler/mosaic/methods/` — pure numpy, zero GDAL dependency |
| Dask lazy dataset construction | `xarray.backends.BackendEntrypoint` + `LazilyIndexedArray`; no dask graph until `.chunk()` |

Spatial queries are handled entirely by DuckDB (via rustac). No pyogrio, no geopandas.

## New Pipeline Architecture

### Phase 0 — Open (instant, no pixel I/O, no dask graph)

```
open(href, collections, datetime, bbox, crs, resolution, bands, chunks, ...)
  |
  +--> rustac.search_to(parquet_path, href, bbox=bbox_4326, datetime=full_range)
  |       writes items.parquet to tempdir  [~100-500ms network call]
  |
  +--> _discover_bands(parquet_path)     [unchanged; uses rustac.read_sync()]
  |
  +--> _parse_datetime_range(datetime)   [unchanged; returns list[date]]
  |
  +--> _compute_output_grid(bbox, crs, resolution)
  |       returns (dst_affine, dst_width, dst_height, x_coords, y_coords)
  |       pure math, no I/O
  |
  +--> For each band:
  |       Create StacBackendArray(
  |           parquet_path=...,  # serializable path, not items list
  |           band=band,
  |           dates=dates,
  |           dst_affine=dst_affine,
  |           dst_crs=dst_crs,
  |           bbox_4326=bbox_4326,
  |           sort_by=sort_by,
  |           dst_width=dst_width,
  |           dst_height=dst_height,
  |           shape=(n_dates, dst_height, dst_width),
  |           dtype=...,
  |       )
  |       wrap in LazilyIndexedArray -> xr.Variable(("time", "y", "x"), lazy)
  |
  +--> Assemble xr.Dataset; convert to xr.DataArray with coords
  |
  +--> if chunks: return da.chunk(chunks)  else: return da  [lazy, no dask graph yet]
  |
  --> returns xr.DataArray(time, band, y, x)  — fully lazy, minimal work done
```

### Phase 1 — Chunk materialization (runs inside a dask worker when a chunk is computed)

`StacBackendArray.__getitem__` receives the index key for the specific chunk being computed.

```
StacBackendArray.__getitem__(key)
  |
  +--> Parse key -> (time_idx, y_slice, x_slice)
  |
  +--> chunk_affine = dst_affine * Affine.translation(x_slice.start, y_slice.start)
  |    chunk_bbox_4326 = Transformer(dst_crs -> EPSG:4326).transform(chunk_corners)
  |
  +--> items = rustac.search_sync(
  |        parquet_path,
  |        bbox=chunk_bbox_4326,      # only COGs overlapping THIS chunk
  |        datetime=day_range(dates[time_idx]),
  |        use_duckdb=True,
  |        sort_by=sort_by,
  |    )
  |    # Fast local DuckDB query; [] means no coverage -> return nodata array immediately
  |
  +--> asyncio.run(_async_mosaic_chunk(
           items, band, chunk_affine, dst_crs, chunk_w, chunk_h, nodata, ...
       ))
  |
  +--> _async_mosaic_chunk internals:
  |       for each item href (concurrently via asyncio.gather):
  |           store = store_from_href(href)
  |           geotiff = await GeoTIFF.open(href, store=store)
  |           overview = _select_overview(geotiff, target_res)
  |           window = _compute_native_window(geotiff, chunk_bbox_in_native_crs)
  |           raster = await overview.read(window=window)  # RasterArray (bands, h, w)
  |           warped = _reproject_array(
  |               data=raster.data,
  |               src_affine=raster.transform,
  |               src_crs=raster.crs,
  |               dst_affine=chunk_affine,
  |               dst_crs=dst_crs,
  |               dst_w=chunk_w, dst_h=chunk_h,
  |               nodata=nodata,
  |           )
  |           # _reproject_array: meshgrid dst pixel centers in dst_crs,
  |           # Transformer(dst_crs->src_crs).transform(xs.ravel(), ys.ravel())  [1 vectorized call],
  |           # ~src_affine * (src_xs, src_ys) -> pixel indices,
  |           # numpy fancy indexing for nearest-neighbor sample
  |           mosaic_method.feed(as_masked(warped, nodata))
  |
  --> return mosaic_method.data  -> np.ndarray (chunk_h, chunk_w)
```

## Lazy Evaluation Strategy

**The problem with eager task graph construction (stackstac / odc-stac pattern):**

With `da.from_delayed` + `da.block()`, the full dask task graph is built at `open()` time — one task per chunk for the entire query bbox × time × band. For a large mosaic, this can be thousands of task nodes in memory before the user has expressed any spatial selection. Pruning happens when dask schedules, not when the user slices. The graph overhead is real and this is exactly the "unwieldy task graph" problem in stackstac.

**The right approach: xarray `BackendEntrypoint` + `LazilyIndexedArray`**

xarray has a first-class protocol for this. When a dataset is opened via a backend, each variable is wrapped in `xarray.core.indexing.LazilyIndexedArray` — a thin lazy wrapper with no dask tasks. Tasks are only created when the user calls `.chunk(...)`, and only for the region they have already selected.

The lifecycle looks like:

```python
da = stac_gti_xarray.open(...)           # LazilyIndexedArray: no dask graph, instant
subset = da.sel(x=slice(100, 200), ...)  # pure slice tracking, free
arr = subset.chunk({"x": 2048, "y": 2048})  # dask graph built HERE, only for subset region
arr.compute()                            # tasks run, COGs fetched
```

Contrast with the eager approach: `da.block()` at open time builds all ~6,900 task nodes even if the user only ever touches 4 chunks.

**Implementation via `BackendEntrypoint`:**

```python
class StacBackendArray(xarray.backends.common.BackendArray):
    """One instance per band — shape is (n_time, dst_h, dst_w)."""

    dtype: np.dtype
    shape: tuple[int, ...]  # (n_time, dst_h, dst_w)

    def __getitem__(self, key: xarray.core.indexing.ExplicitIndexer) -> np.ndarray:
        # Called inside a dask task for the specific chunk being computed.
        # 1. Convert indexing key to (time_idx, y_slice, x_slice)
        # 2. Derive chunk bbox from dst_affine + y_slice/x_slice
        # 3. Query items for that (date, chunk_bbox) from parquet via DuckDB
        # 4. asyncio.run(_async_mosaic_chunk(...))
        # 5. Return np.ndarray
        return xarray.core.indexing.explicit_indexing_adapter(
            key, self.shape, xarray.core.indexing.IndexingSupport.BASIC, self._raw_getitem
        )
```

The `BackendEntrypoint.open_dataset()` creates one `StacBackendArray` per band, wraps it in `LazilyIndexedArray`, and assembles the `xr.Dataset`. The `open()` function wraps this as `xr.DataArray`.

An important consequence: **the DuckDB spatial query for which COGs to read runs inside each dask task** (at compute time), not at open time. You only query for the COGs needed by the chunk being computed. The parquet file path is serialized into each task; DuckDB reads it when the task runs.

**What the `chunks` parameter means in this design:**

When `chunks=` is passed to `open()`, we call `.chunk(chunks)` on the result before returning — so the user gets a dask-backed DataArray as expected. If `chunks=None`, the user gets a non-chunked lazy DataArray they can slice and then chunk themselves.

## New Modules Required

| Module | ~Lines | Role |
|---|---|---|
| `_store.py` | 80 | Parse S3/GCS/HTTPS HREFs → obstore `Store` instances |
| `_grid.py` | 50 | `(bbox, crs, resolution)` → `(Affine, width, height, x_coords, y_coords)` — pure math |
| `_reproject.py` | 100 | `_reproject_array()`: pyproj `Transformer` + numpy fancy indexing; optional grid approximation |
| `_backend.py` | 200 | `StacBackendArray` + `StacGtiBackendEntrypoint` — xarray backend protocol |
| `_chunk_reader.py` | 200 | Async mosaic logic: `GeoTIFF.open` → windowed read → reproject → mosaic per chunk |
| `_mosaic_methods.py` | 316 | Copied from rio-tiler `methods/base.py` + `methods/defaults.py` (MIT, pure numpy) |
| `_core.py` (refactored) | ~200 | `open()` entry point: calls rustac, discovers bands, builds output grid, returns DataArray |

Estimated ~1,150 lines total of new/changed Python.

## Spatial Queries: DuckDB via rustac Only

No geopandas, no pyogrio. Spatial intersection happens inside each dask task:

1. `StacBackendArray.__getitem__` receives the exact `(y_slice, x_slice)` for the chunk being computed.
2. The chunk's bbox is derived from `dst_affine`, then transformed to EPSG:4326 using pyproj.
3. `rustac.search_sync(parquet_path, bbox=chunk_bbox_4326, datetime=day_range, use_duckdb=True)` returns only items whose geometries intersect that specific chunk's footprint. This is a fast local DuckDB query against the parquet file in the temp directory.
4. If the result is empty, return a nodata array immediately — no COG I/O.

The parquet file path is what gets serialized into the `StacBackendArray` at open time, not the item list itself. Each task re-queries DuckDB for its own chunk. This keeps open-time work minimal and ensures each task only fetches the COGs it actually needs.

## New Dependencies

```toml
[project.dependencies]
# Remove:
# rasterio   <- pyproj replaces the only remaining use (bbox CRS conversion)
# rioxarray  <- replaced by BackendEntrypoint + LazilyIndexedArray

# Add:
"async-geotiff>=0.4"  # COG I/O (Rust, no GDAL)
"obstore"             # Cloud storage for async-geotiff
"pyproj>=3.3"         # CRS transforms: reprojection, bbox conversion, spatial intersection

# Unchanged:
"dask>=2025"
"xarray>=2025"
"rustac>=0.9.8"
```

No Rust-based warp library required — reprojection is pure pyproj + numpy.

## Performance Expectations

**Likely faster:**
- Async I/O with request coalescing (`async-tiff`'s `fetch_tiles()` batches HTTP range requests in Rust)
- Multi-COG parallel I/O within a chunk (`asyncio.gather` across all overlapping COGs vs. GTI's sequential reads)
- Rust tile decoding (DEFLATE, LZW, ZSTD) happens off the GIL in async-tiff's thread pool
- Per-chunk spatial filtering uses a local DuckDB query — no Python geometry library needed
- Open time is nearly instant — one STAC API call + band discovery, no pre-built spatial index or task graph

**Likely slower (and why):**
- **PROJ call density during reprojection** — GDAL's `WarpedVRT` uses an approximation grid (~64px spacing) and interpolates between PROJ-computed control points, making ~1,024 PROJ calls per 2048×2048 chunk. `_reproject_array()` calls `Transformer.transform()` on the full (4M,) flattened pixel grid in one vectorized C call — no Python loops — but PROJ still processes 4M coordinate pairs instead of ~1,024. Benchmarking on typical UTM → Albers reprojections will determine if this is acceptable. A grid approximation can be added: sample every Nth pixel, call PROJ on the ~1,024 samples, bilinearly interpolate the warp map for the rest — this mirrors GDAL's approach using only pyproj + numpy.
- **DuckDB query per task** — each chunk task runs one DuckDB query against the local parquet. These are millisecond-level queries, but for large task graphs (hundreds of chunks across many bands and dates) the aggregate overhead could matter. Mitigable: reuse a thread-local DuckDB connection rather than opening a new one per task.
- **No persistent COG header cache** — each `GeoTIFF.open()` fetches the file header fresh. GDAL's VSI layer caches this globally. For workflows that open the same COG from multiple chunks, this means repeated header range requests.
- **asyncio/obstore connection pool** — `asyncio.run()` per dask task means no HTTP connection pool reuse across chunk tasks. Mitigable with a thread-local store cache keyed by bucket + region.

## Risks

**async-geotiff partial band reads (medium risk)** — Reading a single band from a multi-band COG currently reads all bands. Some Sentinel-2 products use multi-band COGs, making this a real I/O efficiency concern until the issue is resolved upstream.

**STAC items missing `proj:` extension fields (medium risk)** — The pipeline needs each COG's native CRS to compute the native pixel window to read. This comes from the COG header (`GeoTIFF.open()`), which adds one header range request per COG per chunk. Many modern STAC catalogs include `proj:epsg` and `proj:transform` in item properties; using those when present eliminates the extra round trip.

**asyncio/dask threading (low risk, known solution)** — `asyncio.run()` inside a dask threaded-scheduler task is safe (each thread gets its own event loop) but loses obstore connection pool sharing. A thread-local `dict[str, Store]` cache keyed by bucket + region resolves this.

## What rio-tiler Contributes (specifically)

Only two files, copied verbatim (MIT license, zero GDAL imports):
- `rio_tiler/mosaic/methods/base.py` — `MosaicMethodBase` abstract class (~56 lines)
- `rio_tiler/mosaic/methods/defaults.py` — `FirstMethod`, `HighestMethod`, `LowestMethod`, `MeanMethod`, `MedianMethod`, `StdevMethod`, `CountMethod` (~260 lines)

These are pure numpy operations on `numpy.ma.MaskedArray`. Do not add rio-tiler as a dependency — it brings rasterio (and therefore GDAL) with it. The mosaic methods are the only thing worth taking.

The `AsyncReader._get_overview_level()` logic (~30 lines of pure Python math on `GeoTIFF.overviews`) is also worth extracting for proper overview selection.

## Verification Plan

1. Unit test `_compute_output_grid()` with known bbox/crs/resolution inputs; compare affine and coordinate arrays to rasterio-computed equivalents.
2. Unit test `StacBackendArray.__getitem__` with a mocked DuckDB response; assert correct chunk bbox derivation and nodata short-circuit for empty results.
3. Integration test `_async_mosaic_chunk()` with a single known COG (use existing test fixtures); compare output pixel values to a rasterio-warped reference.
4. End-to-end test `open()` over the existing Sentinel-2 cassette fixtures; assert DataArray shape, CRS, transform, and a sample of pixel values match the GDAL-backed baseline.
5. Benchmark: time a full `.compute()` for the Sentinel-2 test case against the GDAL baseline; flag if reprojection throughput is more than 2x slower.
