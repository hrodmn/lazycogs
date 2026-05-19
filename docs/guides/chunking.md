# Chunking and concurrency

lazycogs has two independent concurrency controls: dask chunking for task-level parallelism, and `max_concurrent_reads` for async I/O parallelism within a single chunk.

## Default behavior: no chunks

When `chunks` is `None` (the default), lazycogs returns a `LazilyIndexedArray`-backed DataArray. Each access triggers a targeted read — only the pixels you request are fetched. COG reads across all requested time steps are issued concurrently within a single async event loop, so I/O overlaps even when you read many time steps at once. This is the best mode for:

- Point extraction (`.sel(x=..., y=...)`)
- Small spatial subsets
- Interactive exploration

```python
da = lazycogs.open("items.parquet", bbox=dst_bbox, crs=dst_crs, resolution=10.0)

# COG reads for all time steps in the slice overlap concurrently
vals = da.sel(x=299965, y=2653947, method="nearest").sel(time=slice("2025-06", "2025-08")).compute()
```

## When to add chunks

Add `chunks={"time": 1}` when you want dask to distribute work across multiple workers. Without chunks, all time steps share one lazycogs event loop and one bounded reprojection pool. I/O is concurrent but CPU-bound reprojection is bounded by that shared pool. With temporal chunks, dask can run multiple chunk tasks in parallel across worker threads, all submitting to the same lazycogs loop while still sharing the same bounded reprojection pool:

```python
da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
    chunks={"time": 1},
)
da.max(dim="time").compute()  # each time step runs in its own dask task
```

## Spatial chunks

Avoid spatial chunks unless you are under memory pressure. lazycogs handles spatial I/O concurrency internally through its async event loop — adding spatial dask tasks layers extra DuckDB query overhead on top of I/O that was already happening concurrently.

The one case where spatial chunks are useful is when a single time step is too large to fit in memory even at `max_concurrent_reads=1`. In that case, small spatial chunks limit how many pixels are in flight at once.

## `max_concurrent_reads`

Controls how many COG files are opened and read simultaneously within a single chunk. This is pure async I/O — it does not create threads. The default is 32.

Lower it if you are hitting S3 request-rate throttling or want to reduce peak memory per chunk. Raise it (carefully) if you have many non-overlapping tiles and a fast network connection, but note that diminishing returns set in quickly.

When using dask, total concurrent reads across all workers equals `dask_workers × max_concurrent_reads`. On a 16-core machine with default dask settings and `max_concurrent_reads=32`, that is 512 simultaneous reads.

```python
da = lazycogs.open(
    "items.parquet",
    bbox=dst_bbox,
    crs=dst_crs,
    resolution=10.0,
    chunks={"time": 1},
    max_concurrent_reads=16,   # lower if hitting S3 throttling
)
```

## `LAZYCOGS_REPROJECT_WORKERS`

Controls how many threads the shared reprojection pool uses for CPU-bound reprojection (pyproj + numpy). The default is `min(os.cpu_count(), 4)`.

Reprojection is memory-bandwidth-bound rather than compute-bound. Benchmarks show diminishing returns above 4 threads because concurrent large-array operations saturate the memory bus rather than adding throughput. Raising this beyond 4 is rarely useful.

Set the environment variable before the first lazycogs chunk read:

```bash
export LAZYCOGS_REPROJECT_WORKERS=2
```

See also: [API reference for open()](../api/open.md), [API reference for utilities](../api/utils.md), [Architecture](../architecture.md)
