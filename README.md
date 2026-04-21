![lazycogs](./lazycogs.svg)

Open a lazy `(time, band, y, x)` xarray DataArray from thousands of cloud-optimized geotiffs (COGs). No GDAL required.

## Introduction

[stackstac](https://stackstac.readthedocs.io) and [odc-stac](https://odc-stac.readthedocs.io) established the pattern that lazycogs builds on: take a STAC item collection and expose it as a spatially-aligned xarray DataArray ready for dask-parallel computation. Both are excellent tools that cover most satellite imagery workflows well: time series extraction, ML training data, mosaic compositing. These tools rely on the trusty combination of rasterio and GDAL for data i/o and warping operations.

lazycogs builds off of the approach used by stackstac and odc-stac but instead of relying on GDAL and rasterio, lazycogs uses [rustac](https://stac-utils.github.io/rustac-py) to query stac-geoparquet files for determining the assets required for any array operation and [async-geotiff](https://developmentseed/async-geotiff) + [obstore](https://developmentseed.org/obstore) for raster i/o. 

This structure enables you to instantly materialize a lazy xarray DataArray view of massive STAC item archives in any CRS and resolution.

Subsequent queries of the DataArray will perform targeted queries on the stac-geoparquet file for the specified spatial/temporal area of interest to determine which underlying assets need to be accessed for the array operation.

One trade-off worth naming up front: lazycogs only reads Cloud Optimized GeoTIFFs. If your assets are in another format, lazycogs is not the right tool!

Here is a summary of the tool/approach that lazycogs uses for each phase:

| Task | Library |
|---|---|
| STAC search + spatial indexing | `rustac` (DuckDB + geoparquet) |
| COG I/O | `async-geotiff` (Rust, no GDAL) |
| Cloud storage | `obstore` |
| Reprojection | `pyproj` + numpy |
| Lazy dataset construction | xarray `BackendEntrypoint` + `LazilyIndexedArray` |

## Installation

Not yet published to PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/hrodmn/lazycogs.git
```

## Quickstart

```python
import rustac
import lazycogs

# Search a STAC API and write results to a local geoparquet file.
# rustac is async-first, so search_to requires await.
await rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-08-31",
    bbox=[-93.5, 44.5, -93.0, 45.0],
)

# Open the parquet file as a lazy (time, band, y, x) DataArray.
# open() works in scripts and Jupyter notebooks alike — no await needed.
da = lazycogs.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
)
# No COGs have been read yet

# Use time_period="P1W" to composite items within each ISO calendar week.
# The default FirstMethod fills each pixel from the first item with a valid
# (non-nodata) value, skipping remaining items in the week once all pixels
# are filled. This is more efficient than post-hoc ffill or reductions over
# a daily array, which would materialise every time step before reducing.
da_weekly = lazycogs.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    time_period="P1W",
)
```

If you are already inside an async function, use `open_async` to avoid the background thread overhead:

```python
da = await lazycogs.open_async(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
)
```

## Inspecting read plans

Before computing an array, you can ask what DuckDB queries and COG reads would
fire without touching any pixel data. The `da.lazycogs.explain()` method runs
the same spatial queries as `.compute()` but stops before any I/O:

```python
da = await lazycogs.open_async(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    chunks={"time": 1, "x": 512, "y": 512},
)

# Inspect without reading pixels
plan = da.lazycogs.explain()
print(plan.summary())

# Explain a specific slice
plan_subset = da.isel(time=0).lazycogs.explain()

# Convert to a DataFrame for analysis
df = plan.to_dataframe()
df.groupby("band")["n_cog_reads"].describe()

# Fetch COG headers to see which overview level and pixel window would be read
plan_full = da.lazycogs.explain(fetch_headers=True)
print(plan_full.summary())  # shows overview level distribution and avg window size

# Inspect per-item overview and window details
df_full = plan_full.to_dataframe()
df_full[["item_id", "band", "overview_level", "overview_resolution", "window_width", "window_height"]]
```

The `ExplainPlan` returned shows how many items are matched per chunk, the
distribution of items-per-chunk (useful for spotting over-lapping scene edges),
and the empty-chunk fraction (useful for diagnosing sparse time series).

## Custom object stores

By default, `lazycogs.open()` parses each asset HREF into an obstore `ObjectStore` using [`obstore.store.from_url`](https://developmentseed.org/obstore/latest/api/store/from_url/). Native cloud schemes (`s3://`, `s3a://`, `gs://`) default to unsigned requests so public buckets work without credentials.

For anything else — authenticated buckets, signed URLs, request-payer buckets, custom endpoints, MinIO, Cloudflare R2 with an API token, etc. — construct the store yourself and pass it via `store=`. Only the path portion of each HREF is then used to locate objects; the store must be rooted at the same `scheme://netloc` the HREFs point to.

```python
from obstore.store import S3Store, GCSStore, HTTPStore

# Authenticated S3 (credentials from env or boto3 chain)
store = S3Store(bucket="my-private-bucket", region="us-west-2")
da = lazycogs.open("items.parquet", ..., store=store)

# Requester-pays S3
store = S3Store(bucket="usgs-landsat", region="us-west-2", request_payer=True)

# Signed HTTPS (e.g. a SAS-token URL issued by a STAC API)
store = HTTPStore.from_url("https://myaccount.blob.core.windows.net/container?sv=...")

# GCS with a service-account key
store = GCSStore(bucket="my-bucket", service_account_path="/path/to/key.json")
```

See the [obstore store docs](https://developmentseed.org/obstore/latest/api/store/) for the full set of constructors and options.

### Constructing a store from your data

`lazycogs.store_for()` inspects a geoparquet file and builds a matching `ObjectStore` automatically. It reads one sample item, derives the store root from a data asset HREF, and applies the same `skip_signature=True` default used by auto-resolution. If the item contains [STAC Storage Extension](https://github.com/stac-extensions/storage) metadata (v1.0.0 or v2.0.0), `region` and `requester_pays` are also inferred.

```python
# Public S3 bucket — skip_signature=True applied automatically
store = lazycogs.store_for("items.parquet")
da = lazycogs.open("items.parquet", ..., store=store)

# Override any inferred value — caller kwargs always win
store = lazycogs.store_for("items.parquet", skip_signature=False)

# Inspect a specific asset rather than the first data asset
store = lazycogs.store_for("items.parquet", asset="B04")
```

This is most useful when you also need `path_from_href=` — for example, a public S3 dataset where the asset paths don't align with the store root:

```python
store = lazycogs.store_for("items.parquet")
da = lazycogs.open("items.parquet", ..., store=store, path_from_href=my_path_fn)
```

### Overriding path extraction

By default, `lazycogs` extracts the object path from an asset HREF by stripping the `scheme://netloc` prefix. This works for standard S3 and GCS URLs but may not match the root of your custom store — for example, when an Azure Blob Storage store is rooted at a container but the HREFs include the container name in the path.

Use `path_from_href=` to supply a callable that takes the full HREF and returns the object path your store expects:

```python
from urllib.parse import urlparse
from obstore.store import S3Store

store = S3Store(bucket="lp-prod-protected", ...)

def strip_bucket(href: str) -> str:
    # href: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/path/to/file.tif
    # store is rooted at the bucket, so the path is just path/to/file.tif
    return urlparse(href).path.lstrip("/").removeprefix("lp-prod-protected/")

da = lazycogs.open("items.parquet", ..., store=store, path_from_href=strip_bucket)
```

### NASA Earthdata

NASA Earthdata supports in-region direct S3 access through a temporary-credential endpoint. Use the **synchronous** `NasaEarthdataCredentialProvider` (not the async variant) when constructing the store:

```python
from obstore.auth.earthdata import NasaEarthdataCredentialProvider
from obstore.store import S3Store

cp = NasaEarthdataCredentialProvider(
    "https://data.ornldaac.earthdata.nasa.gov/s3credentials"
)
store = S3Store(bucket="ornl-cumulus-prod-protected", region="us-west-2", credential_provider=cp)
da = lazycogs.open("items.parquet", ..., store=store)
```

`NasaEarthdataAsyncCredentialProvider` is not supported. It creates an `aiohttp` session at construction time that is bound to the event loop active when it is created. lazycogs runs each chunk read in a short-lived event loop (and in Jupyter, in a separate thread with its own loop), so the session ends up attached to the wrong loop and raises a runtime error. The synchronous provider uses `requests`, which is event-loop-agnostic and works correctly in all contexts.

## Hive-partitioned STAC datasets

By default, lazycogs creates a plain `DuckdbClient()` and queries a single geoparquet
file. If your STAC items are stored as a **hive-partitioned parquet directory** (e.g.
`year=2023/month=01/...`) you can pass a pre-configured client with
`use_hive_partitioning=True` to enable partition pruning:

```python
from rustac import DuckdbClient
import lazycogs

client = DuckdbClient(use_hive_partitioning=True)

da = lazycogs.open(
    "s3://bucket/stac/",            # directory, not a single file
    duckdb_client=client,
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
)
```

DuckDB skips partition directories that cannot match the spatial/temporal filters,
which can dramatically reduce the number of parquet files scanned on large archives.

You can pass any `DuckdbClient` constructor options (`extension_directory`,
`extensions`, `install_extensions`) using the same approach. When `duckdb_client` is
`None` (the default), lazycogs behaves exactly as before.

## Tuning concurrency

lazycogs uses two independent concurrency controls:

**`max_concurrent_reads`** (passed to `open()`, default 32) limits how many COG files are opened and read simultaneously within a single chunk. This is pure async I/O — it does not create threads. Lower it if you want to reduce peak memory per chunk or are hitting S3 request-rate limits.

**`set_reproject_workers`** controls how many threads each chunk's event loop uses for CPU-bound reprojection (pyproj + numpy). The default is `min(os.cpu_count(), 4)`. Reprojection is memory-bandwidth-bound rather than compute-bound — benchmarks show diminishing returns above 4 threads because concurrent large-array operations saturate the memory bus rather than adding throughput. Raising this beyond 4 is rarely useful.

Each chunk gets its own independent thread pool (not a shared global pool), so dask tasks do not queue behind each other for reprojection.

When using dask, total concurrent COG reads across all workers equals `dask_workers × max_concurrent_reads`. On a 16-core machine with default dask worker count (16) and `max_concurrent_reads=32`, that is 512 simultaneous reads. If you hit S3 throttling or memory pressure, reduce `max_concurrent_reads` at `open()` time.

For better throughput, add time parallelism via dask rather than raising reprojection workers:

```python
# parallelize across time steps — each step gets its own full event loop + thread pool
da = lazycogs.open("items.parquet", ..., chunks={"time": 1})
da.compute()
```

## Documentation

- [Demo notebook](https://hrodmn.github.io/lazycogs/demo/)
- [Architecture](https://hrodmn.github.io/lazycogs/architecture/)
- [API Reference](https://hrodmn.github.io/lazycogs/api/)
