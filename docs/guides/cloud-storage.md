# Cloud storage

lazycogs uses [obstore](https://developmentseed.org/obstore/latest/) as its default way to read COG assets from cloud object storage. It can also accept any custom store object that satisfies the async range-read contract consumed by `async-geotiff`. This guide covers the default obstore path plus the custom-store contract.

## Default behavior

By default, `lazycogs.open()` parses each asset HREF into an `ObjectStore` using [`obstore.store.from_url`](https://developmentseed.org/obstore/latest/api/store/from_url/). No credential defaults are applied; the store uses obstore's own environment-based credential discovery (environment variables, instance metadata, config files, etc.).

`lazycogs.open()` runs a lightweight storage smoketest on startup: it resolves the store for a sample data asset and calls `GeoTIFF.open(..., store=...)` to confirm access through the same contract used by the real reader. If the store cannot reach the asset, a `RuntimeError` is raised immediately with a clear message rather than deferring the failure to the first chunk read.

For public buckets that do not require signed requests, pass `skip_signature=True` when constructing the store. For authenticated buckets, provide credentials via environment variables or a pre-configured store.

## Custom store contract

When you pass `store=` explicitly, lazycogs forwards that object to `async-geotiff`. The object does not need to be an obstore `ObjectStore`; it only needs to satisfy the obspec-compatible async range-read contract accepted by `GeoTIFF.open()`.

For most users, obstore is still the recommended path because `store=None` auto-resolves it for each asset HREF and `lazycogs.store_for()` constructs it for you.

## Constructing a store from your data

`lazycogs.store_for()` inspects a geoparquet file and builds a matching `ObjectStore` automatically. It reads one sample item, derives the store root from a data asset HREF, and infers `region` and `requester_pays` from [STAC Storage Extension](https://github.com/stac-extensions/storage) metadata when present.

```python
# Public S3 bucket — pass skip_signature=True for anonymous access
store = lazycogs.store_for("items.parquet", skip_signature=True)
da = lazycogs.open("items.parquet", ..., store=store)

# Authenticated bucket — credentials from environment (no extra kwargs needed)
store = lazycogs.store_for("items.parquet")
da = lazycogs.open("items.parquet", ..., store=store)

# Inspect a specific asset rather than the first data asset
store = lazycogs.store_for("items.parquet", asset="B04")
```

## Constructing stores manually

For authenticated buckets, requester-pays buckets, custom endpoints, or non-standard authentication, construct an obstore-backed store yourself:

```python
from obstore.store import S3Store, GCSStore, HTTPStore

# Public S3 bucket — unsigned requests
store = S3Store(bucket="sentinel-cogs", region="us-west-2", skip_signature=True)
da = lazycogs.open("items.parquet", ..., store=store)

# Authenticated S3 (credentials from env or boto3 chain)
store = S3Store(bucket="my-private-bucket", region="us-west-2")
da = lazycogs.open("items.parquet", ..., store=store)

# Requester-pays S3
store = S3Store(bucket="usgs-landsat", region="us-west-2", request_payer=True)

# Signed HTTPS (e.g. a SAS-token URL from a STAC API)
store = HTTPStore.from_url("https://myaccount.blob.core.windows.net/container?sv=...")

# GCS with a service-account key
store = GCSStore(bucket="my-bucket", service_account_path="/path/to/key.json")
```

See the [obstore store docs](https://developmentseed.org/obstore/latest/api/store/) for the full list of constructors and options.

## Overriding path extraction

By default, lazycogs extracts the object path from an asset HREF by stripping the `scheme://netloc` prefix. This works for standard S3 and GCS URLs but may not match the root of a custom store — for example, when an Azure Blob Storage store is rooted at a container but the HREFs include the container name in the path.

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

`store_for()` and `path_from_href` are most useful together — for example, a public S3 dataset where the asset paths don't align with the store root:

```python
store = lazycogs.store_for("items.parquet")
da = lazycogs.open("items.parquet", ..., store=store, path_from_href=my_path_fn)
```

## NASA Earthdata

NASA Earthdata supports in-region direct S3 access through a temporary-credential endpoint. Use the **synchronous** `NasaEarthdataCredentialProvider` when constructing the store:

```python
from obstore.auth.earthdata import NasaEarthdataCredentialProvider
from obstore.store import S3Store

cp = NasaEarthdataCredentialProvider(
    "https://data.ornldaac.earthdata.nasa.gov/s3credentials"
)
store = S3Store(bucket="ornl-cumulus-prod-protected", region="us-west-2", credential_provider=cp)
da = lazycogs.open("items.parquet", ..., store=store)
```

`NasaEarthdataAsyncCredentialProvider` is not supported. It creates an `aiohttp` session bound to the event loop active when it is constructed. lazycogs runs each chunk read in a short-lived event loop (and in Jupyter, in a separate thread with its own loop), so the session ends up attached to the wrong loop and raises a runtime error. The synchronous provider uses `requests`, which is event-loop-agnostic and works correctly in all contexts.

See [NASA HLS S3 example notebook](../notebooks/hls-s3.ipynb) for a full worked example with direct S3 access.

See also: [API reference for open()](../api/open.md), [API reference for store_for()](../api/utils.md), [STAC item queries guide](stac-search.md)
