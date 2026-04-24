![lazycogs](./lazycogs.svg)

Open a lazy `(time, band, y, x)` xarray DataArray from thousands of cloud-optimized geotiffs (COGs). No GDAL required.

## What is lazycogs?

[stackstac](https://stackstac.readthedocs.io) and [odc-stac](https://odc-stac.readthedocs.io) established the pattern that lazycogs builds on: take a STAC item collection and expose it as a spatially-aligned xarray DataArray ready for dask-parallel computation. Both are excellent tools that cover most satellite imagery workflows well. They rely on the trusty combination of rasterio and GDAL for data i/o and warping operations.

lazycogs takes the same approach but replaces GDAL and rasterio with a Rust-native stack: [rustac](https://stac-utils.github.io/rustac-py) for STAC queries over stac-geoparquet files, [async-geotiff](https://developmentseed/async-geotiff) for COG i/o, and [obstore](https://developmentseed.org/obstore) for cloud storage access.

The result is a tool that can instantly expose a lazy xarray DataArray view of massive STAC item archives in any CRS and resolution. Each array operation triggers a targeted spatial query on the stac-geoparquet file to find only the assets needed for that specific chunk — no upfront scan of every item required.

One constraint worth naming: lazycogs only reads Cloud Optimized GeoTIFFs. If your assets are in another format, this is not the right tool.

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

## Example

```python
import rustac
import lazycogs

# Search a STAC API and cache results to a local stac-geoparquet file.
await rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-08-31",
    bbox=[-93.5, 44.5, -93.0, 45.0],
)

# Open a fully lazy (time, band, y, x) DataArray. No COGs are read yet.
da = lazycogs.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
)
```

## Documentation

- [Home](https://hrodmn.github.io/lazycogs/) — quickstart and full usage guide
- [Demo notebook](https://hrodmn.github.io/lazycogs/demo/)
- [Architecture](https://hrodmn.github.io/lazycogs/architecture/)
- [API Reference](https://hrodmn.github.io/lazycogs/api/)
- [Contributing](CONTRIBUTING.md)
