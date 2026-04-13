"""Entry point for opening a STAC collection as a lazy xarray DataArray."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import rustac
import xarray as xr
from pyproj import CRS, Transformer
from xarray.core import indexing

from stac_cog_xarray._backend import MultiBandStacBackendArray, StacBackendArray
from stac_cog_xarray._grid import compute_output_grid
from stac_cog_xarray._mosaic_methods import FirstMethod, MosaicMethodBase
from stac_cog_xarray._temporal import _TemporalGrouper, grouper_from_period

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from obstore.store import ObjectStore


def _discover_bands(
    parquet_path: str,
    *,
    bbox: list[float] | None = None,
    datetime: str | None = None,
    filter: str | dict[str, Any] | None = None,
    ids: list[str] | None = None,
) -> list[str]:
    """Return the asset keys present in the first matching item of the parquet file.

    Asset keys whose media type contains ``"image/tiff"`` or whose roles
    include ``"data"`` are returned first; all others follow.

    Args:
        parquet_path: Path to a local geoparquet STAC items file.
        bbox: Bounding box ``[minx, miny, maxx, maxy]`` in EPSG:4326 to
            filter the parquet before selecting a representative item.
        datetime: RFC 3339 datetime or range to filter items.
        filter: CQL2 filter expression (text string or JSON dict).
        ids: List of STAC item IDs to restrict results to.

    Returns:
        Ordered list of asset key strings.

    Raises:
        ValueError: If no matching STAC items are found in the parquet file.

    """
    items = rustac.search_sync(
        parquet_path,
        use_duckdb=True,
        max_items=1,
        bbox=bbox,
        datetime=datetime,
        filter=filter,
        ids=ids,
    )
    if not items:
        raise ValueError(f"No STAC items found in {parquet_path!r}")

    assets: dict[str, Any] = items[0].get("assets", {})
    data_bands: list[str] = []
    other_bands: list[str] = []
    for key, asset in assets.items():
        roles = asset.get("roles", [])
        media_type = asset.get("type", "")
        if "data" in roles or "image/tiff" in media_type:
            data_bands.append(key)
        else:
            other_bands.append(key)

    return data_bands if data_bands else other_bands or list(assets)


def _build_time_steps(
    parquet_path: str,
    *,
    bbox: list[float] | None = None,
    datetime: str | None = None,
    filter: str | dict[str, Any] | None = None,
    ids: list[str] | None = None,
    temporal_grouper: _TemporalGrouper,
) -> tuple[list[str], list[np.datetime64]]:
    """Return filter strings and coordinate values for each unique time step.

    Queries *parquet_path* and buckets matching items by *temporal_grouper*.
    Only groups that have at least one matching item produce a time step, so
    the time axis never contains empty slices.

    Args:
        parquet_path: Path to a local geoparquet STAC items file.
        bbox: Bounding box ``[minx, miny, maxx, maxy]`` in EPSG:4326.
        datetime: RFC 3339 datetime or range to pre-filter items.
        filter: CQL2 filter expression (text string or JSON dict).
        ids: List of STAC item IDs to restrict results to.
        temporal_grouper: Grouper that maps item datetimes to group labels,
            ``rustac`` datetime filter strings, and coordinate values.

    Returns:
        A ``(filter_strings, time_coords)`` tuple where *filter_strings* is
        the list of ``rustac``-compatible datetime filter strings (one per
        time step, sorted in temporal order) and *time_coords* is the
        corresponding list of ``numpy.datetime64[D]`` coordinate values.

    """
    items = rustac.search_sync(
        parquet_path,
        use_duckdb=True,
        bbox=bbox,
        datetime=datetime,
        filter=filter,
        ids=ids,
    )
    keys: set[str] = set()
    for item in items:
        props = item.get("properties", {})
        dt: str | None = props.get("datetime") or props.get("start_datetime")
        if dt:
            keys.add(temporal_grouper.group_key(dt))

    sorted_keys = sorted(keys)
    filter_strings = [temporal_grouper.datetime_filter(k) for k in sorted_keys]
    time_coords = [temporal_grouper.to_datetime64(k) for k in sorted_keys]
    return filter_strings, time_coords


def _build_dataarray(
    *,
    parquet_path: str,
    resolved_bands: list[str],
    filter_strings: list[str],
    time_coords: list[np.datetime64],
    bbox: tuple[float, float, float, float],
    bbox_4326: list[float],
    dst_crs: CRS,
    resolution: float,
    sort_by: list[str] | None,
    filter: str | dict[str, Any] | None,
    ids: list[str] | None,
    nodata: float | None,
    out_dtype: np.dtype,
    method_cls: type[MosaicMethodBase],
    chunks: dict[str, int] | None,
    store: ObjectStore | None = None,
) -> xr.DataArray:
    """Assemble the lazy DataArray from pre-computed parameters.

    This is the shared implementation used by both :func:`open` and
    :func:`open_async` after the STAC search completes.

    Args:
        parquet_path: Path to the local geoparquet file.
        resolved_bands: Ordered list of band/asset keys.
        filter_strings: Sorted list of ``rustac``-compatible datetime filter
            strings, one per time step.  Passed directly to
            :class:`~stac_cog_xarray._backend.StacBackendArray`.
        time_coords: ``numpy.datetime64[D]`` coordinate values corresponding
            to each entry in *filter_strings*.
        bbox: Output bounding box in ``dst_crs``.
        bbox_4326: Bounding box in EPSG:4326.
        dst_crs: Target output CRS.
        resolution: Output pixel size in ``dst_crs`` units.
        sort_by: Optional rustac sort keys.
        filter: CQL2 filter expression forwarded to per-chunk DuckDB queries.
        ids: STAC item IDs forwarded to per-chunk DuckDB queries.
        nodata: No-data fill value.
        out_dtype: Output array dtype.
        method_cls: Mosaic method class.
        chunks: Passed to ``DataArray.chunk()`` if not ``None``.
        store: Pre-configured obstore ``ObjectStore`` instance.  When
            provided, it is used directly for all asset reads instead of
            resolving a store from each HREF.

    Returns:
        Lazy ``xr.DataArray`` with dimensions ``(time, band, y, x)``.

    """
    dst_affine, dst_width, dst_height, x_coords, y_coords = compute_output_grid(
        bbox=bbox, crs=dst_crs, resolution=resolution
    )

    band_arrays = [
        StacBackendArray(
            parquet_path=parquet_path,
            band=band,
            dates=filter_strings,
            dst_affine=dst_affine,
            dst_crs=dst_crs,
            bbox_4326=bbox_4326,
            sort_by=sort_by,
            filter=filter,
            ids=ids,
            dst_width=dst_width,
            dst_height=dst_height,
            dtype=out_dtype,
            nodata=nodata,
            shape=(len(filter_strings), dst_height, dst_width),
            mosaic_method_cls=method_cls,
            store=store,
        )
        for band in resolved_bands
    ]

    multi = MultiBandStacBackendArray(
        band_arrays=band_arrays,
        band_names=resolved_bands,
    )
    lazy = indexing.LazilyIndexedArray(multi)
    var = xr.Variable(("band", "time", "y", "x"), lazy)

    # Only convert to dask when the caller explicitly requests chunking.
    # Without this guard, xr.concat (used inside to_array) would eagerly load
    # LazilyIndexedArray objects.  MultiBandStacBackendArray avoids that concat
    # entirely, so LazilyIndexedArray can stay in play: a narrow slice such as
    # da.isel(time=0, x=0, y=0) fetches only the requested pixels.
    if chunks is not None:
        var = var.chunk(chunks)

    time_coord = np.array(time_coords, dtype="datetime64[D]")

    return xr.DataArray(
        var,
        coords={
            "band": resolved_bands,
            "time": time_coord,
            "y": y_coords,
            "x": x_coords,
        },
    )


async def open_async(  # noqa: A001
    href: str,
    *,
    datetime: str | None = None,
    bbox: tuple[float, float, float, float],
    resolution: float,
    crs: str | CRS,
    filter: str | dict[str, Any] | None = None,
    ids: list[str] | None = None,
    bands: list[str] | None = None,
    chunks: dict[str, int] | None = None,
    sort_by: list[str] | None = None,
    nodata: float | None = None,
    dtype: str | np.dtype | None = None,
    mosaic_method: type[MosaicMethodBase] | None = None,
    time_period: str = "P1D",
    store: ObjectStore | None = None,
) -> xr.DataArray:
    """Open a mosaic of STAC items as a lazy ``(time, band, y, x)`` DataArray.

    Async entry point, suitable for use with ``await`` in Jupyter notebooks
    and other async contexts.  For synchronous scripts, use :func:`open`.

    ``href`` must be a path to a local geoparquet file (``.parquet`` or
    ``.geoparquet``).  To search a STAC API first, use
    ``rustac.search_to("items.parquet", api_url, ...)`` and pass the
    resulting file here.

    Phase 0 work (runs at call time):

    1. Query the geoparquet index via DuckDB to discover bands and unique
       time steps (applying ``bbox``, ``datetime``, ``filter``, and ``ids``
       so the time axis contains no empty slices).
    2. Compute the output grid (affine transform + coordinate arrays).
    3. Create one ``StacBackendArray`` per band wrapped in a
       ``LazilyIndexedArray`` -- no pixel I/O yet.
    4. Assemble an ``xr.Dataset``, convert to ``xr.DataArray``, and
       optionally chunk with dask.

    Args:
        href: Path to a local geoparquet file (``.parquet`` or
            ``.geoparquet``).
        datetime: RFC 3339 datetime or range (e.g. ``"2023-01-01/2023-12-31"``)
            used to pre-filter items from the parquet.
        bbox: ``(minx, miny, maxx, maxy)`` in the target ``crs``.
        crs: Target output CRS.
        resolution: Output pixel size in ``crs`` units.
        filter: CQL2 filter expression (text string or JSON dict) forwarded
            to DuckDB queries, e.g. ``"eo:cloud_cover < 20"``.
        ids: STAC item IDs to restrict the search to.
        bands: Asset keys to include.  If ``None``, auto-detected from the
            first matching item.
        chunks: Chunk sizes passed to ``DataArray.chunk()``.  If ``None``
            (default), returns a ``LazilyIndexedArray``-backed DataArray
            where only the requested pixels are fetched on each access —
            ideal for point or small-region queries.  Pass an explicit dict
            to convert to a dask-backed array for parallel computation over
            larger regions.
        sort_by: Sort keys forwarded to ``rustac`` DuckDB queries.
        nodata: No-data fill value for output arrays.
        dtype: Output array dtype.  Defaults to ``float32``.
        mosaic_method: Mosaic method class (not instance) to use.  Defaults
            to :class:`~stac_cog_xarray._mosaic_methods.FirstMethod`.
        time_period: ISO 8601 duration string controlling how items are
            grouped into time steps.  Supported forms: ``PnD`` (days),
            ``P1W`` (ISO calendar week), ``P1M`` (calendar month), ``P1Y``
            (calendar year).  Defaults to ``"P1D"`` (one step per calendar
            day), which preserves the previous behaviour.  Multi-day windows
            such as ``"P16D"`` are aligned to an epoch of 2000-01-01.
        store: Pre-configured obstore ``ObjectStore`` instance to use for all
            asset reads.  Useful when credentials, custom endpoints, or
            non-default options are needed without relying on automatic store
            resolution from each HREF.  When ``None`` (default), each asset
            URL is parsed to create or reuse a per-thread cached store.

    Returns:
        Lazy ``xr.DataArray`` with dimensions ``(time, band, y, x)``.

    Raises:
        ValueError: If ``href`` is not a ``.parquet`` or ``.geoparquet`` file,
            if no matching items are found, or if ``time_period`` is not a
            recognised ISO 8601 duration.

    """
    if not (href.endswith(".parquet") or href.endswith(".geoparquet")):
        raise ValueError(
            f"href must be a .parquet or .geoparquet file path, got: {href!r}. "
            "To search a STAC API, use rustac.search_to() first."
        )

    # Validate time_period early before any I/O so bad values fail fast.
    grouper = grouper_from_period(time_period)

    dst_crs = CRS.from_user_input(crs)

    epsg_4326 = CRS.from_epsg(4326)
    if dst_crs.equals(epsg_4326):
        bbox_4326 = list(bbox)
    else:
        t = Transformer.from_crs(dst_crs, epsg_4326, always_xy=True)
        xs, ys = t.transform(
            [bbox[0], bbox[2], bbox[0], bbox[2]],
            [bbox[1], bbox[1], bbox[3], bbox[3]],
        )
        bbox_4326 = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

    if bands is not None:
        resolved_bands = bands
    else:
        t0 = time.perf_counter()
        resolved_bands = _discover_bands(
            href,
            bbox=bbox_4326,
            datetime=datetime,
            filter=filter,
            ids=ids,
        )
        logger.debug(
            "_discover_bands took %.3fs, found %d bands",
            time.perf_counter() - t0,
            len(resolved_bands),
        )

    t0 = time.perf_counter()
    filter_strings, time_coords = _build_time_steps(
        href,
        bbox=bbox_4326,
        datetime=datetime,
        filter=filter,
        ids=ids,
        temporal_grouper=grouper,
    )
    logger.debug(
        "_build_time_steps took %.3fs, found %d time steps",
        time.perf_counter() - t0,
        len(filter_strings),
    )

    if not filter_strings:
        raise ValueError(
            f"No STAC items matched the query in {href!r} "
            f"(bbox={bbox_4326}, datetime={datetime})."
        )

    logger.info(
        "Discovered %d bands and %d time steps.",
        len(resolved_bands),
        len(filter_strings),
    )

    out_dtype = np.dtype(dtype) if dtype is not None else np.dtype("float32")
    method_cls = mosaic_method if mosaic_method is not None else FirstMethod

    return _build_dataarray(
        parquet_path=href,
        resolved_bands=resolved_bands,
        filter_strings=filter_strings,
        time_coords=time_coords,
        bbox=bbox,
        bbox_4326=bbox_4326,
        dst_crs=dst_crs,
        resolution=resolution,
        sort_by=sort_by,
        filter=filter,
        ids=ids,
        nodata=nodata,
        out_dtype=out_dtype,
        method_cls=method_cls,
        chunks=chunks,
        store=store,
    )


def open(  # noqa: A001
    href: str,
    *,
    datetime: str | None = None,
    bbox: tuple[float, float, float, float],
    crs: str | CRS,
    resolution: float,
    filter: str | dict[str, Any] | None = None,
    ids: list[str] | None = None,
    bands: list[str] | None = None,
    chunks: dict[str, int] | None = None,
    sort_by: list[str] | None = None,
    nodata: float | None = None,
    dtype: str | np.dtype | None = None,
    mosaic_method: type[MosaicMethodBase] | None = None,
    time_period: str = "P1D",
    store: ObjectStore | None = None,
) -> xr.DataArray:
    """Open a mosaic of STAC items as a lazy ``(time, band, y, x)`` DataArray.

    Synchronous entry point.  Calls :func:`open_async` via ``asyncio.run()``.
    Raises ``RuntimeError`` if called from a context with a running event loop
    (e.g. a Jupyter notebook cell).  Use ``await open_async(...)`` there.

    ``href`` must be a path to a geoparquet file (``.parquet`` or
    ``.geoparquet``).  To search a STAC API first, use
    ``rustac.search_to("items.parquet", api_url, ...)`` and pass the
    resulting file here.

    Args:
        href: Path to a local geoparquet file (``.parquet`` or
            ``.geoparquet``).
        datetime: RFC 3339 datetime or range (e.g. ``"2023-01-01/2023-12-31"``)
            used to pre-filter items from the parquet.
        bbox: ``(minx, miny, maxx, maxy)`` in the target ``crs``.
        crs: Target output CRS.
        resolution: Output pixel size in ``crs`` units.
        filter: CQL2 filter expression (text string or JSON dict) forwarded
            to DuckDB queries, e.g. ``"eo:cloud_cover < 20"``.
        ids: STAC item IDs to restrict the search to.
        bands: Asset keys to include.  If ``None``, auto-detected from the
            first matching item.
        chunks: Chunk sizes passed to ``DataArray.chunk()``.  If ``None``
            (default), returns a ``LazilyIndexedArray``-backed DataArray
            where only the requested pixels are fetched on each access —
            ideal for point or small-region queries.  Pass an explicit dict
            to convert to a dask-backed array for parallel computation over
            larger regions.
        sort_by: Sort keys forwarded to ``rustac`` DuckDB queries.
        nodata: No-data fill value for output arrays.
        dtype: Output array dtype.  Defaults to ``float32``.
        mosaic_method: Mosaic method class (not instance) to use.  Defaults
            to :class:`~stac_cog_xarray._mosaic_methods.FirstMethod`.
        time_period: ISO 8601 duration string controlling how items are
            grouped into time steps.  Supported forms: ``PnD`` (days),
            ``P1W`` (ISO calendar week), ``P1M`` (calendar month), ``P1Y``
            (calendar year).  Defaults to ``"P1D"`` (one step per calendar
            day), which preserves the previous behaviour.  Multi-day windows
            such as ``"P16D"`` are aligned to an epoch of 2000-01-01.
        store: Pre-configured obstore ``ObjectStore`` instance to use for all
            asset reads.  Useful when credentials, custom endpoints, or
            non-default options are needed without relying on automatic store
            resolution from each HREF.  When ``None`` (default), each asset
            URL is parsed to create or reuse a per-thread cached store.

    Returns:
        Lazy ``xr.DataArray`` with dimensions ``(time, band, y, x)``.

    Raises:
        ValueError: If ``href`` is not a ``.parquet`` or ``.geoparquet`` file,
            if no matching items are found, or if ``time_period`` is not a
            recognised ISO 8601 duration.

    """
    return asyncio.run(
        open_async(
            href,
            datetime=datetime,
            bbox=bbox,
            crs=crs,
            resolution=resolution,
            filter=filter,
            ids=ids,
            bands=bands,
            chunks=chunks,
            sort_by=sort_by,
            nodata=nodata,
            dtype=dtype,
            mosaic_method=mosaic_method,
            time_period=time_period,
            store=store,
        )
    )
