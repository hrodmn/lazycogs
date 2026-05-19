"""xarray BackendArray implementation for lazy STAC COG access."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from xarray.backends.common import BackendArray
from xarray.core import indexing

from lazycogs._chunk_reader import read_chunk_async
from lazycogs._cql2 import _extract_filter_fields, _sortby_fields
from lazycogs._executor import run_duckdb, run_on_loop

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable

    from async_geotiff import Store
    from rustac import DuckdbClient

    from lazycogs._mosaic_methods import MosaicMethodBase


@dataclass(frozen=True)
class _ChunkReadPlan:
    """Everything needed to materialise one chunk across all its time steps.

    Built once in ``_async_getitem`` and passed through to
    ``_read_chunk_all_dates`` and ``_run_one_date``. Frozen to make the
    read-only intent explicit.

    Note: ``warp_cache`` is a mutable dict despite the frozen dataclass. This
    is intentional — concurrent writes from ``asyncio.gather`` coroutines are
    safe because ``compute_warp_map`` is deterministic (a duplicate write
    simply overwrites an identical value).

    Attributes:
        duckdb_client: ``DuckdbClient`` instance used for STAC queries.
        parquet_path: Path to the geoparquet file or hive-partitioned directory.
        sortby: Optional sort keys forwarded to ``client.search``.
        filter_expr: Optional CQL2 filter forwarded to ``client.search``.
        ids: Optional STAC item IDs forwarded to ``client.search``.
        filter_fields: Field names extracted from ``filter_expr``.
        dates: Full list of acquisition date strings.
        chunk_bbox_4326: ``[minx, miny, maxx, maxy]`` in EPSG:4326.
        selected_bands: STAC asset keys to read.
        chunk_affine: Affine transform of the chunk.
        dst_crs: CRS of the output grid.
        chunk_width: Chunk width in pixels.
        chunk_height: Chunk height in pixels.
        nodata: No-data fill value, or ``None``.
        mosaic_method_cls: Mosaic method class, or ``None`` for the default.
        store: Pre-configured :class:`async_geotiff.Store` accepted by
            ``GeoTIFF.open``, or ``None``.
        max_concurrent_reads: Maximum concurrent COG reads per chunk.
        warp_cache: Shared warp map cache across time steps.
        path_fn: Optional callable extracting an object path from an asset HREF.

    """

    duckdb_client: DuckdbClient
    parquet_path: str
    sortby: str | list[str | dict[str, str]] | None
    filter_expr: str | dict[str, Any] | None
    ids: list[str] | None
    filter_fields: set[str]
    dates: list[str]
    chunk_bbox_4326: list[float]
    selected_bands: list[str]
    chunk_affine: Affine
    dst_crs: CRS
    chunk_width: int
    chunk_height: int
    nodata: float | None
    mosaic_method_cls: type[MosaicMethodBase] | None
    store: Store | None
    max_concurrent_reads: int
    warp_cache: dict
    path_fn: Callable[[str], str] | None


@dataclass
class _SpatialWindow:
    """Resolved spatial indexing window.

    Attributes:
        chunk_affine: Affine transform of the chunk (top-left origin).
        chunk_bbox_4326: ``[minx, miny, maxx, maxy]`` in EPSG:4326.
        chunk_height: Chunk height in pixels.
        chunk_width: Chunk width in pixels.
        x_start: First x pixel in the destination grid.
        squeeze_y: Whether the y dimension should be squeezed on return.
        squeeze_x: Whether the x dimension should be squeezed on return.

    """

    chunk_affine: Affine
    chunk_bbox_4326: list[float]
    chunk_height: int
    chunk_width: int
    x_start: int
    squeeze_y: bool
    squeeze_x: bool


def _resolve_time_indices(
    time_key: int | np.integer | slice,
    n_dates: int,
) -> tuple[list[int], bool]:
    """Resolve a time indexer to a list of integer indices.

    Args:
        time_key: Integer or slice indexer for the time dimension.
        n_dates: Total number of dates (size of the time dimension).

    Returns:
        ``(time_indices, squeeze_time)`` where ``squeeze_time`` is ``True``
        when ``time_key`` was a scalar integer.

    """
    if isinstance(time_key, (int, np.integer)):
        return [int(time_key)], True
    start = time_key.start if time_key.start is not None else 0
    stop = time_key.stop if time_key.stop is not None else n_dates
    step = time_key.step if time_key.step is not None else 1
    return list(range(start, stop, step)), False


def _resolve_band_indices(
    band_key: int | np.integer | slice,
    n_bands: int,
) -> tuple[list[int], bool]:
    """Resolve a band indexer to a list of integer indices.

    Args:
        band_key: Integer or slice indexer for the band dimension.
        n_bands: Total number of bands.

    Returns:
        ``(band_indices, squeeze_band)`` where ``squeeze_band`` is ``True``
        when ``band_key`` was a scalar integer.

    """
    if isinstance(band_key, (int, np.integer)):
        return [int(band_key)], True
    start = band_key.start if band_key.start is not None else 0
    stop = band_key.stop if band_key.stop is not None else n_bands
    step = band_key.step if band_key.step is not None else 1
    return list(range(start, stop, step)), False


def _search_items_sync(
    plan: _ChunkReadPlan,
    date: str,
) -> list[Any]:
    """Query the STAC parquet for items overlapping a chunk.

    Args:
        plan: Read plan carrying all parameters for this chunk.
        date: Acquisition date string (``"YYYY-MM-DD"``).

    Returns:
        List of STAC items returned by DuckDB.

    """
    label = f"bands={plan.selected_bands!r}"
    t0 = time.perf_counter()
    items = plan.duckdb_client.search(
        plan.parquet_path,
        bbox=plan.chunk_bbox_4326,
        datetime=date,
        sortby=plan.sortby,
        filter=plan.filter_expr,
        ids=plan.ids,
        include=list(
            {"id", "assets"} | plan.filter_fields | _sortby_fields(plan.sortby),
        ),
    )
    logger.debug(
        "duckdb_client.search %s date=%s returned %d items in %.3fs",
        label,
        date,
        len(items),
        time.perf_counter() - t0,
    )
    if items and logger.isEnabledFor(logging.DEBUG):
        unexpected = {k for k in items[0] if k not in {"id", "assets"}}
        if unexpected:
            logger.debug(
                "duckdb_client.search returned unexpected fields %s — "
                "include filter may not be respected",
                unexpected,
            )
    return items


async def _search_items_async(
    plan: _ChunkReadPlan,
    date: str,
) -> list[Any]:
    """Run the DuckDB search on the dedicated DuckDB executor.

    DuckDB queries serialise on a single connection internally, so this
    yields the event loop during the query but does not produce parallel
    queries against the same DuckdbClient.

    Args:
        plan: Read plan carrying all parameters for this chunk.
        date: Acquisition date string (``"YYYY-MM-DD"``).

    Returns:
        List of STAC items returned by DuckDB.

    """
    return await run_duckdb(_search_items_sync, plan, date)


async def _run_one_date(
    t_idx: int,
    plan: _ChunkReadPlan,
) -> dict[str, np.ndarray] | None:
    """Read and mosaic all COGs for a single time step.

    Issues one DuckDB query for items overlapping the chunk at this date, then
    calls read_chunk_async to fetch and reproject all tiles.
    Returns None if no items match the query.

    Args:
        t_idx: Index into ``plan.dates`` for the time step to read.
        plan: Read plan carrying all parameters for this chunk.

    Returns:
        Per-band arrays keyed by band name, or ``None`` if no items matched.

    """
    date = plan.dates[t_idx]
    items = await _search_items_async(plan, date)

    if not items:
        return None
    t0 = time.perf_counter()
    chunk_result = await read_chunk_async(
        items=items,
        bands=plan.selected_bands,
        chunk_affine=plan.chunk_affine,
        dst_crs=plan.dst_crs,
        chunk_width=plan.chunk_width,
        chunk_height=plan.chunk_height,
        nodata=plan.nodata,
        mosaic_method_cls=plan.mosaic_method_cls,
        store=plan.store,
        max_concurrent_reads=plan.max_concurrent_reads,
        warp_cache=plan.warp_cache,
        path_fn=plan.path_fn,
    )
    logger.debug(
        "read_chunk_async bands=%r date=%s (%d items, %dx%d px) took %.3fs",
        plan.selected_bands,
        date,
        len(items),
        plan.chunk_width,
        plan.chunk_height,
        time.perf_counter() - t0,
    )
    return chunk_result


async def _read_chunk_all_dates(
    time_indices: list[int],
    plan: _ChunkReadPlan,
) -> list[dict[str, np.ndarray] | None]:
    """Run all time steps concurrently inside a single event loop.

    DuckDB queries run on the dedicated DuckDB executor; DuckDB itself
    serialises access on a single connection, so concurrent queries on the
    same ``DuckdbClient`` are safe but not parallel.  Mosaic coroutines for
    all time steps are gathered concurrently so COG reads and reprojections
    overlap across time steps.

    Args:
        time_indices: Ordered list of time-dimension indices to materialise.
        plan: Read plan carrying all parameters for this chunk.

    Returns:
        One entry per time index; ``None`` where no items matched.

    """
    return list(await asyncio.gather(*[_run_one_date(t, plan) for t in time_indices]))


@dataclass
class MultiBandStacBackendArray(BackendArray):
    """Lazy ``(band, time, y, x)`` array for a STAC collection.

    One instance is created at ``open()`` time.  No pixel I/O happens until
    ``__getitem__`` is called inside a dask task.  Reads all selected bands
    together per time step via
    :func:`~lazycogs._chunk_reader.async_mosaic_chunk`, issuing a
    single DuckDB query per time step and sharing reprojection warp maps across
    bands that have identical source geometry.

    Attributes:
        parquet_path: Path to the geoparquet file or hive-partitioned directory
            passed to ``duckdb_client.search``.
        duckdb_client: ``DuckdbClient`` instance used for all STAC queries.
            Constructed with default settings in :func:`open` when not supplied
            by the caller.
        bands: Ordered list of STAC asset keys, one per band.
        dates: Sorted list of unique acquisition date strings
            (``"YYYY-MM-DD"``), one entry per time step.
        dst_affine: Affine transform of the full output grid.
        dst_crs: CRS of the output grid.
        bbox_4326: ``[minx, miny, maxx, maxy]`` in EPSG:4326, used as the
            coarse spatial filter for the initial parquet query.
        sortby: Optional list of ``rustac`` sort keys passed to DuckDB
            queries (e.g. ``["-properties.datetime"]``).
        filter: CQL2 filter expression (text string or JSON dict) forwarded
            to per-chunk DuckDB queries.
        ids: STAC item IDs forwarded to per-chunk DuckDB queries.
        dst_width: Full output grid width in pixels.
        dst_height: Full output grid height in pixels.
        dtype: NumPy dtype of the output array.
        nodata: No-data fill value, or ``None``.
        mosaic_method_cls: Mosaic method class instantiated per chunk, or
            ``None`` to use the default
            :class:`~lazycogs._mosaic_methods.FirstMethod`.
        store: Pre-configured :class:`async_geotiff.Store` accepted by
            ``GeoTIFF.open`` and shared across all chunk reads. When ``None``,
            each asset HREF is resolved to an obstore-backed store via the
            shared process-local cache in :func:`~lazycogs._store.resolve`.
        max_concurrent_reads: Maximum number of COG reads to run concurrently
            per chunk.  Limits peak in-flight memory when a chunk overlaps
            many items.  Defaults to 32.
        path_from_href: Optional callable ``(href: str) -> str`` that extracts
            the object path from an asset HREF.  When provided, it replaces the
            default ``urlparse``-based extraction in
            :func:`~lazycogs._store.resolve`.  Most useful when combined with
            a custom ``store`` whose root does not align with the URL structure
            of the asset HREFs (e.g. Azure Blob Storage with a container-rooted
            store).
        shape: ``(n_bands, n_dates, dst_height, dst_width)``.  Derived from
            the other fields; not accepted as a constructor argument.

    """

    parquet_path: str
    duckdb_client: DuckdbClient
    bands: list[str]
    dates: list[str]
    dst_affine: Affine
    dst_crs: CRS
    bbox_4326: list[float]
    sortby: str | list[str | dict[str, str]] | None
    filter: str | dict[str, Any] | None
    ids: list[str] | None
    dst_width: int
    dst_height: int
    dtype: np.dtype
    nodata: float | None
    mosaic_method_cls: type[MosaicMethodBase] | None = field(default=None)
    store: Store | None = field(default=None)
    max_concurrent_reads: int = field(default=32)
    path_from_href: Callable[[str], str] | None = field(default=None)
    shape: tuple[int, ...] = field(init=False)
    _dst_to_4326: Transformer | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Derive shape and cache the dst→EPSG:4326 transformer."""
        self.shape = (len(self.bands), len(self.dates), self.dst_height, self.dst_width)
        epsg_4326 = CRS.from_epsg(4326)
        if self.dst_crs.equals(epsg_4326):
            self._dst_to_4326: Transformer | None = None
        else:
            self._dst_to_4326 = Transformer.from_crs(
                self.dst_crs,
                epsg_4326,
                always_xy=True,
            )

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return f"MultiBandStacBackendArray(bands={self.bands!r}, shape={self.shape})"

    def _resolve_spatial_window(
        self,
        y_key: int | np.integer | slice,
        x_key: int | np.integer | slice,
    ) -> _SpatialWindow:
        """Resolve spatial indexers to a concrete chunk window.

        Computes the chunk affine transform and EPSG:4326 bounding box from
        the top-down y/x indexers.

        Args:
            y_key: Integer or slice indexer for the y dimension.
            x_key: Integer or slice indexer for the x dimension.

        Returns:
            A :class:`_SpatialWindow` describing the chunk geometry.

        """
        if isinstance(y_key, (int, np.integer)):
            yi = int(y_key)
            y_key = slice(yi, yi + 1)
            squeeze_y = True
        else:
            squeeze_y = False

        if isinstance(x_key, (int, np.integer)):
            xi = int(x_key)
            x_key = slice(xi, xi + 1)
            squeeze_x = True
        else:
            squeeze_x = False

        y_start = y_key.start if y_key.start is not None else 0
        y_stop = y_key.stop if y_key.stop is not None else self.dst_height
        x_start = x_key.start if x_key.start is not None else 0
        x_stop = x_key.stop if x_key.stop is not None else self.dst_width

        chunk_height = y_stop - y_start
        chunk_width = x_stop - x_start

        chunk_affine = self.dst_affine * Affine.translation(x_start, y_start)

        minx = chunk_affine.c
        maxy = chunk_affine.f
        maxx = minx + chunk_width * chunk_affine.a
        miny = maxy + chunk_height * chunk_affine.e  # e < 0

        if self._dst_to_4326 is None:
            chunk_bbox_4326 = [minx, miny, maxx, maxy]
        else:
            xs, ys = self._dst_to_4326.transform(
                [minx, maxx, minx, maxx],
                [maxy, maxy, miny, miny],
            )
            chunk_bbox_4326 = [
                float(min(xs)),
                float(min(ys)),
                float(max(xs)),
                float(max(ys)),
            ]

        return _SpatialWindow(
            chunk_affine=chunk_affine,
            chunk_bbox_4326=chunk_bbox_4326,
            chunk_height=chunk_height,
            chunk_width=chunk_width,
            x_start=x_start,
            squeeze_y=squeeze_y,
            squeeze_x=squeeze_x,
        )

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        """Return the data for the requested index.

        Args:
            key: An xarray ``ExplicitIndexer``.

        Returns:
            A numpy array with shape determined by the indexing key.

        """
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._sync_getitem,
        )

    async def async_getitem(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        """Return the data for the requested index without blocking the event loop.

        This method implements xarray's async indexing protocol, enabling
        callers that already have an active event loop to dispatch chunk
        reads without spawning a background thread.

        Args:
            key: An xarray ``ExplicitIndexer``.

        Returns:
            A numpy array (or array-like) with shape determined by the
            indexing key.

        """
        return await indexing.async_explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._async_getitem,
        )

    def _sync_getitem(self, key: tuple[Any, ...]) -> np.ndarray:
        """Sync adapter that runs ``_async_getitem`` on the background loop.

        Args:
            key: A tuple of ``int | slice`` objects for the
                ``(band, time, y, x)`` dimensions.

        Returns:
            Numpy array with shape determined by the indexing key.

        """
        return run_on_loop(self._async_getitem(key))

    async def _async_getitem(self, key: tuple[Any, ...]) -> np.ndarray:
        """Materialise the chunk identified by ``key``.

        Single source of truth for chunk reads. Reads all selected bands
        together per time step via
        :func:`~lazycogs._chunk_reader.read_chunk_async`, issuing a single
        DuckDB query per time step and sharing reprojection warp maps across
        bands that have identical source geometry.

        Args:
            key: A tuple of ``int | slice`` objects for the
                ``(band, time, y, x)`` dimensions.

        Returns:
            Numpy array with shape determined by the indexing key.

        """
        band_key, time_key, y_key, x_key = key

        band_indices, squeeze_band = _resolve_band_indices(band_key, len(self.bands))
        selected_bands = [self.bands[b] for b in band_indices]

        time_indices, squeeze_time = _resolve_time_indices(time_key, len(self.dates))
        win = self._resolve_spatial_window(y_key, x_key)

        fill = self.nodata if self.nodata is not None else 0

        filter_fields = _extract_filter_fields(self.filter) if self.filter else set()
        plan = _ChunkReadPlan(
            duckdb_client=self.duckdb_client,
            parquet_path=self.parquet_path,
            sortby=self.sortby,
            filter_expr=self.filter,
            ids=self.ids,
            filter_fields=filter_fields,
            dates=self.dates,
            chunk_bbox_4326=win.chunk_bbox_4326,
            selected_bands=selected_bands,
            chunk_affine=win.chunk_affine,
            dst_crs=self.dst_crs,
            chunk_width=win.chunk_width,
            chunk_height=win.chunk_height,
            nodata=self.nodata,
            mosaic_method_cls=self.mosaic_method_cls,
            store=self.store,
            max_concurrent_reads=self.max_concurrent_reads,
            warp_cache={},
            path_fn=self.path_from_href,
        )

        all_chunk_data = await _read_chunk_all_dates(time_indices, plan)

        out_shape = (
            len(band_indices),
            len(time_indices),
            win.chunk_height,
            win.chunk_width,
        )
        result: np.ndarray | None = None

        expected_dims = 3

        for i, chunk_data in enumerate(all_chunk_data):
            if chunk_data is None:
                continue
            if result is None:
                result = np.full(out_shape, fill, dtype=self.dtype)
            for bi, band in enumerate(selected_bands):
                arr = chunk_data[band]
                slice_ = arr[0] if arr.ndim == expected_dims else arr
                result[bi, i] = slice_.astype(self.dtype, copy=False)

        if result is None:
            result = np.full(out_shape, fill, dtype=self.dtype)

        # result shape: (n_selected_bands, n_time, chunk_height, chunk_width)
        # Apply squeezes in axis order: time (axis 1), band (axis 0), y (-2), x (-1).
        out: np.ndarray = result
        if squeeze_time:
            out = out[:, 0, :, :]
        if squeeze_band:
            out = out[0]
        if win.squeeze_y:
            out = np.take(out, 0, axis=-2)
        if win.squeeze_x:
            out = out[..., 0]
        return out
