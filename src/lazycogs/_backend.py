"""xarray BackendArray implementation for lazy STAC COG access."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from rustac import DuckdbClient
from xarray.backends.common import BackendArray
from xarray.core import indexing

from lazycogs._chunk_reader import async_mosaic_chunk_multiband
from lazycogs._cql2 import _extract_filter_fields, _sortby_fields
from lazycogs._executor import get_max_workers
from lazycogs._mosaic_methods import MosaicMethodBase

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_TIME_STEPS = 8


def _run_coroutine(coro: Any) -> Any:
    """Run an async coroutine from sync code.

    Uses ``asyncio.run`` normally, but falls back to a thread-pool worker when
    called from inside a running event loop (e.g. a Jupyter kernel), which does
    not allow re-entrant ``asyncio.run`` calls.

    Each call installs a bounded ``ThreadPoolExecutor`` as the new event loop's
    default executor before running the coroutine.  This caps the number of
    reprojection threads per loop (and therefore per dask task) without sharing
    a single pool across concurrent tasks — each task gets its own independent
    pool, so there is no cross-task queuing.  The executor is automatically shut
    down when ``asyncio.run()`` closes the loop.

    Args:
        coro: The coroutine to execute.

    Returns:
        The return value of the coroutine.

    """

    async def _with_bounded_executor(inner: Any) -> Any:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=get_max_workers(),
                thread_name_prefix="lazycogs-reproject",
            )
        )
        return await inner

    try:
        asyncio.get_running_loop()
        # Already inside a running loop — run in a fresh thread so the new
        # asyncio.run() call gets its own event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, _with_bounded_executor(coro)).result()
    except RuntimeError:
        return asyncio.run(_with_bounded_executor(coro))


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
    time_key: int | np.integer | slice, n_dates: int
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


def _resolve_spatial_window(
    y_key: int | np.integer | slice,
    x_key: int | np.integer | slice,
    dst_height: int,
    dst_width: int,
    dst_affine: Affine,
    dst_crs: CRS,
) -> _SpatialWindow:
    """Resolve spatial indexers to a concrete chunk window.

    Converts logical ascending y indices (south-to-north, as exposed to
    xarray) to physical top-down row offsets (north-to-south, as stored in
    the COG), then computes the chunk affine transform and EPSG:4326 bounding
    box.

    Args:
        y_key: Integer or slice indexer for the y dimension.
        x_key: Integer or slice indexer for the x dimension.
        dst_height: Full output grid height in pixels.
        dst_width: Full output grid width in pixels.
        dst_affine: Affine transform of the full output grid.
        dst_crs: CRS of the output grid.

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

    y_start_logical = y_key.start if y_key.start is not None else 0
    y_stop_logical = y_key.stop if y_key.stop is not None else dst_height
    x_start = x_key.start if x_key.start is not None else 0
    x_stop = x_key.stop if x_key.stop is not None else dst_width

    # logical index 0 = southernmost = physical row (dst_height - 1)
    y_start_physical = dst_height - y_stop_logical
    y_stop_physical = dst_height - y_start_logical

    chunk_height = y_stop_physical - y_start_physical
    chunk_width = x_stop - x_start

    chunk_affine = dst_affine * Affine.translation(x_start, y_start_physical)

    minx = chunk_affine.c
    maxy = chunk_affine.f
    maxx = minx + chunk_width * chunk_affine.a
    miny = maxy + chunk_height * chunk_affine.e  # e < 0

    epsg_4326 = CRS.from_epsg(4326)
    if dst_crs.equals(epsg_4326):
        chunk_bbox_4326 = [minx, miny, maxx, maxy]
    else:
        transformer = Transformer.from_crs(dst_crs, epsg_4326, always_xy=True)
        xs, ys = transformer.transform(
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


def _search_items(
    client: DuckdbClient,
    parquet_path: str,
    bbox: list[float],
    date: str,
    sortby: str | list[str | dict[str, str]] | None,
    filter_expr: str | dict[str, Any] | None,
    ids: list[str] | None,
    filter_fields: set[str],
    label: str,
) -> list[Any]:
    """Query the STAC parquet for items overlapping a chunk.

    Args:
        client: ``DuckdbClient`` instance.
        parquet_path: Path to the geoparquet file or hive-partitioned directory.
        bbox: ``[minx, miny, maxx, maxy]`` bounding box in EPSG:4326.
        date: Acquisition date string (``"YYYY-MM-DD"``).
        sortby: Optional sort keys forwarded to ``client.search``.
        filter_expr: Optional CQL2 filter forwarded to ``client.search``.
        ids: Optional STAC item IDs forwarded to ``client.search``.
        filter_fields: Field names extracted from ``filter_expr`` (added to
            the ``include`` list so DuckDB returns them).
        label: Short identifier used in debug log messages (e.g. the band name
            or a list of band names).

    Returns:
        List of STAC items returned by DuckDB.

    """
    t0 = time.perf_counter()
    items = client.search(
        parquet_path,
        bbox=bbox,
        datetime=date,
        sortby=sortby,
        filter=filter_expr,
        ids=ids,
        include=list(
            {"id", "assets"}.union(filter_fields).union(_sortby_fields(sortby))
        ),
    )
    logger.debug(
        "duckdb_client.search %s date=%s returned %d items in %.3fs",
        label,
        date,
        len(items),
        time.perf_counter() - t0,
    )
    return items


@dataclass
class MultiBandStacBackendArray(BackendArray):
    """Lazy ``(band, time, y, x)`` array for a STAC collection.

    One instance is created at ``open()`` time.  No pixel I/O happens until
    ``__getitem__`` is called inside a dask task.  Reads all selected bands
    together per time step via
    :func:`~lazycogs._chunk_reader.async_mosaic_chunk_multiband`, issuing a
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
        store: Pre-configured obstore ``ObjectStore`` instance shared across
            all chunk reads.  When ``None``, each asset HREF is resolved to a
            store via the thread-local cache in
            :func:`~lazycogs._store.resolve`.
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
    store: Any | None = field(default=None)
    max_concurrent_reads: int = field(default=32)
    path_from_href: Callable[[str], str] | None = field(default=None)
    shape: tuple[int, ...] = field(init=False)
    _duckdb_lock: threading.Lock = field(
        init=False, repr=False, compare=False, default_factory=threading.Lock
    )

    def __post_init__(self) -> None:
        """Derive shape from the other fields."""
        self.shape = (len(self.bands), len(self.dates), self.dst_height, self.dst_width)

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return f"MultiBandStacBackendArray(bands={self.bands!r}, shape={self.shape})"

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
            self._raw_getitem,
        )

    def _raw_getitem(self, key: tuple[Any, ...]) -> np.ndarray:
        """Materialise the chunk identified by ``key``.

        Reads all selected bands together per time step via
        :func:`~lazycogs._chunk_reader.async_mosaic_chunk_multiband`, issuing
        a single DuckDB query per time step and sharing reprojection warp maps
        across bands that have identical source geometry.

        Args:
            key: A tuple of ``int | slice`` objects for the
                ``(band, time, y, x)`` dimensions.

        Returns:
            Numpy array with shape determined by the indexing key.

        """
        band_key, time_key, y_key, x_key = key

        # -- Band dimension --------------------------------------------------
        n_bands = len(self.bands)
        if isinstance(band_key, (int, np.integer)):
            band_indices: list[int] = [int(band_key)]
            squeeze_band = True
        else:
            start = band_key.start if band_key.start is not None else 0
            stop = band_key.stop if band_key.stop is not None else n_bands
            step = band_key.step if band_key.step is not None else 1
            band_indices = list(range(start, stop, step))
            squeeze_band = False

        selected_bands = [self.bands[b] for b in band_indices]

        time_indices, squeeze_time = _resolve_time_indices(time_key, len(self.dates))
        win = _resolve_spatial_window(
            y_key, x_key, self.dst_height, self.dst_width, self.dst_affine, self.dst_crs
        )

        fill = self.nodata if self.nodata is not None else 0
        result = np.full(
            (len(band_indices), len(time_indices), win.chunk_height, win.chunk_width),
            fill,
            dtype=self.dtype,
        )

        # warp_cache is shared across time steps: tiles with the same native
        # CRS and window transform reuse the same WarpMap. Concurrent writes
        # from parallel time-step threads are safe — compute_warp_map is
        # deterministic, so a duplicate write just overwrites an identical value.
        warp_cache: dict = {}

        filter_fields = _extract_filter_fields(self.filter) if self.filter else set()

        def _one_date(t_idx: int) -> dict[str, np.ndarray] | None:
            """Query DuckDB and mosaic one time step; returns None when no items match."""
            date = self.dates[t_idx]
            with self._duckdb_lock:
                items = _search_items(
                    self.duckdb_client,
                    self.parquet_path,
                    win.chunk_bbox_4326,
                    date,
                    self.sortby,
                    self.filter,
                    self.ids,
                    filter_fields,
                    label=f"bands={selected_bands!r}",
                )
            if not items:
                return None
            t0 = time.perf_counter()
            chunk_result = _run_coroutine(
                async_mosaic_chunk_multiband(
                    items=items,
                    bands=selected_bands,
                    chunk_affine=win.chunk_affine,
                    dst_crs=self.dst_crs,
                    chunk_width=win.chunk_width,
                    chunk_height=win.chunk_height,
                    nodata=self.nodata,
                    mosaic_method_cls=self.mosaic_method_cls,
                    store=self.store,
                    max_concurrent_reads=self.max_concurrent_reads,
                    warp_cache=warp_cache,
                    path_fn=self.path_from_href,
                )
            )
            logger.debug(
                "async_mosaic_chunk_multiband bands=%r date=%s (%d items, %dx%d px) took %.3fs",
                selected_bands,
                date,
                len(items),
                win.chunk_width,
                win.chunk_height,
                time.perf_counter() - t0,
            )
            return chunk_result

        if len(time_indices) == 1:
            # Fast path: skip thread pool overhead for the common single-step case
            # (e.g. Dask chunks={"time": 1} or point extractions).
            chunk_data = _one_date(time_indices[0])
            if chunk_data is not None:
                for bi, band in enumerate(selected_bands):
                    arr = chunk_data[band]
                    result[bi, 0] = arr[0] if arr.ndim == 3 else arr
        else:
            n_workers = min(len(time_indices), _MAX_CONCURRENT_TIME_STEPS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                for i, chunk_data in enumerate(pool.map(_one_date, time_indices)):
                    if chunk_data is None:
                        continue
                    for bi, band in enumerate(selected_bands):
                        arr = chunk_data[band]
                        result[bi, i] = arr[0] if arr.ndim == 3 else arr

        # Physical data is top-down; flip to ascending y order for xarray.
        result = result[:, :, ::-1, :]

        # result shape: (n_selected_bands, n_time, chunk_height, chunk_width)
        # Apply squeezes in axis order: time (axis 1), band (axis 0), y (-2), x (-1).
        out: np.ndarray = result  # type: ignore[assignment]
        if squeeze_time:
            out = out[:, 0, :, :]
        if squeeze_band:
            out = out[0]
        if win.squeeze_y:
            out = np.take(out, 0, axis=-2)
        if win.squeeze_x:
            out = out[..., 0]
        return out
