"""xarray BackendArray implementation for lazy STAC COG access."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from xarray.backends.common import BackendArray
from xarray.core import indexing

import rustac

from lazycogs._chunk_reader import async_mosaic_chunk, async_mosaic_chunk_multiband
from lazycogs._executor import get_max_workers
from lazycogs._mosaic_methods import MosaicMethodBase

logger = logging.getLogger(__name__)


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
class StacBackendArray(BackendArray):
    """Lazy array for a single band of a STAC collection.

    One instance is created per band at ``open()`` time.  No pixel I/O
    happens until ``__getitem__`` is called inside a dask task.

    Attributes:
        parquet_path: Path to the local geoparquet file written by
            ``rustac.search_to``.
        band: STAC asset key for the band this array represents.
        dates: Sorted list of unique acquisition date strings
            (``"YYYY-MM-DD"``), one entry per time step.
        dst_affine: Affine transform of the full output grid.
        dst_crs: CRS of the output grid.
        bbox_4326: ``[minx, miny, maxx, maxy]`` in EPSG:4326, used as the
            coarse spatial filter for the initial parquet query.
        sort_by: Optional list of ``rustac`` sort keys passed to DuckDB
            queries (e.g. ``["-properties.datetime"]``).
        filter: CQL2 filter expression (text string or JSON dict) forwarded
            to per-chunk DuckDB queries.
        ids: STAC item IDs forwarded to per-chunk DuckDB queries.
        dst_width: Full output grid width in pixels.
        dst_height: Full output grid height in pixels.
        dtype: NumPy dtype of the output array.
        nodata: No-data fill value, or ``None``.
        shape: ``(n_dates, dst_height, dst_width)``.
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

    """

    parquet_path: str
    band: str
    dates: list[str]
    dst_affine: Affine
    dst_crs: CRS
    bbox_4326: list[float]
    sort_by: list[str] | None
    filter: str | dict[str, Any] | None
    ids: list[str] | None
    dst_width: int
    dst_height: int
    dtype: np.dtype
    nodata: float | None
    shape: tuple[int, ...]
    mosaic_method_cls: type[MosaicMethodBase] | None = field(default=None)
    store: Any | None = field(default=None)
    max_concurrent_reads: int = field(default=32)
    path_from_href: Callable[[str], str] | None = field(default=None)

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return f"StacBackendArray(band={self.band!r}, shape={self.shape})"

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        """Return the data for the requested index.

        Args:
            key: An xarray ``ExplicitIndexer``.

        Returns:
            A numpy array containing the requested chunk.

        """
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_getitem,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_bbox_4326(
        self,
        chunk_affine: Affine,
        chunk_width: int,
        chunk_height: int,
    ) -> list[float]:
        """Return the bounding box of a chunk in EPSG:4326.

        Args:
            chunk_affine: Affine transform of the chunk (top-left origin).
            chunk_width: Chunk width in pixels.
            chunk_height: Chunk height in pixels.

        Returns:
            ``[minx, miny, maxx, maxy]`` in EPSG:4326.

        """
        minx = chunk_affine.c
        maxy = chunk_affine.f
        maxx = minx + chunk_width * chunk_affine.a
        miny = maxy + chunk_height * chunk_affine.e  # e < 0

        epsg_4326 = CRS.from_epsg(4326)
        if self.dst_crs.equals(epsg_4326):
            return [minx, miny, maxx, maxy]

        transformer = Transformer.from_crs(self.dst_crs, epsg_4326, always_xy=True)
        xs, ys = transformer.transform(
            [minx, maxx, minx, maxx],
            [maxy, maxy, miny, miny],
        )
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

    def _raw_getitem(self, key: tuple[Any, ...]) -> np.ndarray:
        """Materialise the chunk identified by ``key``.

        Args:
            key: A tuple of ``int | slice`` objects corresponding to the
                ``(time, y, x)`` dimensions.  Spatial dimensions may be
                integers (single-pixel selection) or slices.

        Returns:
            Numpy array with shape determined by the indexing key.

        """
        time_key, y_key, x_key = key

        # Resolve time dimension.
        n_dates = len(self.dates)
        if isinstance(time_key, (int, np.integer)):
            time_indices: list[int] = [int(time_key)]
            squeeze_time = True
        else:
            start = time_key.start if time_key.start is not None else 0
            stop = time_key.stop if time_key.stop is not None else n_dates
            step = time_key.step if time_key.step is not None else 1
            time_indices = list(range(start, stop, step))
            squeeze_time = False

        # Resolve spatial dimensions.
        # y_coords are ascending (south→north), but the affine transform and
        # the underlying COG data are top-down (north→south).  Convert logical
        # ascending indices to physical top-down row indices so the affine
        # correctly addresses the right pixels.
        #
        # Integer keys (single-pixel selection) are normalised to size-1
        # slices here; the dimension is squeezed before returning below.
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
        y_stop_logical = y_key.stop if y_key.stop is not None else self.dst_height
        x_start = x_key.start if x_key.start is not None else 0
        x_stop = x_key.stop if x_key.stop is not None else self.dst_width

        # logical index 0 = southernmost = physical row (dst_height - 1)
        y_start_physical = self.dst_height - y_stop_logical
        y_stop_physical = self.dst_height - y_start_logical

        chunk_height = y_stop_physical - y_start_physical
        chunk_width = x_stop - x_start

        # Translate affine to chunk origin (physical top-down row offset).
        chunk_affine = self.dst_affine * Affine.translation(x_start, y_start_physical)
        chunk_bbox_4326 = self._chunk_bbox_4326(chunk_affine, chunk_width, chunk_height)

        fill = self.nodata if self.nodata is not None else 0
        result = np.full(
            (len(time_indices), chunk_height, chunk_width),
            fill,
            dtype=self.dtype,
        )

        for i, t_idx in enumerate(time_indices):
            date = self.dates[t_idx]

            t0 = time.perf_counter()
            items = rustac.search_sync(
                self.parquet_path,
                bbox=chunk_bbox_4326,
                datetime=date,
                use_duckdb=True,
                sortby=self.sort_by,
                filter=self.filter,
                ids=self.ids,
            )
            logger.debug(
                "rustac.search_sync band=%r date=%s returned %d items in %.3fs",
                self.band,
                date,
                len(items),
                time.perf_counter() - t0,
            )

            if not items:
                logger.debug(
                    "No items for band=%r date=%s chunk_bbox=%s",
                    self.band,
                    date,
                    chunk_bbox_4326,
                )
                continue

            mosaic_method = self.mosaic_method_cls() if self.mosaic_method_cls else None

            t0 = time.perf_counter()
            chunk = _run_coroutine(
                async_mosaic_chunk(
                    items=items,
                    band=self.band,
                    chunk_affine=chunk_affine,
                    dst_crs=self.dst_crs,
                    chunk_width=chunk_width,
                    chunk_height=chunk_height,
                    nodata=self.nodata,
                    mosaic_method=mosaic_method,
                    store=self.store,
                    max_concurrent_reads=self.max_concurrent_reads,
                    path_fn=self.path_from_href,
                )
            )
            logger.debug(
                "async_mosaic_chunk band=%r date=%s (%d items, %dx%d px) took %.3fs",
                self.band,
                date,
                len(items),
                chunk_width,
                chunk_height,
                time.perf_counter() - t0,
            )

            # async_mosaic_chunk returns (bands, h, w); take band 0 for
            # single-band assets (the common case).
            result[i] = chunk[0] if chunk.ndim == 3 else chunk  # type: ignore[index]

        # Physical data is top-down (north→south); flip to match the ascending
        # (south→north) y coordinate order exposed to xarray callers.
        result = result[:, ::-1, :]

        # Apply time squeeze first (removes axis 0), then spatial squeezes
        # (axes shift down by 1 after time squeeze if it happened).
        out = result[0] if squeeze_time else result
        if squeeze_y:
            out = np.take(out, 0, axis=-2)
        if squeeze_x:
            out = out[..., 0]
        return out


@dataclass
class MultiBandStacBackendArray(BackendArray):
    """Lazy ``(band, time, y, x)`` array wrapping one backend per band.

    Holds one :class:`StacBackendArray` per band and stacks them along a
    leading band dimension.  This lets :func:`_build_dataarray` wrap the
    entire dataset in a single :class:`xarray.core.indexing.LazilyIndexedArray`
    instead of building per-band Variables and relying on ``xr.concat``
    (which would eagerly load ``LazilyIndexedArray``-backed objects).

    As a result, a narrow slice such as ``da.isel(time=0, x=0, y=0)``
    translates directly into a minimal I/O operation — one per-band query
    for a single pixel — rather than computing the entire array.

    Attributes:
        band_arrays: One :class:`StacBackendArray` per band, in order.
        band_names: Asset key strings corresponding to each entry in
            ``band_arrays``.
        dtype: NumPy dtype shared by all band arrays.
        shape: ``(n_bands, n_time, dst_height, dst_width)``.

    """

    band_arrays: list[StacBackendArray]
    band_names: list[str]
    dtype: np.dtype = field(init=False)
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Derive dtype and shape from the first band array."""
        first = self.band_arrays[0]
        self.dtype = first.dtype
        self.shape = (len(self.band_arrays),) + first.shape

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
        a single ``rustac.search_sync`` query per time step and sharing
        reprojection warp maps across bands that have identical source geometry.

        Args:
            key: A tuple of ``int | slice`` objects for the
                ``(band, time, y, x)`` dimensions.

        Returns:
            Numpy array with shape determined by the indexing key.

        """
        band_key, time_key, y_key, x_key = key

        # -- Band dimension --------------------------------------------------
        n_bands = len(self.band_arrays)
        if isinstance(band_key, (int, np.integer)):
            band_indices: list[int] = [int(band_key)]
            squeeze_band = True
        else:
            start = band_key.start if band_key.start is not None else 0
            stop = band_key.stop if band_key.stop is not None else n_bands
            step = band_key.step if band_key.step is not None else 1
            band_indices = list(range(start, stop, step))
            squeeze_band = False

        selected_band_names = [self.band_names[b] for b in band_indices]
        ref = self.band_arrays[band_indices[0]]

        # -- Time dimension --------------------------------------------------
        n_dates = len(ref.dates)
        if isinstance(time_key, (int, np.integer)):
            time_indices: list[int] = [int(time_key)]
            squeeze_time = True
        else:
            t_start = time_key.start if time_key.start is not None else 0
            t_stop = time_key.stop if time_key.stop is not None else n_dates
            t_step = time_key.step if time_key.step is not None else 1
            time_indices = list(range(t_start, t_stop, t_step))
            squeeze_time = False

        # -- Spatial dimensions ----------------------------------------------
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
        y_stop_logical = y_key.stop if y_key.stop is not None else ref.dst_height
        x_start = x_key.start if x_key.start is not None else 0
        x_stop = x_key.stop if x_key.stop is not None else ref.dst_width

        y_start_physical = ref.dst_height - y_stop_logical
        y_stop_physical = ref.dst_height - y_start_logical

        chunk_height = y_stop_physical - y_start_physical
        chunk_width = x_stop - x_start

        chunk_affine = ref.dst_affine * Affine.translation(x_start, y_start_physical)
        chunk_bbox_4326 = ref._chunk_bbox_4326(chunk_affine, chunk_width, chunk_height)

        # -- Read all bands together per time step ---------------------------
        fill = ref.nodata if ref.nodata is not None else 0
        result = np.full(
            (len(band_indices), len(time_indices), chunk_height, chunk_width),
            fill,
            dtype=self.dtype,
        )

        # Shared across all time steps: source tiles at the same grid position
        # (identical src_transform + src_crs) reuse the same WarpMap.
        warp_cache: dict = {}

        for i, t_idx in enumerate(time_indices):
            date = ref.dates[t_idx]

            t0 = time.perf_counter()
            items = rustac.search_sync(
                ref.parquet_path,
                bbox=chunk_bbox_4326,
                datetime=date,
                use_duckdb=True,
                sortby=ref.sort_by,
                filter=ref.filter,
                ids=ref.ids,
            )
            logger.debug(
                "rustac.search_sync bands=%r date=%s returned %d items in %.3fs",
                selected_band_names,
                date,
                len(items),
                time.perf_counter() - t0,
            )

            if not items:
                continue

            t0 = time.perf_counter()
            chunk_data = _run_coroutine(
                async_mosaic_chunk_multiband(
                    items=items,
                    bands=selected_band_names,
                    chunk_affine=chunk_affine,
                    dst_crs=ref.dst_crs,
                    chunk_width=chunk_width,
                    chunk_height=chunk_height,
                    nodata=ref.nodata,
                    mosaic_method_cls=ref.mosaic_method_cls,
                    store=ref.store,
                    max_concurrent_reads=ref.max_concurrent_reads,
                    warp_cache=warp_cache,
                    path_fn=ref.path_from_href,
                )
            )
            logger.debug(
                "async_mosaic_chunk_multiband bands=%r date=%s (%d items, %dx%d px) took %.3fs",
                selected_band_names,
                date,
                len(items),
                chunk_width,
                chunk_height,
                time.perf_counter() - t0,
            )

            for bi, band in enumerate(selected_band_names):
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
        if squeeze_y:
            out = np.take(out, 0, axis=-2)
        if squeeze_x:
            out = out[..., 0]
        return out
