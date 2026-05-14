"""Async mosaic logic: open COGs, read windows, reproject, and mosaic."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from async_geotiff import GeoTIFF, Overview, RasterArray, Window
from numpy import ma

from lazycogs._executor import _run_coroutine
from lazycogs._mosaic_methods import FirstMethod, MosaicMethodBase
from lazycogs._store import resolve as _resolve_store
from lazycogs._warp import (
    ReprojectRequest,
    ResamplingMethod,
    _get_transformer,
    reproject_tile,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from affine import Affine
    from obstore.store import ObjectStore
    from pyproj import CRS, Transformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ChunkContext:
    """Immutable per-chunk parameters shared across all item reads.

    Built once per chunk in ``read_chunk_async`` and passed through to all
    internal helpers. Frozen to prevent accidental mutation across concurrent
    coroutines.
    """

    chunk_affine: Affine
    dst_crs: CRS
    chunk_width: int
    chunk_height: int
    nodata: float | None
    resampling: ResamplingMethod
    store: ObjectStore | None
    path_fn: Callable[[str], str] | None


def _log_batch_failure(
    label: str,
    key: object,
    item_id: str,
    err: BaseException,
) -> None:
    """Log a warning for an item that failed inside an asyncio.gather batch."""
    logger.warning(
        "Failed to read %s %r from item %s: %s",
        label,
        key,
        item_id,
        err,
        exc_info=err,
    )


def _target_res_and_transformer(
    chunk_affine: Affine,
    chunk_width: int,
    chunk_height: int,
    dst_crs: CRS,
    src_crs: CRS,
) -> tuple[float, Transformer | None]:
    """Return ``(target_res_native, transformer)`` for the dst→src reprojection.

    *transformer* is ``None`` when source and destination share a CRS, in which
    case *target_res_native* is just the destination pixel width. Otherwise the
    pixel width is estimated at the chunk center by projecting two adjacent
    pixel centers to the source CRS.
    """
    if dst_crs.equals(src_crs):
        return abs(chunk_affine.a), None
    t = _get_transformer(dst_crs, src_crs)
    cx = chunk_affine.c + (chunk_width / 2) * chunk_affine.a
    cy = chunk_affine.f + (chunk_height / 2) * chunk_affine.e
    x0, _ = t.transform(cx, cy)
    x1, _ = t.transform(cx + chunk_affine.a, cy)
    return abs(x1 - x0), t


def _array_to_masked(arr: np.ndarray, effective_nodata: float | None) -> ma.MaskedArray:
    """Wrap ``arr`` in a MaskedArray, masking pixels equal to ``effective_nodata``.

    A pixel is masked only when *all* bands equal ``effective_nodata`` (so a
    valid pixel in any band keeps the position unmasked). When
    ``effective_nodata`` is ``None``, nothing is masked.
    """
    if effective_nodata is None:
        mask = np.zeros(arr.shape, dtype=bool)
    else:
        per_pixel = np.all(arr == effective_nodata, axis=0, keepdims=True)
        mask = np.broadcast_to(per_pixel, arr.shape).copy()
    return ma.MaskedArray(arr, mask=mask)


def _select_overview(geotiff: GeoTIFF, target_res: float) -> Overview | None:
    """Choose the coarsest overview whose resolution is <= ``target_res``.

    Picks the finest source data that avoids upsampling: the selected
    overview's pixel size is no larger than the output pixel size, so each
    output pixel samples at least as much original detail as it represents.
    This preserves spatial variation rather than smearing it with a coarser
    overview level.

    Args:
        geotiff: Open GeoTIFF object.
        target_res: Target pixel size in the COG's native CRS units.

    Returns:
        An ``Overview`` instance, or ``None`` to use full resolution.

    """
    if not geotiff.overviews:
        return None

    native_res = abs(geotiff.transform.a)
    if target_res <= native_res:
        return None

    # Overviews are ordered finest → coarsest.  Walk forward as long as the
    # overview resolution is still <= target_res.  The last qualifying entry
    # is the coarsest overview that won't over-smooth the output.
    selected: Overview | None = None
    for overview in geotiff.overviews:
        if abs(overview.transform.a) <= target_res:
            selected = overview
        else:
            break

    # selected is None when target_res falls between native_res and the finest
    # overview (e.g. 15 m target, 10 m native, 20 m finest overview).
    # Fall back to full resolution rather than upsampling from the overview.
    return selected


def _chunk_bbox_native(
    chunk_affine: Affine,
    chunk_width: int,
    chunk_height: int,
    transformer: Transformer | None,
) -> tuple[float, float, float, float]:
    """Return the chunk's ``(minx, miny, maxx, maxy)`` in the source CRS.

    When ``transformer`` is ``None`` the chunk is assumed to already be in the
    source CRS and the bbox is returned directly. Otherwise the four corners
    are projected and the axis-aligned envelope is returned.
    """
    minx = chunk_affine.c
    maxy = chunk_affine.f
    maxx = minx + chunk_width * chunk_affine.a
    miny = maxy + chunk_height * chunk_affine.e  # e is negative
    if transformer is None:
        return (minx, miny, maxx, maxy)
    xs, ys = transformer.transform(
        [minx, maxx, minx, maxx],
        [maxy, maxy, miny, miny],
    )
    return (min(xs), min(ys), max(xs), max(ys))


def _native_window(
    geotiff: GeoTIFF | Overview,
    bbox_native: tuple[float, float, float, float],
    width: int,
    height: int,
) -> Window | None:
    """Compute the pixel window in a source image that covers ``bbox_native``.

    Args:
        geotiff: Full-resolution ``GeoTIFF`` or ``Overview`` to read from.
        bbox_native: ``(minx, miny, maxx, maxy)`` in the source image's CRS.
        width: Image width in pixels (used for bounds clamping).
        height: Image height in pixels (used for bounds clamping).

    Returns:
        A ``Window`` clipped to the image extent, or ``None`` if the bbox
        falls entirely outside the image.

    """
    inv = ~geotiff.transform
    minx, miny, maxx, maxy = bbox_native

    # Map the four corners of the bbox to pixel space (col, row).
    corners = [
        (inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f)
        for x, y in [(minx, maxy), (maxx, maxy), (minx, miny), (maxx, miny)]
    ]
    col_frac = [c[0] for c in corners]
    row_frac = [c[1] for c in corners]

    col_min = max(0, int(np.floor(min(col_frac))))
    row_min = max(0, int(np.floor(min(row_frac))))
    col_max = min(width, int(np.ceil(max(col_frac))))
    row_max = min(height, int(np.ceil(max(row_frac))))

    if col_max <= col_min or row_max <= row_min:
        return None

    return Window(
        col_off=col_min,
        row_off=row_min,
        width=col_max - col_min,
        height=row_max - row_min,
    )


async def _open_and_window(
    item: dict,
    band: str,
    ctx: _ChunkContext,
) -> tuple[GeoTIFF, GeoTIFF | Overview, Window | None, str] | None:
    """Open a COG asset and compute the pixel window covering the chunk.

    Returns ``(geotiff, reader, window, path)`` where *reader* is an overview
    when one matches the target resolution and *window* is ``None`` if the
    chunk does not overlap the source image. Returns ``None`` when the item
    has no matching asset.
    """
    asset = item.get("assets", {}).get(band)
    if asset is None:
        logger.debug("Item %s has no asset %r; skipping.", item.get("id"), band)
        return None

    href = asset["href"]
    store, path = _resolve_store(href, ctx.store, ctx.path_fn)

    t0 = time.perf_counter()
    geotiff = await GeoTIFF.open(path, store=store)
    logger.debug("GeoTIFF.open %s took %.3fs", path, time.perf_counter() - t0)

    target_res_native, t = _target_res_and_transformer(
        ctx.chunk_affine,
        ctx.chunk_width,
        ctx.chunk_height,
        ctx.dst_crs,
        geotiff.crs,
    )
    overview = _select_overview(geotiff, target_res_native)
    if overview is not None:
        logger.debug(
            "Selected overview level %d (res=%.2f) for target_res=%.2f on %s",
            geotiff.overviews.index(overview),
            abs(overview.transform.a),
            target_res_native,
            path,
        )
    reader: GeoTIFF | Overview = overview if overview is not None else geotiff
    bbox_native = _chunk_bbox_native(
        ctx.chunk_affine,
        ctx.chunk_width,
        ctx.chunk_height,
        t,
    )
    window = _native_window(reader, bbox_native, reader.width, reader.height)
    return geotiff, reader, window, path


def _reproject_bands(
    band_rasters: list[tuple[str, RasterArray, CRS, float | None]],
    dst_transform: Affine,
    dst_crs: CRS,
    dst_width: int,
    dst_height: int,
    resampling: ResamplingMethod = ResamplingMethod.NEAREST,
) -> dict[str, tuple[np.ndarray, float | None]]:
    """Reproject multiple band rasters onto one destination grid.

    Args:
        band_rasters: List of ``(band_name, raster, src_crs, effective_nodata)``
            tuples. ``raster`` must have ``.transform`` (Affine) and ``.data``
            (ndarray of shape ``(bands, h, w)``) attributes.
        dst_transform: Affine transform of the destination grid.
        dst_crs: CRS of the destination grid.
        dst_width: Width of the destination grid in pixels.
        dst_height: Height of the destination grid in pixels.
        resampling: Reprojection resampling method.

    Returns:
        ``dict`` mapping band name to ``(reprojected_array, effective_nodata)``.

    """
    results: dict[str, tuple[np.ndarray, float | None]] = {}

    for band, raster, src_crs, effective_nodata in band_rasters:
        results[band] = (
            reproject_tile(
                ReprojectRequest(
                    data=raster.data,
                    src_transform=raster.transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_width=dst_width,
                    dst_height=dst_height,
                    nodata=effective_nodata,
                    resampling=resampling,
                ),
            ),
            effective_nodata,
        )

    return results


async def _read_item_band(
    item: dict,
    bands: list[str],
    ctx: _ChunkContext,
) -> dict[str, tuple[np.ndarray, float | None]] | None:
    """Read and reproject multiple bands from one STAC item.

    Opens all band COGs concurrently, computes per-band windows independently
    (so bands with different native resolutions or extents are handled correctly),
    reads all windows concurrently, then dispatches a single thread-executor call
    that reprojects all bands onto the destination grid.

    Args:
        item: STAC item dict containing an ``assets`` key.
        bands: Asset keys to read from this item.
        ctx: Per-chunk invariants (affine, CRS, dimensions, nodata, store, etc.).

    Returns:
        ``dict`` mapping band name to ``(array, effective_nodata)`` where
        *array* has shape ``(bands, chunk_height, chunk_width)``.  Returns
        ``None`` if no requested band overlaps the chunk.

    """
    # Collect hrefs for all requested bands.
    band_hrefs: dict[str, str] = {}
    for band in bands:
        asset = item.get("assets", {}).get(band)
        if asset is not None:
            band_hrefs[band] = asset["href"]

    if not band_hrefs:
        return None

    # Open all COGs concurrently for metadata.
    async def _open_band(band: str, href: str) -> tuple[str, GeoTIFF, ObjectStore]:
        band_store, path = _resolve_store(href, ctx.store, ctx.path_fn)
        geotiff = await GeoTIFF.open(path, store=band_store)
        return band, geotiff, band_store

    open_results = await asyncio.gather(
        *[_open_band(b, h) for b, h in band_hrefs.items()],
    )

    # Per-band: select overview, compute window.
    # Each band is handled independently so differing native resolutions or
    # extents are handled correctly.
    band_read_plan: list[
        tuple[str, GeoTIFF, GeoTIFF | Overview, Window, float | None, CRS]
    ] = []
    for band, geotiff, _ in open_results:
        effective_nodata = ctx.nodata if ctx.nodata is not None else geotiff.nodata
        src_crs = geotiff.crs
        target_res_native, t = _target_res_and_transformer(
            ctx.chunk_affine,
            ctx.chunk_width,
            ctx.chunk_height,
            ctx.dst_crs,
            src_crs,
        )
        overview = _select_overview(geotiff, target_res_native)
        reader = overview if overview is not None else geotiff
        bbox_native = _chunk_bbox_native(
            ctx.chunk_affine,
            ctx.chunk_width,
            ctx.chunk_height,
            t,
        )
        window = _native_window(reader, bbox_native, reader.width, reader.height)
        if window is None:
            continue

        band_read_plan.append(
            (band, geotiff, reader, window, effective_nodata, src_crs),
        )

    if not band_read_plan:
        return None

    # Read all windows concurrently.
    async def _read_band(
        band: str,
        reader: GeoTIFF | Overview,
        window: Window,
    ) -> tuple[str, RasterArray]:
        return band, await reader.read(window=window)

    read_results = await asyncio.gather(
        *[_read_band(b, r, w) for b, _, r, w, _, _ in band_read_plan],
    )

    effective_nodatas = {b: n for b, _, _, _, n, _ in band_read_plan}
    crss = {b: c for b, _, _, _, _, c in band_read_plan}

    band_rasters = [
        (band, raster, crss[band], effective_nodatas[band])
        for band, raster in read_results
    ]

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _reproject_bands(
            band_rasters,
            ctx.chunk_affine,
            ctx.dst_crs,
            ctx.chunk_width,
            ctx.chunk_height,
            ctx.resampling,
        ),
    )


async def _drain_in_order(
    tasks: list[asyncio.Task],
    on_result: Callable[[int, Any], None],
    is_done: Callable[[], bool],
    on_error: Callable[[int, BaseException], None],
) -> None:
    """Feed completed tasks to on_result in source-list order.

    Tasks may complete in any order (I/O arrival order). Results are buffered
    by their original list index and handed to on_result strictly in the order
    they appear in tasks, so callers see results in the same order as the
    input list regardless of network timing.

    Stops early when is_done() returns True. Cancels and drains all remaining
    tasks on exit, whether done early or exhausted.

    Args:
        tasks: Pre-created asyncio tasks, in the order results should be fed.
        on_result: Called with (index, result) for each completed task, in
            source order. result is whatever the task returned (may be None).
        is_done: Called after each on_result; if it returns True, remaining
            tasks are cancelled and the function returns.
        on_error: Called with (index, exception) for tasks that raised. The
            task's slot is treated as None for ordering purposes.

    """
    task_index: dict[int, int] = {id(t): i for i, t in enumerate(tasks)}
    completed: dict[int, Any] = {}
    cursor = 0
    pending: set[asyncio.Task] = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for fut in done:
                idx = task_index[id(fut)]
                try:
                    completed[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    on_error(idx, exc)
                    completed[idx] = None

            while cursor in completed:
                result = completed.pop(cursor)
                on_result(cursor, result)
                cursor += 1
                if is_done():
                    return
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def read_chunk_async(
    items: list[dict],
    bands: list[str],
    chunk_affine: Affine,
    dst_crs: CRS,
    chunk_width: int,
    chunk_height: int,
    nodata: float | None = None,
    mosaic_method_cls: type[MosaicMethodBase] | None = None,
    resampling: ResamplingMethod = ResamplingMethod.NEAREST,
    store: ObjectStore | None = None,
    max_concurrent_reads: int = 32,
    path_fn: Callable[[str], str] | None = None,
) -> dict[str, np.ndarray]:
    """Read, reproject, and mosaic multiple bands from a list of STAC items.

    Processes all requested bands together per item so they can be reprojected
    in one thread-executor call.

    Items are processed in batches of ``max_concurrent_reads``.  When all
    per-band mosaic methods signal completion, remaining batches are skipped.

    Args:
        items: List of STAC item dicts to mosaic.  Processed in order.
        bands: Asset keys identifying the bands to read from each item.
        chunk_affine: Affine transform of the destination chunk.
        dst_crs: CRS of the destination chunk.
        chunk_width: Width of the destination chunk in pixels.
        chunk_height: Height of the destination chunk in pixels.
        nodata: No-data fill value.
        mosaic_method_cls: Mosaic method class instantiated once per band.
            Defaults to :class:`~lazycogs._mosaic_methods.FirstMethod`.
        resampling: Reprojection resampling method.
        store: Optional pre-configured obstore ``ObjectStore`` instance.
        max_concurrent_reads: Maximum number of COG reads to run concurrently.
        path_fn: Optional callable that takes an asset HREF and returns the
            object path to use with *store*.  Forwarded to
            :func:`_read_item_band`.

    Returns:
        ``dict`` mapping each band name to an array of shape
        ``(cog_bands, chunk_height, chunk_width)`` with dtype matching the
        source COGs.

    """
    if mosaic_method_cls is None:
        mosaic_method_cls = FirstMethod

    ctx = _ChunkContext(
        chunk_affine=chunk_affine,
        dst_crs=dst_crs,
        chunk_width=chunk_width,
        chunk_height=chunk_height,
        nodata=nodata,
        resampling=resampling,
        store=store,
        path_fn=path_fn,
    )

    semaphore = asyncio.Semaphore(max_concurrent_reads)

    async def _guarded(item: dict) -> dict[str, tuple[np.ndarray, float | None]] | None:
        async with semaphore:
            return await _read_item_band(item, bands, ctx)

    mosaic_methods: dict[str, MosaicMethodBase] = {
        b: mosaic_method_cls() for b in bands
    }

    task_list: list[asyncio.Task] = [
        asyncio.ensure_future(_guarded(item)) for item in items
    ]

    def _feed(
        _idx: int,
        result: dict[str, tuple[np.ndarray, float | None]] | None,
    ) -> None:
        if result is None:
            return
        for band, (arr, effective_nodata) in result.items():
            mosaic_methods[band].feed(_array_to_masked(arr, effective_nodata))

    def _done() -> bool:
        return all(m.is_done for m in mosaic_methods.values())

    def _error(idx: int, exc: BaseException) -> None:
        _log_batch_failure("bands", bands, items[idx].get("id", "<unknown>"), exc)

    await _drain_in_order(task_list, _feed, _done, _error)

    fill = nodata if nodata is not None else 0
    output: dict[str, np.ndarray] = {}
    for band in bands:
        try:
            output[band] = mosaic_methods[band].data
        except ValueError:
            output[band] = np.full(
                (1, chunk_height, chunk_width),
                fill,
                dtype=np.float32,
            )
    return output


def read_chunk(
    items: list[dict],
    bands: list[str],
    chunk_affine: Affine,
    dst_crs: CRS,
    chunk_width: int,
    chunk_height: int,
    nodata: float | None = None,
    mosaic_method_cls: type[MosaicMethodBase] | None = None,
    resampling: ResamplingMethod = ResamplingMethod.NEAREST,
    store: ObjectStore | None = None,
    max_concurrent_reads: int = 32,
    path_fn: Callable[[str], str] | None = None,
) -> dict[str, np.ndarray]:
    """Run :func:`read_chunk_async` on the persistent per-thread background loop.

    All arguments are identical to :func:`read_chunk_async`.

    Args:
        items: List of STAC item dicts to mosaic.  Processed in order.
        bands: Asset keys identifying the bands to read from each item.
        chunk_affine: Affine transform of the destination chunk.
        dst_crs: CRS of the destination chunk.
        chunk_width: Width of the destination chunk in pixels.
        chunk_height: Height of the destination chunk in pixels.
        nodata: No-data fill value.
        mosaic_method_cls: Mosaic method class instantiated once per band.
            Defaults to :class:`~lazycogs._mosaic_methods.FirstMethod`.
        resampling: Reprojection resampling method.
        store: Optional pre-configured obstore ``ObjectStore`` instance.
        max_concurrent_reads: Maximum number of COG reads to run concurrently.
        path_fn: Optional callable that takes an asset HREF and returns the
            object path to use with *store*.

    Returns:
        ``dict`` mapping each band name to an array of shape
        ``(cog_bands, chunk_height, chunk_width)`` with dtype matching the
        source COGs.

    """
    return _run_coroutine(
        read_chunk_async(
            items=items,
            bands=bands,
            chunk_affine=chunk_affine,
            dst_crs=dst_crs,
            chunk_width=chunk_width,
            chunk_height=chunk_height,
            nodata=nodata,
            mosaic_method_cls=mosaic_method_cls,
            resampling=resampling,
            store=store,
            max_concurrent_reads=max_concurrent_reads,
            path_fn=path_fn,
        ),
    )
