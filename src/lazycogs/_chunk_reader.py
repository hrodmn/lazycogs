"""Async mosaic logic: open COGs, read windows, reproject, and mosaic."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.ma as ma
from affine import Affine
from async_geotiff import GeoTIFF, Overview, RasterArray, Window
from pyproj import CRS, Transformer

from lazycogs._mosaic_methods import FirstMethod, MosaicMethodBase
from lazycogs._reproject import (
    WarpMap,
    _get_transformer,
    apply_warp_map,
    compute_warp_map,
    reproject_array,
)
from lazycogs._store import resolve as _resolve_store

if TYPE_CHECKING:
    from obstore.store import ObjectStore

logger = logging.getLogger(__name__)


def _log_batch_failure(
    label: str, key: object, item_id: str, err: BaseException
) -> None:
    """Log a warning for an item that failed inside an asyncio.gather batch."""
    logger.warning(
        "Failed to read %s %r from item %s: %s", label, key, item_id, err, exc_info=err
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
    chunk_affine: Affine,
    dst_crs: CRS,
    chunk_width: int,
    chunk_height: int,
    store: ObjectStore | None = None,
    path_fn: Callable[[str], str] | None = None,
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
    store, path = _resolve_store(href, store, path_fn)

    t0 = time.perf_counter()
    geotiff = await GeoTIFF.open(path, store=store)
    logger.debug("GeoTIFF.open %s took %.3fs", path, time.perf_counter() - t0)

    target_res_native, t = _target_res_and_transformer(
        chunk_affine, chunk_width, chunk_height, dst_crs, geotiff.crs
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
    bbox_native = _chunk_bbox_native(chunk_affine, chunk_width, chunk_height, t)
    window = _native_window(reader, bbox_native, reader.width, reader.height)
    return geotiff, reader, window, path


async def _read_item_band(
    item: dict,
    band: str,
    chunk_affine: Affine,
    dst_crs: CRS,
    chunk_width: int,
    chunk_height: int,
    nodata: float | None,
    store: ObjectStore | None = None,
    path_fn: Callable[[str], str] | None = None,
) -> tuple[np.ndarray, float | None] | None:
    """Read and reproject one band from one STAC item.

    Returns a tuple of ``(array, effective_nodata)`` where *array* has shape
    ``(bands, chunk_height, chunk_width)`` and *effective_nodata* is the
    nodata value that was applied (may be ``None``).  Returns ``None`` if the
    item has no matching asset or its footprint does not overlap the chunk.
    """
    opened = await _open_and_window(
        item,
        band,
        chunk_affine,
        dst_crs,
        chunk_width,
        chunk_height,
        store=store,
        path_fn=path_fn,
    )
    if opened is None:
        return None
    geotiff, reader, window, path = opened
    if window is None:
        return None

    # Prefer the caller-supplied nodata; fall back to the value in the COG header.
    effective_nodata = nodata if nodata is not None else geotiff.nodata

    t0 = time.perf_counter()
    raster = await reader.read(window=window)
    logger.debug(
        "reader.read window=%s on %s took %.3fs",
        window,
        path,
        time.perf_counter() - t0,
    )

    # Fast path: skip reprojection when the read window already matches the
    # destination chunk exactly (same CRS, same affine, same pixel dimensions).
    if (
        geotiff.crs.equals(dst_crs)
        and raster.transform == chunk_affine
        and raster.data.shape[1] == chunk_height
        and raster.data.shape[2] == chunk_width
    ):
        logger.debug("Skipping reprojection for %s (identity grid)", path)
        return raster.data, effective_nodata

    # Reproject to the destination chunk grid.
    # Run in a thread executor so the event loop stays free to process
    # concurrent I/O completions from other items in the same gather.
    # pyproj and numpy both release the GIL, so threads give real parallelism.
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    arr = await loop.run_in_executor(
        None,
        lambda: reproject_array(
            data=raster.data,
            src_transform=raster.transform,
            src_crs=geotiff.crs,
            dst_transform=chunk_affine,
            dst_crs=dst_crs,
            dst_width=chunk_width,
            dst_height=chunk_height,
            nodata=effective_nodata,
        ),
    )
    logger.debug("reproject_array for %s took %.3fs", path, time.perf_counter() - t0)
    return arr, effective_nodata


async def async_mosaic_chunk(
    items: list[dict],
    band: str,
    chunk_affine: Affine,
    dst_crs: CRS,
    chunk_width: int,
    chunk_height: int,
    nodata: float | None = None,
    mosaic_method: MosaicMethodBase | None = None,
    store: ObjectStore | None = None,
    max_concurrent_reads: int = 32,
    path_fn: Callable[[str], str] | None = None,
) -> np.ndarray:
    """Read, reproject, and mosaic a single chunk from multiple STAC items.

    Items are processed in batches of ``max_concurrent_reads`` to bound peak
    memory usage.  When the mosaic method signals completion (e.g.
    :class:`~lazycogs._mosaic_methods.FirstMethod` once all pixels are
    filled), remaining batches are skipped entirely.

    Args:
        items: List of STAC item dicts to mosaic.  Processed in order.
        band: Asset key identifying the band to read from each item.
        chunk_affine: Affine transform of the destination chunk.
        dst_crs: CRS of the destination chunk.
        chunk_width: Width of the destination chunk in pixels.
        chunk_height: Height of the destination chunk in pixels.
        nodata: No-data fill value.
        mosaic_method: Pixel-selection strategy.  Defaults to
            :class:`~lazycogs._mosaic_methods.FirstMethod`.
        store: Optional pre-configured obstore ``ObjectStore`` instance
            forwarded to :func:`_read_item_band`.  When ``None``, each item's
            store is resolved from its HREF.
        max_concurrent_reads: Maximum number of COG reads to run concurrently.
            Limits peak in-flight memory when a chunk overlaps many items.
            Defaults to 32.
        path_fn: Optional callable that takes an asset HREF and returns the
            object path to use with *store*.  Forwarded to
            :func:`_read_item_band`.

    Returns:
        Array of shape ``(bands, chunk_height, chunk_width)`` with dtype
        matching the source COGs.

    """
    if mosaic_method is None:
        mosaic_method = FirstMethod()

    semaphore = asyncio.Semaphore(max_concurrent_reads)

    async def _guarded(item: dict) -> tuple[np.ndarray, float | None] | None:
        async with semaphore:
            return await _read_item_band(
                item,
                band,
                chunk_affine,
                dst_crs,
                chunk_width,
                chunk_height,
                nodata,
                store=store,
                path_fn=path_fn,
            )

    # Warn when the estimated peak in-flight memory is large. Each concurrent
    # read holds a reprojected (chunk_height, chunk_width) array; assume 4
    # bytes per pixel as a conservative upper bound regardless of source dtype.
    batch = min(max_concurrent_reads, len(items))
    estimated_peak_mb = batch * chunk_width * chunk_height * 4 / (1024**2)
    if estimated_peak_mb > 500:
        logger.warning(
            "Estimated peak in-flight memory for band=%r is ~%.0f MB "
            "(%d concurrent reads × %dx%d px). "
            "Lower max_concurrent_reads or add spatial chunks to reduce memory use.",
            band,
            estimated_peak_mb,
            batch,
            chunk_width,
            chunk_height,
        )

    # Launch all item reads as tasks up front; the semaphore inside _guarded
    # caps in-flight reads at max_concurrent_reads. Consume completions in
    # arrival order so a slow COG does not block already-finished reads from
    # being fed into the mosaic, and exit as soon as the mosaic is done.
    tasks: dict[asyncio.Task, dict] = {
        asyncio.ensure_future(_guarded(item)): item for item in items
    }
    pending: set[asyncio.Task] = set(tasks)
    try:
        mosaic_done = False
        while pending and not mosaic_done:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                item = tasks[fut]
                try:
                    result = fut.result()
                except BaseException as exc:
                    _log_batch_failure("band", band, item.get("id", "<unknown>"), exc)
                    continue

                if result is None:
                    continue

                arr, effective_nodata = result
                mosaic_method.feed(_array_to_masked(arr, effective_nodata))

                if mosaic_method.is_done:
                    mosaic_done = True
                    break
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    if mosaic_method._mosaic is None:
        bands = 1
        fill = nodata if nodata is not None else 0
        return np.full((bands, chunk_height, chunk_width), fill, dtype=np.float32)

    return mosaic_method.data


def _apply_bands_with_warp_cache(
    band_rasters: list[tuple[str, RasterArray, CRS, float | None]],
    dst_transform: Affine,
    dst_crs: CRS,
    dst_width: int,
    dst_height: int,
    warp_cache: dict[tuple[tuple[float, ...], str], WarpMap] | None = None,
) -> dict[str, tuple[np.ndarray, float | None]]:
    """Apply warp maps to multiple band rasters, reusing maps for identical geometries.

    Checks ``warp_cache`` (keyed on ``(tuple(raster.transform), src_crs.to_wkt())``)
    before computing a new warp map.  When ``warp_cache`` is shared across calls
    (e.g. across time steps in a single chunk read), warp maps for recurring tile
    geometries are computed only once.  Bands with different geometries each get
    their own correct warp map.

    This function is designed to run inside a thread executor — it is CPU-bound
    and must not be called from the async event loop directly.  When ``warp_cache``
    is shared across concurrent executor calls, two threads may both compute the
    same warp map before either stores it; this is safe because ``compute_warp_map``
    is deterministic and the duplicate result is simply overwritten.

    Args:
        band_rasters: List of ``(band_name, raster, src_crs, effective_nodata)``
            tuples.  ``raster`` must have ``.transform`` (Affine) and ``.data``
            (ndarray of shape ``(bands, h, w)``) attributes.
        dst_transform: Affine transform of the destination grid.
        dst_crs: CRS of the destination grid.
        dst_width: Width of the destination grid in pixels.
        dst_height: Height of the destination grid in pixels.
        warp_cache: Optional external cache shared across calls.  When ``None``
            a fresh local dict is used (original per-item behaviour).

    Returns:
        ``dict`` mapping band name to ``(reprojected_array, effective_nodata)``.

    """
    cache: dict[tuple[tuple[float, ...], str], WarpMap] = (
        warp_cache if warp_cache is not None else {}
    )
    results: dict[str, tuple[np.ndarray, float | None]] = {}

    for band, raster, src_crs, effective_nodata in band_rasters:
        # Fast path: skip reprojection when the read window already matches the
        # destination chunk exactly (same CRS, same affine, same pixel dimensions).
        if (
            src_crs.equals(dst_crs)
            and raster.transform == dst_transform
            and raster.data.shape[1] == dst_height
            and raster.data.shape[2] == dst_width
        ):
            results[band] = (raster.data, effective_nodata)
            continue
        cache_key = (tuple(raster.transform), src_crs)
        if cache_key not in cache:
            cache[cache_key] = compute_warp_map(
                src_transform=raster.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_width=dst_width,
                dst_height=dst_height,
            )
        results[band] = (
            apply_warp_map(raster.data, cache[cache_key], effective_nodata),
            effective_nodata,
        )

    return results


async def _read_item_bands(
    item: dict,
    bands: list[str],
    chunk_affine: Affine,
    dst_crs: CRS,
    chunk_width: int,
    chunk_height: int,
    nodata: float | None,
    store: ObjectStore | None = None,
    warp_cache: dict | None = None,
    path_fn: Callable[[str], str] | None = None,
) -> dict[str, tuple[np.ndarray, float | None]] | None:
    """Read and reproject multiple bands from one STAC item, sharing warp maps.

    Opens all band COGs concurrently, computes per-band windows independently
    (so bands with different native resolutions or extents are handled correctly),
    reads all windows concurrently, then dispatches a single thread-executor call
    that applies warp maps with caching: bands sharing the same source CRS and
    window transform reuse the same warp map.

    Args:
        item: STAC item dict containing an ``assets`` key.
        bands: Asset keys to read from this item.
        chunk_affine: Affine transform of the destination chunk.
        dst_crs: CRS of the destination chunk.
        chunk_width: Width of the destination chunk in pixels.
        chunk_height: Height of the destination chunk in pixels.
        nodata: No-data fill value.  When ``None``, the value stored in the
            COG header (``GeoTIFF.nodata``) is used if present.
        store: Optional pre-configured obstore ``ObjectStore`` instance.
        warp_cache: Optional cache shared across calls for reusing warp maps
            computed in earlier time steps.
        path_fn: Optional callable that takes an asset HREF and returns the
            object path to use with *store*.

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
        band_store, path = _resolve_store(href, store, path_fn)
        geotiff = await GeoTIFF.open(path, store=band_store)
        return band, geotiff, band_store

    open_results = await asyncio.gather(
        *[_open_band(b, h) for b, h in band_hrefs.items()]
    )

    # Per-band: select overview, compute window.
    # Each band is handled independently so differing native resolutions or
    # extents are handled correctly.
    band_read_plan: list[
        tuple[str, GeoTIFF, GeoTIFF | Overview, Window, float | None, CRS]
    ] = []
    for band, geotiff, _ in open_results:
        effective_nodata = nodata if nodata is not None else geotiff.nodata
        src_crs = geotiff.crs
        target_res_native, t = _target_res_and_transformer(
            chunk_affine, chunk_width, chunk_height, dst_crs, src_crs
        )
        overview = _select_overview(geotiff, target_res_native)
        reader = overview if overview is not None else geotiff
        bbox_native = _chunk_bbox_native(chunk_affine, chunk_width, chunk_height, t)
        window = _native_window(reader, bbox_native, reader.width, reader.height)
        if window is None:
            continue

        band_read_plan.append(
            (band, geotiff, reader, window, effective_nodata, src_crs)
        )

    if not band_read_plan:
        return None

    # Read all windows concurrently.
    async def _read_band(
        band: str, reader: GeoTIFF | Overview, window: Window
    ) -> tuple[str, RasterArray]:
        return band, await reader.read(window=window)

    read_results = await asyncio.gather(
        *[_read_band(b, r, w) for b, _, r, w, _, _ in band_read_plan]
    )

    effective_nodatas = {b: n for b, _, _, _, n, _ in band_read_plan}
    crss = {b: c for b, _, _, _, _, c in band_read_plan}

    band_rasters = [
        (band, raster, crss[band], effective_nodatas[band])
        for band, raster in read_results
    ]

    # Compute warp maps and apply, sharing maps across bands with identical geometry.
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        lambda: _apply_bands_with_warp_cache(
            band_rasters,
            chunk_affine,
            dst_crs,
            chunk_width,
            chunk_height,
            warp_cache,
        ),
    )
    return results


async def async_mosaic_chunk_multiband(
    items: list[dict],
    bands: list[str],
    chunk_affine: Affine,
    dst_crs: CRS,
    chunk_width: int,
    chunk_height: int,
    nodata: float | None = None,
    mosaic_method_cls: type[MosaicMethodBase] | None = None,
    store: ObjectStore | None = None,
    max_concurrent_reads: int = 32,
    warp_cache: dict | None = None,
    path_fn: Callable[[str], str] | None = None,
) -> dict[str, np.ndarray]:
    """Read, reproject, and mosaic multiple bands from a list of STAC items.

    Multi-band variant of :func:`async_mosaic_chunk`.  Processes all requested
    bands together per item so that bands sharing the same source geometry
    compute the reprojection warp map only once (via
    :func:`_apply_bands_with_warp_cache`).

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
        store: Optional pre-configured obstore ``ObjectStore`` instance.
        max_concurrent_reads: Maximum number of COG reads to run concurrently.
        warp_cache: Optional cache shared across calls for reusing warp maps
            from earlier time steps.
        path_fn: Optional callable that takes an asset HREF and returns the
            object path to use with *store*.  Forwarded to
            :func:`_read_item_bands`.

    Returns:
        ``dict`` mapping each band name to an array of shape
        ``(cog_bands, chunk_height, chunk_width)`` with dtype matching the
        source COGs.

    """
    if mosaic_method_cls is None:
        mosaic_method_cls = FirstMethod

    semaphore = asyncio.Semaphore(max_concurrent_reads)

    async def _guarded(item: dict) -> dict[str, tuple[np.ndarray, float | None]] | None:
        async with semaphore:
            return await _read_item_bands(
                item,
                bands,
                chunk_affine,
                dst_crs,
                chunk_width,
                chunk_height,
                nodata,
                store=store,
                warp_cache=warp_cache,
                path_fn=path_fn,
            )

    batch_size = min(max_concurrent_reads, len(items))
    estimated_peak_mb = (
        batch_size * len(bands) * chunk_width * chunk_height * 4 / (1024**2)
    )
    if estimated_peak_mb > 500:
        logger.warning(
            "Estimated peak in-flight memory for bands=%r is ~%.0f MB "
            "(%d concurrent reads × %d bands × %dx%d px). "
            "Lower max_concurrent_reads or add spatial chunks to reduce memory use.",
            bands,
            estimated_peak_mb,
            batch_size,
            len(bands),
            chunk_width,
            chunk_height,
        )

    mosaic_methods: dict[str, MosaicMethodBase] = {
        b: mosaic_method_cls() for b in bands
    }

    # Launch all item reads as tasks up front; the semaphore inside _guarded
    # caps in-flight reads at max_concurrent_reads. Consume completions in
    # arrival order so a slow COG does not block already-finished reads from
    # being fed into the mosaics, and exit as soon as every band is done.
    tasks: dict[asyncio.Task, dict] = {
        asyncio.ensure_future(_guarded(item)): item for item in items
    }
    pending: set[asyncio.Task] = set(tasks)
    try:
        mosaic_done = False
        while pending and not mosaic_done:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                item = tasks[fut]
                try:
                    result = fut.result()
                except BaseException as exc:
                    _log_batch_failure("bands", bands, item.get("id", "<unknown>"), exc)
                    continue

                if result is None:
                    continue

                for band, (arr, effective_nodata) in result.items():
                    mosaic_methods[band].feed(_array_to_masked(arr, effective_nodata))

                if all(m.is_done for m in mosaic_methods.values()):
                    mosaic_done = True
                    break
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    fill = nodata if nodata is not None else 0
    output: dict[str, np.ndarray] = {}
    for band in bands:
        method = mosaic_methods[band]
        if method._mosaic is None:
            output[band] = np.full(
                (1, chunk_height, chunk_width), fill, dtype=np.float32
            )
        else:
            output[band] = method.data
    return output
