"""Async mosaic logic: open COGs, read windows, reproject, and mosaic."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import numpy.ma as ma
from affine import Affine
from async_geotiff import GeoTIFF, Overview, Window
from pyproj import CRS, Transformer

from lazycogs._mosaic_methods import FirstMethod, MosaicMethodBase
from lazycogs._reproject import (
    WarpMap,
    apply_warp_map,
    compute_warp_map,
    reproject_array,
)
from lazycogs._store import path_from_href, store_from_href

if TYPE_CHECKING:
    from obstore.store import ObjectStore

logger = logging.getLogger(__name__)


def _select_overview(geotiff: GeoTIFF, target_res: float) -> Overview | None:
    """Choose the finest overview whose resolution is >= ``target_res``.

    Args:
        geotiff: Open GeoTIFF object.
        target_res: Target pixel size in the COG's native CRS units.

    Returns:
        An ``Overview`` instance, or ``None`` to indicate the full-resolution
        image should be used.

    """
    if not geotiff.overviews:
        return None

    native_res = abs(geotiff.transform.a)
    if target_res <= native_res:
        return None

    # Overviews are ordered finest → coarsest.  Find the first one whose
    # resolution is at least as coarse as the target (no detail lost).
    for overview in geotiff.overviews:
        if abs(overview.transform.a) >= target_res:
            return overview

    # All overviews are finer than target; use the coarsest to minimise I/O.
    return geotiff.overviews[-1]


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


async def _read_item_band(
    item: dict,
    band: str,
    chunk_affine: Affine,
    dst_crs: CRS,
    chunk_width: int,
    chunk_height: int,
    nodata: float | None,
    store: ObjectStore | None = None,
) -> tuple[np.ndarray, float | None] | None:
    """Read and reproject one band from one STAC item.

    Args:
        item: STAC item dict containing an ``assets`` key.
        band: Asset key identifying the band to read.
        chunk_affine: Affine transform of the destination chunk.
        dst_crs: CRS of the destination chunk.
        chunk_width: Width of the destination chunk in pixels.
        chunk_height: Height of the destination chunk in pixels.
        nodata: No-data fill value.  When ``None``, the value stored in the
            COG header (``GeoTIFF.nodata``) is used if present.
        store: Optional pre-configured obstore ``ObjectStore`` instance.
            When provided, it is used directly and the path is extracted from
            the asset HREF (path component only).  When ``None``, the store
            is resolved and cached via :func:`~lazycogs._store.store_from_href`.

    Returns:
        A tuple of ``(array, effective_nodata)`` where *array* has shape
        ``(bands, chunk_height, chunk_width)`` and *effective_nodata* is the
        nodata value that was applied (may be ``None``).  Returns ``None`` if
        the item's footprint does not overlap the chunk.

    """
    asset = item.get("assets", {}).get(band)
    if asset is None:
        logger.debug("Item %s has no asset %r; skipping.", item.get("id"), band)
        return None

    href = asset["href"]
    if store is not None:
        path = path_from_href(href)
    else:
        store, path = store_from_href(href)

    t0 = time.perf_counter()
    geotiff = await GeoTIFF.open(path, store=store)
    logger.debug("GeoTIFF.open %s took %.3fs", path, time.perf_counter() - t0)

    # Prefer the caller-supplied nodata; fall back to the value in the COG header.
    effective_nodata = nodata if nodata is not None else geotiff.nodata

    # Select appropriate overview for the target resolution.
    target_res_native = abs(chunk_affine.a)
    if not dst_crs.equals(geotiff.crs):
        # Rough conversion: transform a 1-pixel offset at chunk centre.
        cx = chunk_affine.c + (chunk_width / 2) * chunk_affine.a
        cy = chunk_affine.f + (chunk_height / 2) * chunk_affine.e
        t = Transformer.from_crs(dst_crs, geotiff.crs, always_xy=True)
        x0, y0 = t.transform(cx, cy)
        x1, y1 = t.transform(cx + chunk_affine.a, cy)
        target_res_native = abs(x1 - x0)

    reader: GeoTIFF | Overview
    overview = _select_overview(geotiff, target_res_native)
    if overview is not None:
        logger.debug(
            "Selected overview level %d (res=%.2f) for target_res=%.2f on %s",
            geotiff.overviews.index(overview),
            abs(overview.transform.a),
            target_res_native,
            path,
        )
    reader = overview if overview is not None else geotiff
    src_width = reader.width
    src_height = reader.height

    # Transform chunk corners to source CRS for window calculation.
    chunk_minx = chunk_affine.c
    chunk_maxy = chunk_affine.f
    chunk_maxx = chunk_minx + chunk_width * chunk_affine.a
    chunk_miny = chunk_maxy + chunk_height * chunk_affine.e  # e is negative

    if dst_crs.equals(geotiff.crs):
        bbox_native = (chunk_minx, chunk_miny, chunk_maxx, chunk_maxy)
    else:
        t_to_src = Transformer.from_crs(dst_crs, geotiff.crs, always_xy=True)
        xs, ys = t_to_src.transform(
            [chunk_minx, chunk_maxx, chunk_minx, chunk_maxx],
            [chunk_maxy, chunk_maxy, chunk_miny, chunk_miny],
        )
        bbox_native = (min(xs), min(ys), max(xs), max(ys))

    window = _native_window(reader, bbox_native, src_width, src_height)
    if window is None:
        return None

    t0 = time.perf_counter()
    raster = await reader.read(window=window)
    logger.debug(
        "reader.read window=%s on %s took %.3fs",
        window,
        path,
        time.perf_counter() - t0,
    )

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

    Returns:
        Array of shape ``(bands, chunk_height, chunk_width)`` with dtype
        matching the source COGs.

    """
    if mosaic_method is None:
        mosaic_method = FirstMethod()

    logger.debug(
        "async_mosaic_chunk band=%r %dx%d px, %d items (max_concurrent_reads=%d)",
        band,
        chunk_width,
        chunk_height,
        len(items),
        max_concurrent_reads,
    )

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

    # Process items in batches of max_concurrent_reads so that:
    # 1. At most max_concurrent_reads arrays are held in memory at once.
    # 2. When the mosaic method signals is_done, we skip remaining batches.
    t0 = time.perf_counter()
    done = False
    items_read = 0
    for batch_start in range(0, len(items), max_concurrent_reads):
        batch = items[batch_start : batch_start + max_concurrent_reads]
        batch_results = await asyncio.gather(
            *[_guarded(item) for item in batch],
            return_exceptions=True,
        )
        items_read += len(batch)

        for j, result in enumerate(batch_results):
            item_idx = batch_start + j
            if isinstance(result, BaseException):
                item_id = items[item_idx].get("id", "<unknown>")
                logger.warning(
                    "Failed to read band %r from item %s: %s",
                    band,
                    item_id,
                    result,
                    exc_info=result,
                )
                continue

            if result is None:
                continue

            arr, effective_nodata = result
            arr_mask: np.ndarray
            if effective_nodata is not None:
                arr_mask = np.all(arr == effective_nodata, axis=0, keepdims=True)
                arr_mask = np.broadcast_to(arr_mask, arr.shape).copy()
            else:
                arr_mask = np.zeros(arr.shape, dtype=bool)

            mosaic_method.feed(ma.MaskedArray(arr, mask=arr_mask))

            if mosaic_method.is_done:
                done = True
                break

        if done:
            break

    logger.debug(
        "async_mosaic_chunk band=%r read %d/%d items in %.3fs",
        band,
        items_read,
        len(items),
        time.perf_counter() - t0,
    )

    if mosaic_method._mosaic is None:
        bands = 1
        fill = nodata if nodata is not None else 0
        return np.full((bands, chunk_height, chunk_width), fill, dtype=np.float32)

    return mosaic_method.data


def _apply_bands_with_warp_cache(
    band_rasters: list[tuple[str, object, CRS, float | None]],
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
        cache_key = (tuple(raster.transform), src_crs.to_wkt())
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
    store: object | None = None,
    warp_cache: dict | None = None,
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
    async def _open_band(band: str, href: str) -> tuple[str, object, object]:
        if store is not None:
            path = path_from_href(href)
            geotiff = await GeoTIFF.open(path, store=store)
            return band, geotiff, store
        band_store, path = store_from_href(href)
        geotiff = await GeoTIFF.open(path, store=band_store)
        return band, geotiff, band_store

    open_results = await asyncio.gather(
        *[_open_band(b, h) for b, h in band_hrefs.items()]
    )

    # Per-band: select overview, compute window.
    # Each band is handled independently so differing native resolutions or
    # extents are handled correctly.
    band_read_plan: list[tuple[str, object, object, Window, float | None]] = []
    for band, geotiff, _ in open_results:
        effective_nodata = nodata if nodata is not None else geotiff.nodata

        target_res_native = abs(chunk_affine.a)
        if not dst_crs.equals(geotiff.crs):
            cx = chunk_affine.c + (chunk_width / 2) * chunk_affine.a
            cy = chunk_affine.f + (chunk_height / 2) * chunk_affine.e
            t = Transformer.from_crs(dst_crs, geotiff.crs, always_xy=True)
            x0, y0 = t.transform(cx, cy)
            x1, y1 = t.transform(cx + chunk_affine.a, cy)
            target_res_native = abs(x1 - x0)

        overview = _select_overview(geotiff, target_res_native)
        reader = overview if overview is not None else geotiff

        chunk_minx = chunk_affine.c
        chunk_maxy = chunk_affine.f
        chunk_maxx = chunk_minx + chunk_width * chunk_affine.a
        chunk_miny = chunk_maxy + chunk_height * chunk_affine.e

        if dst_crs.equals(geotiff.crs):
            bbox_native = (chunk_minx, chunk_miny, chunk_maxx, chunk_maxy)
        else:
            t_to_src = Transformer.from_crs(dst_crs, geotiff.crs, always_xy=True)
            xs, ys = t_to_src.transform(
                [chunk_minx, chunk_maxx, chunk_minx, chunk_maxx],
                [chunk_maxy, chunk_maxy, chunk_miny, chunk_miny],
            )
            bbox_native = (min(xs), min(ys), max(xs), max(ys))

        window = _native_window(reader, bbox_native, reader.width, reader.height)
        if window is None:
            logger.debug(
                "Item %s band %r does not overlap chunk; skipping.",
                item.get("id"),
                band,
            )
            continue

        band_read_plan.append((band, geotiff, reader, window, effective_nodata))

    if not band_read_plan:
        return None

    # Read all windows concurrently.
    async def _read_band(
        band: str, reader: object, window: Window
    ) -> tuple[str, object]:
        return band, await reader.read(window=window)

    read_results = await asyncio.gather(
        *[_read_band(b, r, w) for b, _, r, w, _ in band_read_plan]
    )

    effective_nodatas = {b: n for b, _, _, _, n in band_read_plan}
    crss = {b: g.crs for b, g, _, _, _ in band_read_plan}

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
    store: object | None = None,
    max_concurrent_reads: int = 32,
    warp_cache: dict | None = None,
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

    Returns:
        ``dict`` mapping each band name to an array of shape
        ``(cog_bands, chunk_height, chunk_width)`` with dtype matching the
        source COGs.

    """
    if mosaic_method_cls is None:
        mosaic_method_cls = FirstMethod

    logger.debug(
        "async_mosaic_chunk_multiband bands=%r %dx%d px, %d items (max_concurrent_reads=%d)",
        bands,
        chunk_width,
        chunk_height,
        len(items),
        max_concurrent_reads,
    )

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

    t0 = time.perf_counter()
    done = False
    items_read = 0
    for batch_start in range(0, len(items), max_concurrent_reads):
        batch = items[batch_start : batch_start + max_concurrent_reads]
        batch_results = await asyncio.gather(
            *[_guarded(item) for item in batch],
            return_exceptions=True,
        )
        items_read += len(batch)

        for j, result in enumerate(batch_results):
            item_idx = batch_start + j
            if isinstance(result, BaseException):
                item_id = items[item_idx].get("id", "<unknown>")
                logger.warning(
                    "Failed to read bands %r from item %s: %s",
                    bands,
                    item_id,
                    result,
                    exc_info=result,
                )
                continue

            if result is None:
                continue

            for band, (arr, effective_nodata) in result.items():
                arr_mask: np.ndarray
                if effective_nodata is not None:
                    arr_mask = np.all(arr == effective_nodata, axis=0, keepdims=True)
                    arr_mask = np.broadcast_to(arr_mask, arr.shape).copy()
                else:
                    arr_mask = np.zeros(arr.shape, dtype=bool)
                mosaic_methods[band].feed(ma.MaskedArray(arr, mask=arr_mask))

            if all(m.is_done for m in mosaic_methods.values()):
                done = True
                break

        if done:
            break

    logger.debug(
        "async_mosaic_chunk_multiband bands=%r read %d/%d items in %.3fs",
        bands,
        items_read,
        len(items),
        time.perf_counter() - t0,
    )

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
