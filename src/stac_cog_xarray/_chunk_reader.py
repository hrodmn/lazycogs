"""Async mosaic logic: open COGs, read windows, reproject, and mosaic."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.ma as ma
from affine import Affine
from async_geotiff import GeoTIFF, Overview, Window
from pyproj import CRS, Transformer

from stac_cog_xarray._mosaic_methods import FirstMethod, MosaicMethodBase
from stac_cog_xarray._reproject import reproject_array
from stac_cog_xarray._store import path_from_href, store_from_href

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
            is resolved and cached via :func:`~stac_cog_xarray._store.store_from_href`.

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

    geotiff = await GeoTIFF.open(path, store=store)

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

    raster = await reader.read(window=window)

    # Reproject to the destination chunk grid.
    arr = reproject_array(
        data=raster.data,
        src_transform=raster.transform,
        src_crs=geotiff.crs,
        dst_transform=chunk_affine,
        dst_crs=dst_crs,
        dst_width=chunk_width,
        dst_height=chunk_height,
        nodata=effective_nodata,
    )
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
) -> np.ndarray:
    """Read, reproject, and mosaic a single chunk from multiple STAC items.

    Args:
        items: List of STAC item dicts to mosaic.  Processed in order.
        band: Asset key identifying the band to read from each item.
        chunk_affine: Affine transform of the destination chunk.
        dst_crs: CRS of the destination chunk.
        chunk_width: Width of the destination chunk in pixels.
        chunk_height: Height of the destination chunk in pixels.
        nodata: No-data fill value.
        mosaic_method: Pixel-selection strategy.  Defaults to
            :class:`~stac_cog_xarray._mosaic_methods.FirstMethod`.
        store: Optional pre-configured obstore ``ObjectStore`` instance
            forwarded to :func:`_read_item_band`.  When ``None``, each item's
            store is resolved from its HREF.

    Returns:
        Array of shape ``(bands, chunk_height, chunk_width)`` with dtype
        matching the source COGs.

    """
    if mosaic_method is None:
        mosaic_method = FirstMethod()

    # Read all items concurrently.
    results = await asyncio.gather(
        *[
            _read_item_band(
                item,
                band,
                chunk_affine,
                dst_crs,
                chunk_width,
                chunk_height,
                nodata,
                store=store,
            )
            for item in items
        ],
        return_exceptions=True,
    )

    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            item_id = items[i].get("id", "<unknown>")
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
            break

    if mosaic_method._mosaic is None:
        bands = 1
        fill = nodata if nodata is not None else 0
        return np.full((bands, chunk_height, chunk_width), fill, dtype=np.float32)

    return mosaic_method.data
