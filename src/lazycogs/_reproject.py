"""Reproject raster arrays using pyproj and numpy nearest-neighbor sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer


@dataclass
class WarpMap:
    """Precomputed pixel-coordinate mapping from a destination grid to a source grid.

    Stores the source column and row index for every destination pixel centre,
    computed by a single vectorised ``Transformer.transform`` call.  The ``valid``
    mask is not stored here; ``apply_warp_map`` derives it from the actual source
    array shape so the same ``WarpMap`` can be reused across bands that share the
    same source CRS and window transform but may have slightly different window
    dimensions due to rounding.

    Attributes:
        src_col_idx: Source column indices, shape ``(dst_height, dst_width)``,
            dtype ``intp``.  May contain out-of-bounds values for pixels that
            map outside the source extent.
        src_row_idx: Source row indices, shape ``(dst_height, dst_width)``,
            dtype ``intp``.

    """

    src_col_idx: np.ndarray
    src_row_idx: np.ndarray


def compute_warp_map(
    src_transform: Affine,
    src_crs: CRS,
    dst_transform: Affine,
    dst_crs: CRS,
    dst_width: int,
    dst_height: int,
) -> WarpMap:
    """Build a pixel-coordinate mapping from destination grid to source grid.

    Transforms every destination pixel centre into the source CRS with a single
    vectorised ``Transformer.transform`` call, then converts to fractional source
    pixel coordinates.  The result can be reused across multiple bands that share
    the same source CRS and window transform via :func:`apply_warp_map`.

    Args:
        src_transform: Affine transform of the source array (window transform).
        src_crs: CRS of the source array.
        dst_transform: Affine transform of the destination grid.
        dst_crs: CRS of the destination grid.
        dst_width: Width of the destination grid in pixels.
        dst_height: Height of the destination grid in pixels.

    Returns:
        :class:`WarpMap` with ``src_col_idx`` and ``src_row_idx`` arrays of
        shape ``(dst_height, dst_width)``.

    """
    col_idx = np.arange(dst_width)
    row_idx = np.arange(dst_height)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)

    dst_xs = dst_transform.c + (col_grid + 0.5) * dst_transform.a
    dst_ys = dst_transform.f + (row_grid + 0.5) * dst_transform.e

    transformer = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    src_xs, src_ys = transformer.transform(dst_xs.ravel(), dst_ys.ravel())

    # ~src_transform maps (x, y) → (col_frac, row_frac).
    inv = ~src_transform
    frac_cols = (inv.a * src_xs + inv.b * src_ys + inv.c).reshape(dst_height, dst_width)
    frac_rows = (inv.d * src_xs + inv.e * src_ys + inv.f).reshape(dst_height, dst_width)

    return WarpMap(
        src_col_idx=np.floor(frac_cols).astype(np.intp),
        src_row_idx=np.floor(frac_rows).astype(np.intp),
    )


def apply_warp_map(
    data: np.ndarray,
    warp_map: WarpMap,
    nodata: float | None = None,
) -> np.ndarray:
    """Sample a source array using a precomputed :class:`WarpMap`.

    The valid mask is derived from ``data.shape`` at call time so the same
    ``warp_map`` can be safely applied to bands with slightly different window
    dimensions.

    Args:
        data: Source data with shape ``(bands, src_h, src_w)``.
        warp_map: Pixel-coordinate mapping from destination to source.
        nodata: Fill value for destination pixels that fall outside the source
            extent, or ``None`` to use zero.

    Returns:
        Array with shape ``(bands, dst_height, dst_width)`` and the same dtype
        as ``data``.

    """
    bands, src_h, src_w = data.shape
    dst_height, dst_width = warp_map.src_col_idx.shape
    fill = nodata if nodata is not None else 0

    valid = (
        (warp_map.src_col_idx >= 0)
        & (warp_map.src_col_idx < src_w)
        & (warp_map.src_row_idx >= 0)
        & (warp_map.src_row_idx < src_h)
    )

    out = np.full((bands, dst_height, dst_width), fill, dtype=data.dtype)
    out[:, valid] = data[:, warp_map.src_row_idx[valid], warp_map.src_col_idx[valid]]
    return out


def reproject_array(
    data: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    dst_transform: Affine,
    dst_crs: CRS,
    dst_width: int,
    dst_height: int,
    nodata: float | None = None,
) -> np.ndarray:
    """Reproject a raster array using nearest-neighbor sampling.

    Convenience wrapper around :func:`compute_warp_map` and
    :func:`apply_warp_map`.  Use those functions directly when the same source
    CRS and window transform are shared across multiple bands, so the warp map
    can be computed once and reused.

    Args:
        data: Source data with shape ``(bands, src_h, src_w)``.
        src_transform: Affine transform of the source array.
        src_crs: CRS of the source array.
        dst_transform: Affine transform of the destination grid.
        dst_crs: CRS of the destination grid.
        dst_width: Width of the output array in pixels.
        dst_height: Height of the output array in pixels.
        nodata: Value to use for destination pixels that fall outside the
            source extent, or ``None`` to use zero.

    Returns:
        Reprojected array with shape ``(bands, dst_height, dst_width)`` and
        the same dtype as ``data``.

    """
    warp_map = compute_warp_map(
        src_transform, src_crs, dst_transform, dst_crs, dst_width, dst_height
    )
    return apply_warp_map(data, warp_map, nodata)
