"""Reproject raster arrays using pyproj and numpy nearest-neighbor sampling."""

from __future__ import annotations

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer


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

    Builds a dense warp map by transforming every destination pixel centre
    into the source CRS with a single vectorised ``Transformer.transform``
    call, then samples the source array with numpy fancy indexing.

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
    bands, src_h, src_w = data.shape
    fill = nodata if nodata is not None else 0

    # Grid of destination pixel-centre coordinates in dst_crs.
    col_idx = np.arange(dst_width)
    row_idx = np.arange(dst_height)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)  # (dst_h, dst_w)

    dst_xs = dst_transform.c + (col_grid + 0.5) * dst_transform.a
    dst_ys = dst_transform.f + (row_grid + 0.5) * dst_transform.e

    # Transform pixel centres from dst_crs to src_crs in one vectorised call.
    transformer = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    src_xs, src_ys = transformer.transform(dst_xs.ravel(), dst_ys.ravel())

    # Convert src-CRS coordinates to fractional src pixel coordinates.
    # ~src_transform maps (x, y) → (col_frac, row_frac) where the top-left
    # corner of pixel (j, i) is at fractional coords (j, i).
    inv = ~src_transform
    frac_cols = (inv.a * src_xs + inv.b * src_ys + inv.c).reshape(dst_height, dst_width)
    frac_rows = (inv.d * src_xs + inv.e * src_ys + inv.f).reshape(dst_height, dst_width)

    src_col_idx = np.floor(frac_cols).astype(np.intp)
    src_row_idx = np.floor(frac_rows).astype(np.intp)

    valid = (
        (src_col_idx >= 0)
        & (src_col_idx < src_w)
        & (src_row_idx >= 0)
        & (src_row_idx < src_h)
    )

    out = np.full((bands, dst_height, dst_width), fill, dtype=data.dtype)
    out[:, valid] = data[:, src_row_idx[valid], src_col_idx[valid]]

    return out
