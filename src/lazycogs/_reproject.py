"""Reproject raster arrays using a backend-neutral request interface."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import numpy as np
from pyproj import CRS, Transformer

from lazycogs._rust_warp import reproject_array_rust_warp

if TYPE_CHECKING:
    from affine import Affine


class ResamplingMethod(StrEnum):
    """Supported public reprojection resampling methods."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"


@functools.lru_cache(maxsize=256)
def _get_transformer(src_crs: CRS, dst_crs: CRS) -> Transformer:
    """Return a cached ``Transformer`` for a CRS pair.

    ``Transformer.from_crs`` involves PROJ database lookups and pipeline
    initialisation.  The same (src_crs, dst_crs) pair recurs for every item
    in a collection, so caching avoids recreating the same object hundreds of
    times per chunk read.  ``pyproj.CRS`` is hashable via its WKT
    representation, and ``Transformer`` is thread-safe from PROJ 6+.

    Args:
        src_crs: Source CRS.
        dst_crs: Destination CRS.

    Returns:
        A ``Transformer`` that maps ``src_crs`` → ``dst_crs``.

    """
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


@dataclass(frozen=True)
class ReprojectRequest:
    """All inputs required to reproject one source tile.

    Attributes:
        data: Source data with shape ``(bands, src_h, src_w)``.
        src_transform: Affine transform of the source array.
        src_crs: CRS of the source array.
        dst_transform: Affine transform of the destination grid.
        dst_crs: CRS of the destination grid.
        dst_width: Destination width in pixels.
        dst_height: Destination height in pixels.
        nodata: Fill value for pixels that fall outside the source extent.
        resampling: Requested resampling method.

    """

    data: np.ndarray
    src_transform: Affine
    src_crs: CRS
    dst_transform: Affine
    dst_crs: CRS
    dst_width: int
    dst_height: int
    nodata: float | None = None
    resampling: ResamplingMethod = ResamplingMethod.NEAREST


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

    transformer = _get_transformer(dst_crs, src_crs)
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


def _same_grid(request: ReprojectRequest) -> bool:
    """Return ``True`` when reprojection is an exact no-op."""
    return (
        request.src_crs.equals(request.dst_crs)
        and request.src_transform == request.dst_transform
        and request.data.shape[1] == request.dst_height
        and request.data.shape[2] == request.dst_width
    )


def _legacy_cache_key(request: ReprojectRequest) -> tuple[tuple[float, ...], CRS]:
    """Return the cache key for the legacy nearest-neighbor backend."""
    return (tuple(request.src_transform), request.src_crs)


def _reproject_tile_legacy(
    request: ReprojectRequest,
    warp_map: WarpMap | None = None,
) -> np.ndarray:
    """Reproject a tile with the legacy pyproj/numpy nearest backend."""
    if request.resampling != "nearest":
        raise ValueError(
            "The legacy reprojection backend only supports resampling='nearest'.",
        )

    resolved_warp_map = warp_map or compute_warp_map(
        request.src_transform,
        request.src_crs,
        request.dst_transform,
        request.dst_crs,
        request.dst_width,
        request.dst_height,
    )
    return apply_warp_map(request.data, resolved_warp_map, request.nodata)


_DEFAULT_REPROJECT_BACKEND: Literal["legacy", "rust-warp"] = "rust-warp"


def reproject_tile(
    request: ReprojectRequest,
    *,
    backend: Literal["legacy", "rust-warp"] | None = None,
    warp_cache: dict[tuple[tuple[float, ...], CRS], WarpMap] | None = None,
) -> np.ndarray:
    """Reproject one ``(bands, y, x)`` source tile onto a destination grid.

    Args:
        request: Reprojection inputs for one tile.
        backend: Internal backend selector. When ``None``, the module's
            current default backend is used.
        warp_cache: Optional cache for the legacy backend's precomputed warp
            maps, keyed by source transform and CRS. Ignored by rust-warp.

    Returns:
        Reprojected array with shape ``(bands, dst_height, dst_width)``.

    """
    if _same_grid(request):
        return request.data

    resolved_backend = _DEFAULT_REPROJECT_BACKEND if backend is None else backend
    if resolved_backend == "rust-warp":
        return reproject_array_rust_warp(
            data=request.data,
            src_transform=request.src_transform,
            src_crs=request.src_crs,
            dst_transform=request.dst_transform,
            dst_crs=request.dst_crs,
            dst_width=request.dst_width,
            dst_height=request.dst_height,
            nodata=request.nodata,
            resampling=request.resampling,
        )
    if resolved_backend != "legacy":
        raise ValueError(f"Unsupported reprojection backend: {resolved_backend}")

    warp_map: WarpMap | None = None
    if warp_cache is not None:
        cache_key = _legacy_cache_key(request)
        warp_map = warp_cache.get(cache_key)
        if warp_map is None:
            warp_map = compute_warp_map(
                request.src_transform,
                request.src_crs,
                request.dst_transform,
                request.dst_crs,
                request.dst_width,
                request.dst_height,
            )
            warp_cache[cache_key] = warp_map

    return _reproject_tile_legacy(request, warp_map)


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

    Convenience wrapper around :class:`ReprojectRequest` and
    :func:`reproject_tile`.  Use :func:`reproject_tile` directly for the new
    backend-neutral path.

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
    return reproject_tile(
        ReprojectRequest(
            data=data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_width=dst_width,
            dst_height=dst_height,
            nodata=nodata,
        ),
    )
