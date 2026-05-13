"""Private adapter for rust-warp's low-level reprojection API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rust_warp

if TYPE_CHECKING:
    from affine import Affine
    from pyproj import CRS

_SUPPORTED_DTYPES = frozenset(
    {
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int8),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.int16),
    },
)
_EXPECTED_ARRAY_NDIM = 3


def _affine_to_rust_warp(
    transform: Affine,
) -> tuple[float, float, float, float, float, float]:
    """Return affine coefficients in rust-warp's 6-value rasterio order."""
    return (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )


def _normalize_crs(crs: CRS) -> str:
    """Return a rust-warp-compatible CRS string for ``crs``.

    Prefers ``EPSG:<code>`` when ``pyproj`` can resolve one, then falls back to
    the CRS's PROJ string representation.
    """
    epsg = crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"

    proj4 = crs.to_proj4().strip()
    if proj4:
        return proj4

    raise ValueError(f"Could not normalize CRS {crs!r} to an EPSG or PROJ string.")


def _validate_supported_dtype(data: np.ndarray) -> np.dtype:
    """Return ``data.dtype`` when rust-warp supports it, else raise ``TypeError``."""
    dtype = data.dtype
    if dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(sorted(dt.name for dt in _SUPPORTED_DTYPES))
        raise TypeError(
            "rust-warp does not support dtype "
            f"{dtype.name!r}. Supported dtypes: {supported}.",
        )
    return dtype


def _normalize_nodata(nodata: float | None, dtype: np.dtype) -> float | int:
    """Cast ``nodata`` to the source dtype, defaulting to lazycogs' zero fill.

    rust-warp uses ``NaN`` as the implicit fill for floating-point arrays when
    ``nodata`` is omitted. lazycogs has historically treated ``nodata=None`` as
    a request for zero fill regardless of dtype, so normalize that explicitly
    before dispatch.
    """
    if nodata is None:
        return np.array([0]).astype(dtype, casting="unsafe")[0].item()
    return np.array([nodata]).astype(dtype, casting="unsafe")[0].item()


def reproject_array_rust_warp(
    data: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    dst_transform: Affine,
    dst_crs: CRS,
    dst_width: int,
    dst_height: int,
    nodata: float | None = None,
    resampling: str = "nearest",
) -> np.ndarray:
    """Reproject a ``(bands, y, x)`` array via rust-warp's 2D kernel.

    Args:
        data: Source array with shape ``(bands, src_h, src_w)``.
        src_transform: Affine transform of the source array.
        src_crs: CRS of the source array.
        dst_transform: Affine transform of the destination grid.
        dst_crs: CRS of the destination grid.
        dst_width: Destination width in pixels.
        dst_height: Destination height in pixels.
        nodata: Fill value for pixels outside the source extent.
        resampling: rust-warp resampling method name.

    Returns:
        Reprojected array with shape ``(bands, dst_height, dst_width)``.

    Raises:
        ValueError: If ``data`` is not 3D or the CRS cannot be normalized.
        TypeError: If the input dtype is unsupported by rust-warp.

    """
    if data.ndim != _EXPECTED_ARRAY_NDIM:
        raise ValueError(
            "rust-warp adapter expects data with shape (bands, src_height, src_width).",
        )

    dtype = _validate_supported_dtype(data)
    src_crs_str = _normalize_crs(src_crs)
    dst_crs_str = _normalize_crs(dst_crs)
    src_transform_tuple = _affine_to_rust_warp(src_transform)
    dst_transform_tuple = _affine_to_rust_warp(dst_transform)
    dst_shape = (dst_height, dst_width)
    normalized_nodata = _normalize_nodata(nodata, dtype)

    reprojected_bands = [
        rust_warp.reproject_array(
            src=band,
            src_crs=src_crs_str,
            src_transform=src_transform_tuple,
            dst_crs=dst_crs_str,
            dst_transform=dst_transform_tuple,
            dst_shape=dst_shape,
            resampling=resampling,
            nodata=normalized_nodata,
        )
        for band in data
    ]
    return np.stack(reprojected_bands, axis=0)
