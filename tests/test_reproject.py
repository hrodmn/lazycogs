"""Tests for _reproject: reproject_array, compute_warp_map, apply_warp_map."""

from unittest.mock import patch

import numpy as np
import pytest
from affine import Affine
from pyproj import CRS

from lazycogs._reproject import (
    ReprojectRequest,
    ResamplingMethod,
    WarpMap,
    apply_warp_map,
    compute_warp_map,
    reproject_array,
    reproject_tile,
)
from lazycogs._rust_warp import _affine_to_rust_warp, _normalize_crs


@pytest.fixture
def wgs84() -> CRS:
    return CRS.from_epsg(4326)


@pytest.fixture
def utm32n() -> CRS:
    return CRS.from_epsg(32632)


def _make_transform(minx: float, maxy: float, res: float) -> Affine:
    return Affine(res, 0.0, minx, 0.0, -res, maxy)


def test_identity_same_crs_same_transform(wgs84):
    """Reprojecting to the identical grid returns the same values."""
    transform = _make_transform(0.0, 3.0, 1.0)
    data = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    out = reproject_array(data, transform, wgs84, transform, wgs84, 3, 3)
    np.testing.assert_array_equal(out, data)


def test_output_shape(wgs84):
    """Output shape matches (bands, dst_height, dst_width)."""
    src_transform = _make_transform(0.0, 2.0, 1.0)
    dst_transform = _make_transform(0.0, 4.0, 2.0)
    data = np.ones((2, 2, 2), dtype=np.float32)
    out = reproject_array(data, src_transform, wgs84, dst_transform, wgs84, 1, 2)
    assert out.shape == (2, 2, 1)


def test_reproject_tile_same_grid_returns_original_array(wgs84):
    """The backend-neutral dispatcher short-circuits exact same-grid reads."""
    transform = _make_transform(0.0, 3.0, 1.0)
    data = np.arange(9, dtype=np.float32).reshape(1, 3, 3)

    out = reproject_tile(
        ReprojectRequest(
            data=data,
            src_transform=transform,
            src_crs=wgs84,
            dst_transform=transform,
            dst_crs=wgs84,
            dst_width=3,
            dst_height=3,
        ),
    )

    assert out is data


def test_reproject_tile_matches_legacy_wrapper(wgs84):
    """The backend-neutral path preserves current nearest-neighbor behavior."""
    src_transform = _make_transform(0.0, 2.0, 1.0)
    dst_transform = _make_transform(0.0, 4.0, 2.0)
    data = np.ones((2, 2, 2), dtype=np.float32)

    request = ReprojectRequest(
        data=data,
        src_transform=src_transform,
        src_crs=wgs84,
        dst_transform=dst_transform,
        dst_crs=wgs84,
        dst_width=1,
        dst_height=2,
        nodata=-9999.0,
    )

    np.testing.assert_array_equal(
        reproject_tile(request),
        reproject_array(
            data,
            src_transform,
            wgs84,
            dst_transform,
            wgs84,
            1,
            2,
            nodata=-9999.0,
        ),
    )


def test_reproject_tile_defaults_to_rust_warp_backend(wgs84):
    """The default backend selection now routes nearest through rust-warp."""
    src_transform = _make_transform(0.0, 2.0, 1.0)
    dst_transform = _make_transform(0.0, 2.0, 0.5)
    data = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    calls: list[ResamplingMethod] = []

    def _fake_rust_warp(**kwargs):
        calls.append(kwargs["resampling"])
        return np.zeros((1, 4, 4), dtype=np.float32)

    with patch(
        "lazycogs._reproject.reproject_array_rust_warp",
        side_effect=_fake_rust_warp,
    ):
        out = reproject_tile(
            ReprojectRequest(
                data=data,
                src_transform=src_transform,
                src_crs=wgs84,
                dst_transform=dst_transform,
                dst_crs=wgs84,
                dst_width=4,
                dst_height=4,
                resampling=ResamplingMethod.NEAREST,
            ),
        )

    assert calls == [ResamplingMethod.NEAREST]
    assert out.shape == (1, 4, 4)


def test_affine_to_rust_warp_uses_six_value_rasterio_order():
    """Affine conversion emits the 6-tuple rust-warp expects."""
    transform = Affine(2.0, 0.5, 10.0, -0.25, -3.0, 20.0)

    assert _affine_to_rust_warp(transform) == (2.0, 0.5, 10.0, -0.25, -3.0, 20.0)


def test_normalize_crs_prefers_epsg_strings(wgs84):
    """EPSG-backed CRSes normalize to ``EPSG:<code>`` strings."""
    assert _normalize_crs(wgs84) == "EPSG:4326"


def test_normalize_crs_falls_back_to_proj_string():
    """Non-EPSG CRSes fall back to a usable PROJ string."""
    custom_crs = CRS.from_proj4(
        "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
    )

    normalized = _normalize_crs(custom_crs)

    assert normalized.startswith("+proj=aea")
    assert "+lat_1=29.5" in normalized


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.int8, np.uint8, np.uint16, np.int16],
)
def test_reproject_tile_rust_warp_supports_expected_dtypes(wgs84, dtype):
    """Supported dtypes pass through the rust-warp backend unchanged."""
    src_transform = _make_transform(0.0, 2.0, 1.0)
    dst_transform = _make_transform(0.0, 2.0, 0.5)
    data = np.arange(4, dtype=dtype).reshape(1, 2, 2)

    out = reproject_tile(
        ReprojectRequest(
            data=data,
            src_transform=src_transform,
            src_crs=wgs84,
            dst_transform=dst_transform,
            dst_crs=wgs84,
            dst_width=4,
            dst_height=4,
            nodata=-1.0,
        ),
        backend="rust-warp",
    )

    assert out.shape == (1, 4, 4)
    assert out.dtype == data.dtype


def test_reproject_tile_rust_warp_rejects_unsupported_dtype(wgs84):
    """Unsupported dtypes fail deterministically before async chunk execution."""
    src_transform = _make_transform(0.0, 2.0, 1.0)
    dst_transform = _make_transform(0.0, 2.0, 0.5)
    data = np.arange(4, dtype=np.int32).reshape(1, 2, 2)

    with pytest.raises(TypeError, match="does not support dtype"):
        reproject_tile(
            ReprojectRequest(
                data=data,
                src_transform=src_transform,
                src_crs=wgs84,
                dst_transform=dst_transform,
                dst_crs=wgs84,
                dst_width=4,
                dst_height=4,
            ),
            backend="rust-warp",
        )


def test_reproject_tile_rust_warp_preserves_band_order(wgs84):
    """Band-plane iteration keeps shape and band ordering intact."""
    src_transform = _make_transform(0.0, 2.0, 1.0)
    dst_transform = _make_transform(0.0, 2.0, 0.5)
    data = np.stack(
        [
            np.full((2, 2), 10.0, dtype=np.float32),
            np.full((2, 2), 20.0, dtype=np.float32),
            np.full((2, 2), 30.0, dtype=np.float32),
        ],
    )

    out = reproject_tile(
        ReprojectRequest(
            data=data,
            src_transform=src_transform,
            src_crs=wgs84,
            dst_transform=dst_transform,
            dst_crs=wgs84,
            dst_width=4,
            dst_height=4,
            nodata=-9999.0,
        ),
        backend="rust-warp",
    )

    assert out.shape == (3, 4, 4)
    np.testing.assert_array_equal(out[0], 10.0)
    np.testing.assert_array_equal(out[1], 20.0)
    np.testing.assert_array_equal(out[2], 30.0)


def test_out_of_bounds_pixels_get_nodata(wgs84):
    """Destination pixels outside the source extent are filled with nodata."""
    src_transform = _make_transform(5.0, 5.0, 1.0)  # covers x=5..8, y=2..5
    data = np.ones((1, 3, 3), dtype=np.float32)
    # Destination covers x=0..3, entirely outside source
    dst_transform = _make_transform(0.0, 3.0, 1.0)
    out = reproject_array(
        data,
        src_transform,
        wgs84,
        dst_transform,
        wgs84,
        3,
        3,
        nodata=-9999.0,
    )
    np.testing.assert_array_equal(out, -9999.0)


def test_out_of_bounds_default_fill_is_zero(wgs84):
    """When nodata is None, out-of-bounds pixels default to zero."""
    src_transform = _make_transform(100.0, 100.0, 1.0)
    data = np.ones((1, 2, 2), dtype=np.float32)
    dst_transform = _make_transform(0.0, 2.0, 1.0)
    out = reproject_array(data, src_transform, wgs84, dst_transform, wgs84, 2, 2)
    np.testing.assert_array_equal(out, 0.0)


def test_dtype_preserved(wgs84):
    """Output dtype matches source dtype."""
    transform = _make_transform(0.0, 2.0, 1.0)
    for dtype in (np.uint8, np.int16, np.float64):
        data = np.zeros((1, 2, 2), dtype=dtype)
        out = reproject_array(data, transform, wgs84, transform, wgs84, 2, 2)
        assert out.dtype == dtype


def test_multiband_preserved(wgs84):
    """All bands are reprojected independently."""
    transform = _make_transform(0.0, 2.0, 1.0)
    data = np.stack(
        [np.ones((2, 2), dtype=np.float32) * b for b in range(4)],
    )  # shape (4, 2, 2)
    out = reproject_array(data, transform, wgs84, transform, wgs84, 2, 2)
    assert out.shape == (4, 2, 2)
    for b in range(4):
        np.testing.assert_array_equal(out[b], b)


def test_cross_crs_reproject(wgs84, utm32n):
    """Reprojecting between WGS84 and UTM preserves values at matched pixels.

    We project a uniform field so that the exact pixel mapping doesn't matter —
    every source pixel has the same value, so any valid sample should match.
    """
    # UTM 32N chunk near central Europe: ~10 km at 1000 m resolution
    utm_transform = _make_transform(500_000.0, 5_550_000.0, 1000.0)
    data = np.full((1, 10, 10), 42.0, dtype=np.float32)

    # Destination grid in WGS84, centred over the UTM source extent
    # (which maps to roughly lon 9.0-9.14, lat 50.01-50.10)
    wgs84_transform = _make_transform(9.0, 50.1, 0.01)

    out = reproject_array(
        data,
        utm_transform,
        utm32n,
        wgs84_transform,
        wgs84,
        5,
        5,
        nodata=0.0,
    )
    # Any pixel that mapped back to a valid source location should be 42.
    valid_pixels = out[out != 0.0]
    assert len(valid_pixels) > 0
    np.testing.assert_array_equal(valid_pixels, 42.0)


def test_partial_overlap_nodata(wgs84):
    """Pixels that fall outside the source extent use nodata; overlapping ones copy."""
    # 4x1 source strip along x=0..4
    src_transform = _make_transform(0.0, 1.0, 1.0)
    data = np.full((1, 1, 4), 7.0, dtype=np.float32)

    # Destination covers x=2..6 — right half overlaps, left half does not
    dst_transform = _make_transform(2.0, 1.0, 1.0)
    out = reproject_array(
        data,
        src_transform,
        wgs84,
        dst_transform,
        wgs84,
        4,
        1,
        nodata=-1.0,
    )
    # x=2 and x=3 overlap source (values 7); x=4 and x=5 are outside
    np.testing.assert_array_equal(out[0, 0, :2], 7.0)
    np.testing.assert_array_equal(out[0, 0, 2:], -1.0)


# ---------------------------------------------------------------------------
# compute_warp_map / apply_warp_map
# ---------------------------------------------------------------------------


def test_compute_warp_map_returns_correct_shape(wgs84):
    """WarpMap arrays have shape (dst_height, dst_width)."""
    transform = _make_transform(0.0, 4.0, 1.0)
    wm = compute_warp_map(transform, wgs84, transform, wgs84, dst_width=4, dst_height=3)
    assert isinstance(wm, WarpMap)
    assert wm.src_col_idx.shape == (3, 4)
    assert wm.src_row_idx.shape == (3, 4)


def test_apply_warp_map_matches_reproject_array(wgs84):
    """apply_warp_map with a precomputed map matches reproject_array."""
    src_transform = _make_transform(0.0, 3.0, 1.0)
    dst_transform = _make_transform(0.0, 3.0, 1.0)
    data = np.arange(9, dtype=np.float32).reshape(1, 3, 3)

    wm = compute_warp_map(src_transform, wgs84, dst_transform, wgs84, 3, 3)
    out_warp = apply_warp_map(data, wm, nodata=0.0)
    out_reproject = reproject_array(
        data,
        src_transform,
        wgs84,
        dst_transform,
        wgs84,
        3,
        3,
        nodata=0.0,
    )
    np.testing.assert_array_equal(out_warp, out_reproject)


def test_apply_warp_map_reused_across_bands(wgs84):
    """A single WarpMap applied to two bands matches reproject_array per band."""
    transform = _make_transform(0.0, 2.0, 1.0)
    band_a = np.full((1, 2, 2), 1.0, dtype=np.float32)
    band_b = np.full((1, 2, 2), 2.0, dtype=np.float32)

    wm = compute_warp_map(transform, wgs84, transform, wgs84, 2, 2)

    out_a = apply_warp_map(band_a, wm)
    out_b = apply_warp_map(band_b, wm)

    np.testing.assert_array_equal(
        out_a,
        reproject_array(band_a, transform, wgs84, transform, wgs84, 2, 2),
    )
    np.testing.assert_array_equal(
        out_b,
        reproject_array(band_b, transform, wgs84, transform, wgs84, 2, 2),
    )


def test_apply_warp_map_different_src_dimensions(wgs84):
    """apply_warp_map derives valid mask from actual data shape, not stored metadata."""
    # Compute warp map for a 4x4 source extent.
    src_transform = _make_transform(0.0, 4.0, 1.0)
    dst_transform = _make_transform(0.0, 4.0, 1.0)
    wm = compute_warp_map(src_transform, wgs84, dst_transform, wgs84, 4, 4)

    # Apply to a 3x3 source array — pixels that map to row/col >= 3 should use nodata.
    data_small = np.ones((1, 3, 3), dtype=np.float32)
    out = apply_warp_map(data_small, wm, nodata=-1.0)

    # Top-left 3x3 destination pixels map into the valid 3x3 source.
    np.testing.assert_array_equal(out[0, :3, :3], 1.0)
    # Bottom row and right column of destination map outside the 3x3 source.
    np.testing.assert_array_equal(out[0, 3, :], -1.0)
    np.testing.assert_array_equal(out[0, :, 3], -1.0)
