"""Tests for _reproject.reproject_array."""

import numpy as np
import pytest
from affine import Affine
from pyproj import CRS

from stac_cog_xarray._reproject import reproject_array


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


def test_out_of_bounds_pixels_get_nodata(wgs84):
    """Destination pixels outside the source extent are filled with nodata."""
    src_transform = _make_transform(5.0, 5.0, 1.0)  # covers x=5..8, y=2..5
    data = np.ones((1, 3, 3), dtype=np.float32)
    # Destination covers x=0..3, entirely outside source
    dst_transform = _make_transform(0.0, 3.0, 1.0)
    out = reproject_array(
        data, src_transform, wgs84, dst_transform, wgs84, 3, 3, nodata=-9999.0
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
        [np.ones((2, 2), dtype=np.float32) * b for b in range(4)]
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
    # (which maps to roughly lon 9.0–9.14, lat 50.01–50.10)
    wgs84_transform = _make_transform(9.0, 50.1, 0.01)

    out = reproject_array(
        data, utm_transform, utm32n, wgs84_transform, wgs84, 5, 5, nodata=0.0
    )
    # Any pixel that mapped back to a valid source location should be 42.
    valid_pixels = out[out != 0.0]
    assert len(valid_pixels) > 0
    np.testing.assert_array_equal(valid_pixels, 42.0)


def test_partial_overlap_nodata(wgs84):
    """Pixels that fall outside the source extent use nodata; overlapping ones copy."""
    # 4×1 source strip along x=0..4
    src_transform = _make_transform(0.0, 1.0, 1.0)
    data = np.full((1, 1, 4), 7.0, dtype=np.float32)

    # Destination covers x=2..6 — right half overlaps, left half does not
    dst_transform = _make_transform(2.0, 1.0, 1.0)
    out = reproject_array(
        data, src_transform, wgs84, dst_transform, wgs84, 4, 1, nodata=-1.0
    )
    # x=2 and x=3 overlap source (values 7); x=4 and x=5 are outside
    np.testing.assert_array_equal(out[0, 0, :2], 7.0)
    np.testing.assert_array_equal(out[0, 0, 2:], -1.0)
