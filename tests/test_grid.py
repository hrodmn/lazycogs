"""Tests for _grid.compute_output_grid."""

import numpy as np
import pytest
from pyproj import CRS

from stac_cog_xarray._grid import compute_output_grid


@pytest.fixture
def wgs84() -> CRS:
    return CRS.from_epsg(4326)


def test_basic_dimensions(wgs84):
    """Width and height are derived from bbox / resolution."""
    transform, w, h, x, y = compute_output_grid((0.0, 0.0, 10.0, 5.0), wgs84, 1.0)
    assert w == 10
    assert h == 5


def test_affine_origin(wgs84):
    """Transform origin is the top-left corner of the top-left pixel."""
    transform, w, h, x, y = compute_output_grid((10.0, 20.0, 30.0, 40.0), wgs84, 1.0)
    assert transform.c == pytest.approx(10.0)
    assert transform.f == pytest.approx(40.0)
    assert transform.a == pytest.approx(1.0)
    assert transform.e == pytest.approx(-1.0)


def test_pixel_centres(wgs84):
    """x/y coordinate arrays hold pixel centres, not edges."""
    transform, w, h, x, y = compute_output_grid((0.0, 0.0, 4.0, 2.0), wgs84, 1.0)
    np.testing.assert_allclose(x, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(y, [0.5, 1.5])


def test_x_coords_increase(wgs84):
    """x coordinates increase left-to-right."""
    _, _, _, x, _ = compute_output_grid((0.0, 0.0, 10.0, 10.0), wgs84, 1.0)
    assert np.all(np.diff(x) > 0)


def test_y_coords_increase(wgs84):
    """y coordinates increase south-to-north so ascending slices work naturally."""
    _, _, _, _, y = compute_output_grid((0.0, 0.0, 10.0, 10.0), wgs84, 1.0)
    assert np.all(np.diff(y) > 0)


def test_coord_array_lengths(wgs84):
    """x and y arrays have lengths matching width and height."""
    transform, w, h, x, y = compute_output_grid((0.0, 0.0, 10.0, 5.0), wgs84, 1.0)
    assert len(x) == w
    assert len(y) == h


def test_small_bbox_rounds_to_one_pixel(wgs84):
    """A bbox smaller than one resolution step yields a 1×1 grid."""
    transform, w, h, x, y = compute_output_grid((0.0, 0.0, 0.1, 0.1), wgs84, 1.0)
    assert w == 1
    assert h == 1


def test_non_unit_resolution(wgs84):
    """Non-unit resolution produces correct pixel count and spacing."""
    transform, w, h, x, y = compute_output_grid((0.0, 0.0, 100.0, 50.0), wgs84, 10.0)
    assert w == 10
    assert h == 5
    assert transform.a == pytest.approx(10.0)
    assert transform.e == pytest.approx(-10.0)
    np.testing.assert_allclose(x[0], 5.0)
    np.testing.assert_allclose(y[0], 5.0)


def test_first_and_last_pixel_centres_span_bbox(wgs84):
    """First and last pixel centres lie half-pixel inside the bbox edges."""
    res = 2.0
    minx, miny, maxx, maxy = 0.0, 0.0, 10.0, 6.0
    _, w, h, x, y = compute_output_grid((minx, miny, maxx, maxy), wgs84, res)
    assert x[0] == pytest.approx(minx + res / 2)
    assert x[-1] == pytest.approx(maxx - res / 2)
    assert y[0] == pytest.approx(miny + res / 2)
    assert y[-1] == pytest.approx(maxy - res / 2)
