"""Tests for _mosaic_methods."""

import numpy as np
import numpy.ma as ma
import pytest

from stac_cog_xarray._mosaic_methods import (
    CountMethod,
    FirstMethod,
    HighestMethod,
    LowestMethod,
    MeanMethod,
    MedianMethod,
    StdevMethod,
)


def _masked(data: list, mask: list) -> ma.MaskedArray:
    """Shorthand for building a (1, h, w) masked array from flat lists."""
    d = np.array(data, dtype=np.float32).reshape(1, 1, -1)
    m = np.array(mask, dtype=bool).reshape(1, 1, -1)
    return ma.MaskedArray(d, mask=m)


# ---------------------------------------------------------------------------
# Common contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls", [FirstMethod, HighestMethod, LowestMethod, MeanMethod, MedianMethod]
)
def test_data_raises_before_feed(cls):
    """`data` raises before any tile is fed."""
    m = cls()
    with pytest.raises(ValueError):
        _ = m.data


@pytest.mark.parametrize(
    "cls", [FirstMethod, HighestMethod, LowestMethod, MeanMethod, MedianMethod]
)
def test_is_done_false_before_feed(cls):
    """`is_done` is False before any tile is fed."""
    m = cls()
    assert not m.is_done


@pytest.mark.parametrize(
    "cls", [FirstMethod, HighestMethod, LowestMethod, MeanMethod, MedianMethod]
)
def test_fully_valid_tile_is_done(cls):
    """A single fully-valid tile makes `is_done` True."""
    m = cls()
    arr = ma.MaskedArray(np.ones((1, 2, 2), dtype=np.float32), mask=False)
    m.feed(arr)
    assert m.is_done


# ---------------------------------------------------------------------------
# FirstMethod
# ---------------------------------------------------------------------------


def test_first_method_takes_first_valid():
    """The first valid pixel at each position wins."""
    m = FirstMethod()
    # Tile 1: left pixel valid (1), right pixel masked
    m.feed(_masked([1.0, 0.0], [False, True]))
    # Tile 2: both pixels valid (9, 2) — right gap should be filled
    m.feed(_masked([9.0, 2.0], [False, False]))
    result = m.data
    assert result[0, 0, 0] == pytest.approx(1.0)  # kept from tile 1
    assert result[0, 0, 1] == pytest.approx(2.0)  # filled from tile 2


def test_first_method_is_done_after_gap_filled():
    m = FirstMethod()
    m.feed(_masked([1.0, 0.0], [False, True]))
    assert not m.is_done
    m.feed(_masked([0.0, 2.0], [True, False]))
    assert m.is_done


def test_first_method_does_not_overwrite_valid():
    """Later tiles do not overwrite pixels already filled."""
    m = FirstMethod()
    m.feed(_masked([5.0, 6.0], [False, False]))
    m.feed(_masked([99.0, 99.0], [False, False]))
    result = m.data
    np.testing.assert_array_equal(result[0, 0], [5.0, 6.0])


# ---------------------------------------------------------------------------
# HighestMethod
# ---------------------------------------------------------------------------


def test_highest_keeps_max():
    m = HighestMethod()
    m.feed(_masked([3.0, 1.0], [False, False]))
    m.feed(_masked([1.0, 5.0], [False, False]))
    result = m.data
    assert result[0, 0, 0] == pytest.approx(3.0)
    assert result[0, 0, 1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# LowestMethod
# ---------------------------------------------------------------------------


def test_lowest_keeps_min():
    m = LowestMethod()
    m.feed(_masked([3.0, 1.0], [False, False]))
    m.feed(_masked([1.0, 5.0], [False, False]))
    result = m.data
    assert result[0, 0, 0] == pytest.approx(1.0)
    assert result[0, 0, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MeanMethod
# ---------------------------------------------------------------------------


def test_mean_of_two_tiles():
    m = MeanMethod()
    m.feed(_masked([2.0, 4.0], [False, False]))
    m.feed(_masked([4.0, 8.0], [False, False]))
    result = m.data
    assert result[0, 0, 0] == pytest.approx(3.0)
    assert result[0, 0, 1] == pytest.approx(6.0)


def test_mean_ignores_masked_pixels():
    """Masked pixels do not contribute to the mean."""
    m = MeanMethod()
    # Pixel 0: only tile 1 valid → mean = 4
    # Pixel 1: both tiles valid → mean = (4+8)/2 = 6
    m.feed(_masked([4.0, 4.0], [False, False]))
    m.feed(_masked([99.0, 8.0], [True, False]))
    result = m.data
    assert result[0, 0, 0] == pytest.approx(4.0)
    assert result[0, 0, 1] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# MedianMethod
# ---------------------------------------------------------------------------


def test_median_of_three_tiles():
    m = MedianMethod()
    for v in [1.0, 3.0, 5.0]:
        m.feed(_masked([v], [False]))
    result = m.data
    assert result[0, 0, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# StdevMethod
# ---------------------------------------------------------------------------


def test_stdev_constant_field_is_zero():
    m = StdevMethod()
    for _ in range(3):
        m.feed(_masked([7.0], [False]))
    result = m.data
    assert result[0, 0, 0] == pytest.approx(0.0, abs=1e-5)


def test_stdev_known_values():
    m = StdevMethod()
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    for v in values:
        m.feed(_masked([v], [False]))
    expected_std = float(np.std(values))
    result = m.data
    assert result[0, 0, 0] == pytest.approx(expected_std, abs=1e-5)


# ---------------------------------------------------------------------------
# CountMethod
# ---------------------------------------------------------------------------


def test_count_method():
    m = CountMethod()
    # Pixel 0: valid in both; pixel 1: valid only in second
    m.feed(_masked([1.0, 1.0], [False, True]))
    m.feed(_masked([1.0, 1.0], [False, False]))
    result = m.data
    assert result[0, 0, 0] == 2
    assert result[0, 0, 1] == 1


def test_count_always_done():
    """CountMethod reports is_done=True after at least one fully-valid tile."""
    m = CountMethod()
    m.feed(ma.MaskedArray(np.ones((1, 2, 2), dtype=np.uint16), mask=False))
    assert m.is_done
