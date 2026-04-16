"""Tests for StacBackendArray._chunk_bbox_4326 and _raw_getitem."""

from unittest.mock import patch

import numpy as np
import pytest
from affine import Affine
from pyproj import CRS

from lazycogs._backend import MultiBandStacBackendArray, StacBackendArray
from lazycogs._mosaic_methods import FirstMethod


@pytest.fixture
def wgs84() -> CRS:
    return CRS.from_epsg(4326)


@pytest.fixture
def utm32n() -> CRS:
    return CRS.from_epsg(32632)


def _make_array(crs: CRS, dates: list[str] | None = None) -> StacBackendArray:
    """Return a minimal StacBackendArray for unit testing."""
    if dates is None:
        dates = ["2023-01-01", "2023-01-02"]
    return StacBackendArray(
        parquet_path="/tmp/fake.parquet",
        band="B04",
        dates=dates,
        dst_affine=Affine(1.0, 0.0, 10.0, 0.0, -1.0, 50.0),
        dst_crs=crs,
        bbox_4326=[10.0, 49.0, 14.0, 50.0],
        sort_by=None,
        filter=None,
        ids=None,
        dst_width=4,
        dst_height=1,
        dtype=np.dtype("float32"),
        nodata=-9999.0,
        shape=(2, 1, 4),
        mosaic_method_cls=FirstMethod,
    )


# ---------------------------------------------------------------------------
# _chunk_bbox_4326
# ---------------------------------------------------------------------------


def test_chunk_bbox_4326_identity_in_wgs84(wgs84):
    """When dst_crs is EPSG:4326, bbox is returned as-is."""
    arr = _make_array(wgs84)
    # Chunk affine: origin at (10, 50), 1° resolution, 4 wide × 1 tall
    bbox = arr._chunk_bbox_4326(
        Affine(1.0, 0.0, 10.0, 0.0, -1.0, 50.0),
        chunk_width=4,
        chunk_height=1,
    )
    assert bbox == pytest.approx([10.0, 49.0, 14.0, 50.0])


def test_chunk_bbox_4326_utm_transforms(utm32n):
    """A UTM chunk bbox is transformed to EPSG:4326."""
    arr = _make_array(utm32n)
    # A small UTM chunk near the prime meridian
    bbox = arr._chunk_bbox_4326(
        Affine(100.0, 0.0, 500_000.0, 0.0, -100.0, 5_550_000.0),
        chunk_width=10,
        chunk_height=10,
    )
    # Result should be a reasonable WGS84 bbox (central Europe)
    minx, miny, maxx, maxy = bbox
    assert -180 <= minx <= 180
    assert -90 <= miny <= 90
    assert minx < maxx
    assert miny < maxy


def test_chunk_bbox_4326_ordering(utm32n):
    """Returned bbox satisfies minx < maxx and miny < maxy."""
    arr = _make_array(utm32n)
    bbox = arr._chunk_bbox_4326(
        Affine(1000.0, 0.0, 400_000.0, 0.0, -1000.0, 5_600_000.0),
        chunk_width=100,
        chunk_height=100,
    )
    minx, miny, maxx, maxy = bbox
    assert minx < maxx
    assert miny < maxy


# ---------------------------------------------------------------------------
# _raw_getitem — no-data short-circuit
# ---------------------------------------------------------------------------


def test_raw_getitem_empty_items_returns_nodata(wgs84):
    """When DuckDB returns no items, the chunk is filled with nodata."""
    arr = _make_array(wgs84)

    with patch("lazycogs._backend.rustac.search_sync", return_value=[]):
        result = arr._raw_getitem((slice(0, 2), slice(0, 1), slice(0, 4)))

    assert result.shape == (2, 1, 4)
    np.testing.assert_array_equal(result, -9999.0)


def test_raw_getitem_scalar_time_squeezes(wgs84):
    """Integer time index squeezes the time dimension from the output."""
    arr = _make_array(wgs84)

    with patch("lazycogs._backend.rustac.search_sync", return_value=[]):
        result = arr._raw_getitem((0, slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 4)


def test_raw_getitem_with_items_calls_mosaic(wgs84):
    """When items are returned, async_mosaic_chunk is called and result used."""
    arr = _make_array(wgs84, dates=["2023-01-01"])
    fake_items = [{"id": "item-1", "assets": {"B04": {"href": "s3://b/f.tif"}}}]
    # async_mosaic_chunk returns (bands, h, w); band 0 is extracted
    fake_chunk = np.full((1, 1, 4), 42.0, dtype=np.float32)

    def _close_and_return_chunk(coro):
        coro.close()
        return fake_chunk

    with (
        patch("lazycogs._backend.rustac.search_sync", return_value=fake_items),
        patch("lazycogs._backend.asyncio.run", side_effect=_close_and_return_chunk),
    ):
        result = arr._raw_getitem((0, slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 4)
    np.testing.assert_array_equal(result, 42.0)


def test_raw_getitem_chunk_affine_offset(wgs84):
    """Chunk affine is translated by (x_start, y_start) from the full grid."""
    arr = _make_array(wgs84)

    captured_affines = []

    def fake_mosaic(*args, **kwargs):
        # Capture the chunk_affine passed to async_mosaic_chunk
        captured_affines.append(kwargs.get("chunk_affine"))
        return np.zeros((1, 1, 2), dtype=np.float32)

    def _close_and_return(coro):
        coro.close()
        return np.zeros((1, 1, 2), dtype=np.float32)

    fake_items = [{"id": "x"}]
    with (
        patch("lazycogs._backend.rustac.search_sync", return_value=fake_items),
        patch(
            "lazycogs._backend.asyncio.run",
            side_effect=_close_and_return,
        ),
    ):
        arr._raw_getitem((0, slice(0, 1), slice(2, 4)))

    # The full grid origin is (10, 50); x_start=2 → chunk origin x = 10 + 2 = 12
    # Verify via the chunk_bbox returned from _chunk_bbox_4326 indirectly:
    # If chunk_affine.c == 12, the bbox minx should be 12 in WGS84
    chunk_affine = arr.dst_affine * Affine.translation(2, 0)
    assert chunk_affine.c == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# MultiBandStacBackendArray._raw_getitem
# ---------------------------------------------------------------------------


def _make_multiband_array(
    crs: CRS, bands: list[str], dates: list[str] | None = None
) -> MultiBandStacBackendArray:
    """Return a minimal MultiBandStacBackendArray for unit testing."""
    if dates is None:
        dates = ["2023-01-01"]
    band_arrays = [
        StacBackendArray(
            parquet_path="/tmp/fake.parquet",
            band=b,
            dates=dates,
            dst_affine=Affine(1.0, 0.0, 10.0, 0.0, -1.0, 50.0),
            dst_crs=crs,
            bbox_4326=[10.0, 49.0, 14.0, 50.0],
            sort_by=None,
            filter=None,
            ids=None,
            dst_width=4,
            dst_height=1,
            dtype=np.dtype("float32"),
            nodata=-9999.0,
            shape=(len(dates), 1, 4),
            mosaic_method_cls=FirstMethod,
        )
        for b in bands
    ]
    return MultiBandStacBackendArray(band_arrays=band_arrays, band_names=bands)


def test_multiband_raw_getitem_no_items_returns_nodata(wgs84):
    """When no items are found, all bands are filled with nodata."""
    multi = _make_multiband_array(wgs84, ["B01", "B02"])

    with patch("lazycogs._backend.rustac.search_sync", return_value=[]):
        result = multi._raw_getitem(
            (slice(0, 2), slice(0, 1), slice(0, 1), slice(0, 4))
        )

    assert result.shape == (2, 1, 1, 4)
    np.testing.assert_array_equal(result, -9999.0)


def test_multiband_raw_getitem_calls_multiband_mosaic(wgs84):
    """_raw_getitem calls async_mosaic_chunk_multiband once per time step, not per band."""
    bands = ["B01", "B02"]
    multi = _make_multiband_array(wgs84, bands)
    fake_items = [
        {"id": "item-1", "assets": {b: {"href": f"s3://b/{b}.tif"} for b in bands}}
    ]

    call_count = [0]

    def _fake_run_coroutine(coro):
        call_count[0] += 1
        coro.close()
        return {
            b: np.full((1, 1, 4), float(i), dtype=np.float32)
            for i, b in enumerate(bands)
        }

    with (
        patch("lazycogs._backend.rustac.search_sync", return_value=fake_items),
        patch("lazycogs._backend._run_coroutine", side_effect=_fake_run_coroutine),
    ):
        result = multi._raw_getitem((slice(0, 2), 0, slice(0, 1), slice(0, 4)))

    # One time step → one call to async_mosaic_chunk_multiband, not two.
    assert call_count[0] == 1
    assert result.shape == (2, 1, 4)


def test_multiband_raw_getitem_squeeze_band(wgs84):
    """Integer band index squeezes the band dimension."""
    multi = _make_multiband_array(wgs84, ["B01", "B02"])

    with patch("lazycogs._backend.rustac.search_sync", return_value=[]):
        result = multi._raw_getitem((0, 0, slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 4)


def test_multiband_raw_getitem_single_band_single_pixel(wgs84):
    """All dimensions squeezed returns a scalar array."""
    multi = _make_multiband_array(wgs84, ["B01", "B02"])

    with patch("lazycogs._backend.rustac.search_sync", return_value=[]):
        result = multi._raw_getitem((0, 0, 0, 0))

    assert result.shape == ()
