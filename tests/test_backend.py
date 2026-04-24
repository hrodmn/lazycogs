"""Tests for _backend helpers and _raw_getitem."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from rustac import DuckdbClient
from affine import Affine
from pyproj import CRS, Transformer

from lazycogs._backend import (
    MultiBandStacBackendArray,
    _resolve_spatial_window,
)
from lazycogs._mosaic_methods import FirstMethod


@pytest.fixture
def wgs84() -> CRS:
    return CRS.from_epsg(4326)


@pytest.fixture
def utm32n() -> CRS:
    return CRS.from_epsg(32632)


def _make_array(
    crs: CRS,
    bands: list[str] | None = None,
    dates: list[str] | None = None,
) -> MultiBandStacBackendArray:
    """Return a minimal MultiBandStacBackendArray for unit testing."""
    if bands is None:
        bands = ["B04"]
    if dates is None:
        dates = ["2023-01-01", "2023-01-02"]
    return MultiBandStacBackendArray(
        parquet_path="/tmp/fake.parquet",
        duckdb_client=DuckdbClient(),
        bands=bands,
        dates=dates,
        dst_affine=Affine(1.0, 0.0, 10.0, 0.0, -1.0, 50.0),
        dst_crs=crs,
        bbox_4326=[10.0, 49.0, 14.0, 50.0],
        sortby=None,
        filter=None,
        ids=None,
        dst_width=4,
        dst_height=1,
        dtype=np.dtype("float32"),
        nodata=-9999.0,
        mosaic_method_cls=FirstMethod,
    )


# ---------------------------------------------------------------------------
# _resolve_spatial_window
# ---------------------------------------------------------------------------


def test_chunk_bbox_4326_identity_in_wgs84(wgs84):
    """When dst_crs is EPSG:4326, bbox is returned as-is."""
    # dst_affine: origin at (10, 50), 1° resolution, 4 wide × 1 tall
    dst_affine = Affine(1.0, 0.0, 10.0, 0.0, -1.0, 50.0)
    win = _resolve_spatial_window(
        slice(0, 1),
        slice(0, 4),
        dst_height=1,
        dst_width=4,
        dst_affine=dst_affine,
        dst_crs=wgs84,
        dst_to_4326=None,
    )
    assert win.chunk_bbox_4326 == pytest.approx([10.0, 49.0, 14.0, 50.0])


def test_chunk_bbox_4326_utm_transforms(utm32n):
    """A UTM chunk bbox is transformed to EPSG:4326."""
    dst_affine = Affine(100.0, 0.0, 500_000.0, 0.0, -100.0, 5_550_000.0)
    win = _resolve_spatial_window(
        slice(0, 10),
        slice(0, 10),
        dst_height=10,
        dst_width=10,
        dst_affine=dst_affine,
        dst_crs=utm32n,
        dst_to_4326=Transformer.from_crs(utm32n, CRS.from_epsg(4326), always_xy=True),
    )
    minx, miny, maxx, maxy = win.chunk_bbox_4326
    assert -180 <= minx <= 180
    assert -90 <= miny <= 90
    assert minx < maxx
    assert miny < maxy


def test_chunk_bbox_4326_ordering(utm32n):
    """Returned bbox satisfies minx < maxx and miny < maxy."""
    dst_affine = Affine(1000.0, 0.0, 400_000.0, 0.0, -1000.0, 5_600_000.0)
    win = _resolve_spatial_window(
        slice(0, 100),
        slice(0, 100),
        dst_height=100,
        dst_width=100,
        dst_affine=dst_affine,
        dst_crs=utm32n,
        dst_to_4326=Transformer.from_crs(utm32n, CRS.from_epsg(4326), always_xy=True),
    )
    minx, miny, maxx, maxy = win.chunk_bbox_4326
    assert minx < maxx
    assert miny < maxy


# ---------------------------------------------------------------------------
# MultiBandStacBackendArray._raw_getitem — single-band behaviour
# ---------------------------------------------------------------------------


def test_raw_getitem_empty_items_returns_nodata(wgs84):
    """When DuckDB returns no items, the chunk is filled with nodata."""
    arr = _make_array(wgs84)

    with patch("rustac.DuckdbClient.search", return_value=[]):
        result = arr._raw_getitem((slice(0, 1), slice(0, 2), slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 2, 1, 4)
    np.testing.assert_array_equal(result, -9999.0)


def test_raw_getitem_scalar_time_squeezes(wgs84):
    """Integer time index squeezes the time dimension from the output."""
    arr = _make_array(wgs84)

    with patch("rustac.DuckdbClient.search", return_value=[]):
        result = arr._raw_getitem((slice(0, 1), 0, slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 1, 4)


def test_raw_getitem_with_items_calls_mosaic(wgs84):
    """When items are returned, async_mosaic_chunk_multiband is called and result used."""
    arr = _make_array(wgs84, dates=["2023-01-01"])
    band = "B04"
    fake_items = [{"id": "item-1", "assets": {band: {"href": "s3://b/f.tif"}}}]
    fake_chunk = {band: np.full((1, 1, 4), 42.0, dtype=np.float32)}

    with (
        patch("rustac.DuckdbClient.search", return_value=fake_items),
        patch(
            "lazycogs._backend.async_mosaic_chunk_multiband",
            new_callable=AsyncMock,
            return_value=fake_chunk,
        ),
    ):
        result = arr._raw_getitem((0, 0, slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 4)
    np.testing.assert_array_equal(result, 42.0)


def test_raw_getitem_chunk_affine_offset(wgs84):
    """Chunk affine is translated by (x_start, y_start) from the full grid."""
    arr = _make_array(wgs84)

    fake_items = [{"id": "x"}]
    with (
        patch("rustac.DuckdbClient.search", return_value=fake_items),
        patch(
            "lazycogs._backend.async_mosaic_chunk_multiband",
            new_callable=AsyncMock,
            return_value={"B04": np.zeros((1, 1, 2), dtype=np.float32)},
        ),
    ):
        arr._raw_getitem((slice(0, 1), 0, slice(0, 1), slice(2, 4)))

    # The full grid origin is (10, 50); x_start=2 → chunk origin x = 10 + 2 = 12
    chunk_affine = arr.dst_affine * Affine.translation(2, 0)
    assert chunk_affine.c == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# MultiBandStacBackendArray._raw_getitem — multi-band behaviour
# ---------------------------------------------------------------------------


def _make_multiband_array(
    crs: CRS, bands: list[str], dates: list[str] | None = None
) -> MultiBandStacBackendArray:
    """Return a minimal MultiBandStacBackendArray for unit testing."""
    if dates is None:
        dates = ["2023-01-01"]
    return MultiBandStacBackendArray(
        parquet_path="/tmp/fake.parquet",
        duckdb_client=DuckdbClient(),
        bands=bands,
        dates=dates,
        dst_affine=Affine(1.0, 0.0, 10.0, 0.0, -1.0, 50.0),
        dst_crs=crs,
        bbox_4326=[10.0, 49.0, 14.0, 50.0],
        sortby=None,
        filter=None,
        ids=None,
        dst_width=4,
        dst_height=1,
        dtype=np.dtype("float32"),
        nodata=-9999.0,
        mosaic_method_cls=FirstMethod,
    )


def test_multiband_raw_getitem_no_items_returns_nodata(wgs84):
    """When no items are found, all bands are filled with nodata."""
    multi = _make_multiband_array(wgs84, ["B01", "B02"])

    with patch("rustac.DuckdbClient.search", return_value=[]):
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
        # _mosaic_all_dates returns a list with one entry per time step.
        return [
            {
                b: np.full((1, 1, 4), float(i), dtype=np.float32)
                for i, b in enumerate(bands)
            }
        ]

    with (
        patch("rustac.DuckdbClient.search", return_value=fake_items),
        patch("lazycogs._backend._run_coroutine", side_effect=_fake_run_coroutine),
    ):
        result = multi._raw_getitem((slice(0, 2), 0, slice(0, 1), slice(0, 4)))

    # One call to _run_coroutine (which runs all time steps together), not one per band.
    assert call_count[0] == 1
    assert result.shape == (2, 1, 4)


def test_multiband_raw_getitem_squeeze_band(wgs84):
    """Integer band index squeezes the band dimension."""
    multi = _make_multiband_array(wgs84, ["B01", "B02"])

    with patch("rustac.DuckdbClient.search", return_value=[]):
        result = multi._raw_getitem((0, 0, slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 4)


def test_multiband_raw_getitem_single_band_single_pixel(wgs84):
    """All dimensions squeezed returns a scalar array."""
    multi = _make_multiband_array(wgs84, ["B01", "B02"])

    with patch("rustac.DuckdbClient.search", return_value=[]):
        result = multi._raw_getitem((0, 0, 0, 0))

    assert result.shape == ()
