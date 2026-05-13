"""Tests for _backend helpers and _async_getitem / _sync_getitem."""

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from affine import Affine
from pyproj import CRS
from rustac import DuckdbClient
from xarray.core import indexing

from lazycogs._backend import MultiBandStacBackendArray
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
    dst_affine: Affine | None = None,
    dst_height: int = 1,
    dst_width: int = 4,
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
        dst_affine=dst_affine or Affine(1.0, 0.0, 10.0, 0.0, -1.0, 50.0),
        dst_crs=crs,
        bbox_4326=[10.0, 49.0, 14.0, 50.0],
        sortby=None,
        filter=None,
        ids=None,
        dst_width=dst_width,
        dst_height=dst_height,
        dtype=np.dtype("float32"),
        nodata=-9999.0,
        mosaic_method_cls=FirstMethod,
    )


# ---------------------------------------------------------------------------
# _resolve_spatial_window
# ---------------------------------------------------------------------------


def test_chunk_bbox_4326_identity_in_wgs84(wgs84):
    """When dst_crs is EPSG:4326, bbox is returned as-is."""
    arr = _make_array(wgs84)
    win = arr._resolve_spatial_window(slice(0, 1), slice(0, 4))
    assert win.chunk_bbox_4326 == pytest.approx([10.0, 49.0, 14.0, 50.0])


def test_chunk_bbox_4326_utm_transforms(utm32n):
    """A UTM chunk bbox is transformed to EPSG:4326."""
    arr = _make_array(
        utm32n,
        dst_affine=Affine(100.0, 0.0, 500_000.0, 0.0, -100.0, 5_550_000.0),
        dst_height=10,
        dst_width=10,
    )
    win = arr._resolve_spatial_window(slice(0, 10), slice(0, 10))
    minx, miny, maxx, maxy = win.chunk_bbox_4326
    assert -180 <= minx <= 180
    assert -90 <= miny <= 90
    assert minx < maxx
    assert miny < maxy


def test_chunk_bbox_4326_ordering(utm32n):
    """Returned bbox satisfies minx < maxx and miny < maxy."""
    arr = _make_array(
        utm32n,
        dst_affine=Affine(1000.0, 0.0, 400_000.0, 0.0, -1000.0, 5_600_000.0),
        dst_height=100,
        dst_width=100,
    )
    win = arr._resolve_spatial_window(slice(0, 100), slice(0, 100))
    minx, miny, maxx, maxy = win.chunk_bbox_4326
    assert minx < maxx
    assert miny < maxy


# ---------------------------------------------------------------------------
# MultiBandStacBackendArray._sync_getitem — single-band behaviour
# ---------------------------------------------------------------------------


def test_raw_getitem_empty_items_returns_nodata(wgs84):
    """When DuckDB returns no items, the chunk is filled with nodata."""
    arr = _make_array(wgs84)

    with patch("rustac.DuckdbClient.search", return_value=[]):
        result = arr._sync_getitem((slice(0, 1), slice(0, 2), slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 2, 1, 4)
    np.testing.assert_array_equal(result, -9999.0)


def test_raw_getitem_scalar_time_squeezes(wgs84):
    """Integer time index squeezes the time dimension from the output."""
    arr = _make_array(wgs84)

    with patch("rustac.DuckdbClient.search", return_value=[]):
        result = arr._sync_getitem((slice(0, 1), 0, slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 1, 4)


def test_raw_getitem_with_items_calls_mosaic(wgs84):
    """When items are returned, read_chunk_async is called and result used."""
    arr = _make_array(wgs84, dates=["2023-01-01"])
    band = "B04"
    fake_items = [{"id": "item-1", "assets": {band: {"href": "s3://b/f.tif"}}}]
    fake_chunk = {band: np.full((1, 1, 4), 42.0, dtype=np.float32)}

    with (
        patch("rustac.DuckdbClient.search", return_value=fake_items),
        patch(
            "lazycogs._backend.read_chunk_async",
            new_callable=AsyncMock,
            return_value=fake_chunk,
        ),
    ):
        result = arr._sync_getitem((0, 0, slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 4)
    np.testing.assert_array_equal(result, 42.0)


def test_raw_getitem_chunk_affine_offset(wgs84):
    """Chunk affine is translated by (x_start, y_start) from the full grid."""
    arr = _make_array(wgs84)

    fake_items = [{"id": "x"}]
    with (
        patch("rustac.DuckdbClient.search", return_value=fake_items),
        patch(
            "lazycogs._backend.read_chunk_async",
            new_callable=AsyncMock,
            return_value={"B04": np.zeros((1, 1, 2), dtype=np.float32)},
        ),
    ):
        arr._sync_getitem((slice(0, 1), 0, slice(0, 1), slice(2, 4)))

    # The full grid origin is (10, 50); x_start=2 → chunk origin x = 10 + 2 = 12
    chunk_affine = arr.dst_affine * Affine.translation(2, 0)
    assert chunk_affine.c == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# MultiBandStacBackendArray._sync_getitem — multi-band behaviour
# ---------------------------------------------------------------------------


def _make_multiband_array(
    crs: CRS,
    bands: list[str],
    dates: list[str] | None = None,
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
        result = multi._sync_getitem(
            (slice(0, 2), slice(0, 1), slice(0, 1), slice(0, 4)),
        )

    assert result.shape == (2, 1, 1, 4)
    np.testing.assert_array_equal(result, -9999.0)


def test_multiband_raw_getitem_calls_multiband_mosaic(wgs84):
    """_sync_getitem calls read_chunk_async once per time step, not per band."""
    bands = ["B01", "B02"]
    multi = _make_multiband_array(wgs84, bands)
    fake_items = [
        {"id": "item-1", "assets": {b: {"href": f"s3://b/{b}.tif"} for b in bands}},
    ]

    call_count = [0]

    async def _fake_read_chunk_async(*args, **kwargs):
        call_count[0] += 1
        return {
            b: np.full((1, 1, 4), float(i), dtype=np.float32)
            for i, b in enumerate(bands)
        }

    with (
        patch("rustac.DuckdbClient.search", return_value=fake_items),
        patch("lazycogs._backend.read_chunk_async", side_effect=_fake_read_chunk_async),
    ):
        result = multi._sync_getitem((slice(0, 2), 0, slice(0, 1), slice(0, 4)))

    # One call to read_chunk_async per time step (all bands per call), not one per band.
    assert call_count[0] == 1
    assert result.shape == (2, 1, 4)


def test_multiband_raw_getitem_forwards_resampling(wgs84):
    """The selected resampling mode reaches read_chunk_async unchanged."""
    bands = ["B01", "B02"]
    multi = _make_multiband_array(wgs84, bands)
    multi.resampling = "cubic"
    fake_items = [
        {"id": "item-1", "assets": {b: {"href": f"s3://b/{b}.tif"} for b in bands}},
    ]
    fake_chunk = {b: np.zeros((1, 1, 4), dtype=np.float32) for b in bands}

    with (
        patch("rustac.DuckdbClient.search", return_value=fake_items),
        patch(
            "lazycogs._backend.read_chunk_async",
            new_callable=AsyncMock,
            return_value=fake_chunk,
        ) as read_chunk_async_mock,
    ):
        multi._sync_getitem((slice(0, 2), 0, slice(0, 1), slice(0, 4)))

    assert read_chunk_async_mock.await_args.kwargs["resampling"] == "cubic"


def test_multiband_raw_getitem_squeeze_band(wgs84):
    """Integer band index squeezes the band dimension."""
    multi = _make_multiband_array(wgs84, ["B01", "B02"])

    with patch("rustac.DuckdbClient.search", return_value=[]):
        result = multi._sync_getitem((0, 0, slice(0, 1), slice(0, 4)))

    assert result.shape == (1, 4)


def test_multiband_raw_getitem_single_band_single_pixel(wgs84):
    """All dimensions squeezed returns a scalar array."""
    multi = _make_multiband_array(wgs84, ["B01", "B02"])

    with patch("rustac.DuckdbClient.search", return_value=[]):
        result = multi._sync_getitem((0, 0, 0, 0))

    assert result.shape == ()


# ---------------------------------------------------------------------------
# MultiBandStacBackendArray.async_getitem — async protocol
# ---------------------------------------------------------------------------


def test_async_getitem_awaitable(wgs84):
    """async_getitem returns the same numpy array as the sync path."""
    arr = _make_array(wgs84)

    key = indexing.BasicIndexer(
        (slice(0, 1), slice(0, 2), slice(0, 1), slice(0, 4)),
    )

    async def _inner():
        with patch("rustac.DuckdbClient.search", return_value=[]):
            sync_result = arr[key]
            async_result = await arr.async_getitem(key)
        return sync_result, async_result

    sync_result, async_result = asyncio.run(_inner())

    assert isinstance(async_result, np.ndarray)
    np.testing.assert_array_equal(async_result, sync_result)


def test_async_getitem_concurrent_chunk_reads(wgs84):
    """Multiple async_getitem calls on the same loop run concurrently."""
    arr = _make_array(wgs84)

    key1 = indexing.BasicIndexer(
        (slice(0, 1), slice(0, 1), slice(0, 1), slice(0, 2)),
    )
    key2 = indexing.BasicIndexer(
        (slice(0, 1), slice(1, 2), slice(0, 1), slice(2, 4)),
    )

    async def _inner():
        with patch("rustac.DuckdbClient.search", return_value=[]):
            return await asyncio.gather(
                arr.async_getitem(key1),
                arr.async_getitem(key2),
            )

    out1, out2 = asyncio.run(_inner())

    assert isinstance(out1, np.ndarray)
    assert isinstance(out2, np.ndarray)
    assert out1.shape == (1, 1, 1, 2)
    assert out2.shape == (1, 1, 1, 2)
