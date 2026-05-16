"""Tests for the open() entry point."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pytest
import rustac
from obstore.store import MemoryStore
from rustac import DuckdbClient

import lazycogs
from lazycogs._backend import MultiBandStacBackendArray
from lazycogs._core import _build_time_steps, _smoketest_store
from lazycogs._temporal import _DayGrouper, _FixedDayGrouper, _MonthGrouper


def _items_to_arrow(items: list[dict]) -> rustac.DuckdbClient:
    """Convert simplified fake items to an Arrow table via rustac.to_arrow.

    Accepts the same simplified item dicts used in existing tests
    (``{"properties": {"datetime": "..."}}``) and wraps them into
    complete-enough STAC items for ``rustac.to_arrow`` to accept.
    Returns ``None`` when *items* is empty, matching ``search_to_arrow``
    behaviour.
    """
    if not items:
        return None
    full_items = []
    for i, item in enumerate(items):
        props = dict(item.get("properties", {}))
        full_items.append(
            {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": f"fake-{i}",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "bbox": [-0.1, -0.1, 0.1, 0.1],
                "properties": props,
                "links": [],
                "assets": {},
            },
        )
    return rustac.to_arrow(full_items)


def test_open_rejects_non_parquet_href():
    """open() raises ValueError when href is not a geoparquet file."""
    with pytest.raises(ValueError, match=r"\.parquet"):
        lazycogs.open(
            "https://earth-search.aws.element84.com/v1",
            bbox=(-93.5, 44.5, -93.0, 45.0),
            crs="EPSG:4326",
            resolution=0.0001,
        )


def test_open_rejects_json_href():
    """open() raises ValueError for non-parquet file extensions."""
    with pytest.raises(ValueError, match=r"\.parquet"):
        lazycogs.open(
            "items.json",
            bbox=(-93.5, 44.5, -93.0, 45.0),
            crs="EPSG:4326",
            resolution=0.0001,
        )


def test_open_accepts_parquet_extension_passes_validation(tmp_path):
    """.parquet extension passes the extension check; error is about file content."""
    path = str(tmp_path / "items.parquet")
    path_obj = tmp_path / "items.parquet"
    path_obj.write_bytes(b"")  # empty file — rustac will error, but not on extension

    with pytest.raises(Exception) as exc_info:  # noqa: PT011
        lazycogs.open(
            path,
            bbox=(-93.5, 44.5, -93.0, 45.0),
            crs="EPSG:4326",
            resolution=0.0001,
        )
    # The error should not be the extension validation error
    assert "must be a .parquet" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# _build_time_steps
# ---------------------------------------------------------------------------

_FAKE_ITEMS_SAME_DAY = [
    {"properties": {"datetime": "2023-01-15T10:00:00Z"}},
    {"properties": {"datetime": "2023-01-15T14:30:00Z"}},
]

_FAKE_ITEMS_TWO_DAYS = [
    {"properties": {"datetime": "2023-01-15T10:00:00Z"}},
    {"properties": {"datetime": "2023-01-16T08:00:00Z"}},
]

_FAKE_ITEMS_SAME_MONTH = [
    {"properties": {"datetime": "2023-01-05T10:00:00Z"}},
    {"properties": {"datetime": "2023-01-20T14:30:00Z"}},
]

_FAKE_ITEMS_TWO_MONTHS = [
    {"properties": {"datetime": "2023-01-15T00:00:00Z"}},
    {"properties": {"datetime": "2023-02-10T00:00:00Z"}},
]


def test_build_time_steps_day_deduplicates_same_day():
    """Items on the same day collapse to one time step with DayGrouper."""
    table = _items_to_arrow(_FAKE_ITEMS_SAME_DAY)
    with patch("rustac.DuckdbClient.search_to_arrow", return_value=table):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            duckdb_client=DuckdbClient(),
            temporal_grouper=_DayGrouper(),
        )
    assert filter_strings == ["2023-01-15"]
    assert len(time_coords) == 1
    assert time_coords[0] == np.datetime64("2023-01-15", "D")


def test_build_time_steps_day_two_days():
    """Items on two different days produce two time steps."""
    table = _items_to_arrow(_FAKE_ITEMS_TWO_DAYS)
    with patch("rustac.DuckdbClient.search_to_arrow", return_value=table):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            duckdb_client=DuckdbClient(),
            temporal_grouper=_DayGrouper(),
        )
    assert filter_strings == ["2023-01-15", "2023-01-16"]
    assert len(time_coords) == 2


def test_build_time_steps_month_deduplicates_same_month():
    """Items in the same month collapse to one time step with MonthGrouper."""
    table = _items_to_arrow(_FAKE_ITEMS_SAME_MONTH)
    with patch("rustac.DuckdbClient.search_to_arrow", return_value=table):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            duckdb_client=DuckdbClient(),
            temporal_grouper=_MonthGrouper(),
        )
    assert filter_strings == ["2023-01-01/2023-01-31"]
    assert len(time_coords) == 1
    assert time_coords[0] == np.datetime64("2023-01-01", "D")


def test_build_time_steps_month_two_months():
    """Items in two different months produce two time steps."""
    table = _items_to_arrow(_FAKE_ITEMS_TWO_MONTHS)
    with patch("rustac.DuckdbClient.search_to_arrow", return_value=table):
        filter_strings, _ = _build_time_steps(
            "fake.parquet",
            duckdb_client=DuckdbClient(),
            temporal_grouper=_MonthGrouper(),
        )
    assert len(filter_strings) == 2
    assert filter_strings[0] == "2023-01-01/2023-01-31"
    assert filter_strings[1] == "2023-02-01/2023-02-28"


def test_build_time_steps_p16d_same_bucket():
    """Items within the same 16-day window produce one time step."""
    # Epoch is 2000-01-01. Jan 10 and Jan 12 2023 are in the same 16-day bucket.
    items = [
        {"properties": {"datetime": "2023-01-10T00:00:00Z"}},
        {"properties": {"datetime": "2023-01-12T00:00:00Z"}},
    ]
    table = _items_to_arrow(items)
    with patch("rustac.DuckdbClient.search_to_arrow", return_value=table):
        filter_strings, _ = _build_time_steps(
            "fake.parquet",
            duckdb_client=DuckdbClient(),
            temporal_grouper=_FixedDayGrouper(16),
        )
    assert len(filter_strings) == 1


def test_build_time_steps_p16d_adjacent_buckets():
    """Items in adjacent 16-day buckets produce two time steps."""
    # Epoch 2000-01-01. Bucket boundaries fall every 16 days.
    # 2000-01-01 = day 0, bucket 0: days 0..15 = 2000-01-01..2000-01-16
    # bucket 1: 2000-01-17..2000-02-01
    items = [
        {"properties": {"datetime": "2000-01-16T00:00:00Z"}},  # last day of bucket 0
        {"properties": {"datetime": "2000-01-17T00:00:00Z"}},  # first day of bucket 1
    ]
    table = _items_to_arrow(items)
    with patch("rustac.DuckdbClient.search_to_arrow", return_value=table):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            duckdb_client=DuckdbClient(),
            temporal_grouper=_FixedDayGrouper(16),
        )
    assert len(filter_strings) == 2
    assert time_coords[0] < time_coords[1]


def test_build_time_steps_empty_items():
    """Empty item list returns empty lists."""
    with patch("rustac.DuckdbClient.search_to_arrow", return_value=None):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            duckdb_client=DuckdbClient(),
            temporal_grouper=_DayGrouper(),
        )
    assert filter_strings == []
    assert time_coords == []


def test_build_time_steps_uses_start_datetime_fallback():
    """Items with start_datetime (no datetime) are handled."""
    items = [
        {
            "properties": {
                "datetime": None,
                "start_datetime": "2023-03-10T00:00:00Z",
                "end_datetime": "2023-03-10T23:59:59Z",
            },
        },
    ]
    table = _items_to_arrow(items)
    with patch("rustac.DuckdbClient.search_to_arrow", return_value=table):
        filter_strings, _ = _build_time_steps(
            "fake.parquet",
            duckdb_client=DuckdbClient(),
            temporal_grouper=_DayGrouper(),
        )
    assert filter_strings == ["2023-03-10"]


def test_open_time_period_kwarg_wires_through():
    """time_period parameter is accepted and wires through open()."""
    with pytest.raises(ValueError, match=r"\.parquet"):
        lazycogs.open(
            "https://example.com/stac",
            bbox=(-93.5, 44.5, -93.0, 45.0),
            crs="EPSG:4326",
            resolution=0.0001,
            time_period="P1M",
        )


def test_open_invalid_time_period_raises():
    """open() raises ValueError for an unrecognised time_period."""
    with pytest.raises(ValueError, match="Unsupported time_period"):
        lazycogs.open(
            "items.parquet",
            bbox=(-93.5, 44.5, -93.0, 45.0),
            crs="EPSG:4326",
            resolution=0.0001,
            time_period="bad",
        )


def test_open_works_inside_running_event_loop(tmp_path):
    """open() does not raise RuntimeError when called inside a running event loop."""

    path = str(tmp_path / "items.parquet")
    (tmp_path / "items.parquet").write_bytes(b"")

    result: dict[str, Exception] = {}

    async def _call_open() -> None:
        try:
            lazycogs.open(
                path,
                bbox=(-93.5, 44.5, -93.0, 45.0),
                crs="EPSG:4326",
                resolution=0.0001,
            )
        except RuntimeError as exc:
            result["error"] = exc
        except Exception:
            pass  # rustac will error on empty file — that is fine

    asyncio.run(_call_open())
    assert "error" not in result, f"Got RuntimeError: {result.get('error')}"


def _fake_open_item() -> dict:
    """Return a minimal STAC item used by open() tests."""
    return {
        "id": "test-item",
        "stac_extensions": [],
        "properties": {"datetime": "2023-01-15T10:00:00Z"},
        "assets": {
            "B04": {
                "href": "s3://bucket/B04.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data"],
            },
        },
    }


@pytest.fixture
def opened_dataarray(tmp_path):
    """Return a small DataArray from open() with DuckDB calls patched."""

    parquet = tmp_path / "items.parquet"
    parquet.write_bytes(b"")

    store = MemoryStore()
    store.put("B04.tif", b"dummy")

    table = _items_to_arrow([{"properties": {"datetime": "2023-01-15T10:00:00Z"}}])

    async def fake_open(path: str, *, store):
        return object()

    with (
        patch("rustac.DuckdbClient.search", return_value=[_fake_open_item()]),
        patch("rustac.DuckdbClient.search_to_arrow", return_value=table),
        patch("lazycogs._core.GeoTIFF.open", side_effect=fake_open),
    ):
        return lazycogs.open(
            str(parquet),
            bbox=(0.0, 0.0, 100.0, 100.0),
            crs="EPSG:32632",
            resolution=10.0,
            store=store,
            path_from_href=lambda href: href.split("/", 3)[-1],
        )


def test_open_sets_expected_dataarray_attributes(opened_dataarray):
    """open() attaches all expected extra attributes to the returned DataArray."""
    da = opened_dataarray

    # Coordinates
    assert da.coords["band"].values.tolist() == ["B04"]
    assert len(da.coords["time"]) == 1
    assert da.coords["time"].values[0] == np.datetime64("2023-01-15", "D")
    assert "spatial_ref" in da.coords
    assert "x" in da.coords
    assert "y" in da.coords

    # Dimensions
    assert da.dims == ("band", "time", "y", "x")

    # Attributes
    assert da.attrs["grid_mapping"] == "spatial_ref"
    assert da.attrs["spatial:transform_type"] == "affine"
    assert da.attrs["spatial:registration"] == "pixel"
    assert da.attrs["proj:code"] == "EPSG:32632"

    # spatial:transform should be the 6-element GeoTransform list
    assert isinstance(da.attrs["spatial:transform"], list)
    assert len(da.attrs["spatial:transform"]) == 6

    # zarr_conventions
    assert isinstance(da.attrs["zarr_conventions"], list)
    names = {c["name"] for c in da.attrs["zarr_conventions"]}
    assert "spatial:" in names
    assert "proj:" in names

    # Internal bookkeeping attributes
    assert isinstance(da.attrs["_stac_backend"], MultiBandStacBackendArray)
    assert da.attrs["_stac_time_coords"].dtype == np.dtype("datetime64[D]")


def test_chunked_spatial_selection_computes_scalar_spatial_coords(opened_dataarray):
    """Chunked nearest-neighbour spatial selection keeps x/y coords scalar."""
    da = opened_dataarray

    selected = da.chunk(time=1).sel(x=25.0, y=75.0, method="nearest")

    x_coord = selected.coords["x"].compute()
    y_coord = selected.coords["y"].compute()

    assert x_coord.dims == ()
    assert y_coord.dims == ()
    assert x_coord.shape == ()
    assert y_coord.shape == ()


def test_chunked_spatial_selection_full_compute_succeeds(opened_dataarray):
    """Chunked nearest-neighbour spatial selection can compute end-to-end."""
    selected = opened_dataarray.chunk(time=1).sel(x=25.0, y=75.0, method="nearest")

    with patch("rustac.DuckdbClient.search", return_value=[]):
        computed = selected.compute()

    assert computed.dims == ("band", "time")
    assert computed.shape == (1, 1)


# ---------------------------------------------------------------------------
# _smoketest_store
# ---------------------------------------------------------------------------


class _ProtocolStore:
    """Minimal non-obstore store that satisfies the async_tiff protocol shape."""

    async def get_range_async(self, _start: int, _end: int):
        return b""

    async def get_ranges_async(self, starts_ends):
        return [b"" for _ in starts_ends]


_SMOKETEST_ITEM = {
    "id": "smoke-item",
    "stac_extensions": [],
    "properties": {"datetime": "2023-06-01T00:00:00Z"},
    "assets": {
        "B04": {
            "href": "s3://my-bucket/B04.tif",
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "roles": ["data"],
        },
    },
}


def test_smoketest_passes_when_geotiff_open_succeeds():
    """_smoketest_store does not raise when GeoTIFF.open succeeds."""

    store = MemoryStore()
    store.put("B04.tif", b"dummy")

    async def fake_open(path: str, *, store):
        assert path == "B04.tif"
        assert store is store_obj
        return object()

    store_obj = store
    with (
        patch("rustac.DuckdbClient.search", return_value=[_SMOKETEST_ITEM]),
        patch("lazycogs._core.GeoTIFF.open", side_effect=fake_open),
    ):
        _smoketest_store(
            "items.parquet",
            duckdb_client=DuckdbClient(),
            store=store,
            path_from_href=lambda href: href.split("/", 3)[-1],
        )


def test_smoketest_raises_runtime_error_on_geotiff_open_failure():
    """_smoketest_store raises RuntimeError when GeoTIFF.open fails."""

    store = MemoryStore()

    async def fake_open(path: str, *, store):
        raise FileNotFoundError(path)

    with (
        patch("rustac.DuckdbClient.search", return_value=[_SMOKETEST_ITEM]),
        patch("lazycogs._core.GeoTIFF.open", side_effect=fake_open),
        pytest.raises(RuntimeError, match=r"GeoTIFF\.open"),
    ):
        _smoketest_store(
            "items.parquet",
            duckdb_client=DuckdbClient(),
            store=store,
            path_from_href=lambda href: href.split("/", 3)[-1],
        )


def test_smoketest_accepts_protocol_store_without_head(tmp_path):
    """A custom protocol store without head() still passes startup validation."""

    parquet = tmp_path / "items.parquet"
    parquet.write_bytes(b"")
    store = _ProtocolStore()
    table = _items_to_arrow([{"properties": {"datetime": "2023-01-15T10:00:00Z"}}])

    async def fake_open(path: str, *, store):
        assert path == "B04.tif"
        assert store is protocol_store
        return object()

    protocol_store = store
    with (
        patch("rustac.DuckdbClient.search", return_value=[_fake_open_item()]),
        patch("rustac.DuckdbClient.search_to_arrow", return_value=table),
        patch("lazycogs._core.GeoTIFF.open", side_effect=fake_open),
    ):
        da = lazycogs.open(
            str(parquet),
            bbox=(0.0, 0.0, 100.0, 100.0),
            crs="EPSG:32632",
            resolution=10.0,
            store=store,
            path_from_href=lambda href: href.split("/", 3)[-1],
        )

    assert da.attrs["_stac_backend"].store is store


def test_smoketest_no_op_when_no_items():
    """_smoketest_store does nothing when the query returns no items."""
    with patch("rustac.DuckdbClient.search", return_value=[]):
        _smoketest_store("items.parquet", duckdb_client=DuckdbClient())


def test_smoketest_prefers_specified_band():
    """_smoketest_store uses the first specified band when bands= is given."""

    store = MemoryStore()

    item = {
        "id": "multi-band-item",
        "stac_extensions": [],
        "properties": {"datetime": "2023-06-01T00:00:00Z"},
        "assets": {
            "B04": {"href": "s3://bucket/B04.tif", "roles": ["data"]},
            "B08": {"href": "s3://bucket/B08.tif", "roles": ["data"]},
        },
    }

    seen: list[str] = []

    async def fake_open(path: str, *, store):
        seen.append(path)
        return object()

    with (
        patch("rustac.DuckdbClient.search", return_value=[item]),
        patch("lazycogs._core.GeoTIFF.open", side_effect=fake_open),
    ):
        _smoketest_store(
            "items.parquet",
            duckdb_client=DuckdbClient(),
            bands=["B08"],
            store=store,
            path_from_href=lambda href: href.split("/", 3)[-1],
        )

    assert seen == ["B08.tif"]
