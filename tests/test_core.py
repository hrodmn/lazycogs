"""Tests for the open() / open_async() entry points."""

from __future__ import annotations
from unittest.mock import patch

import numpy as np
import pytest

import lazycogs
from lazycogs._core import _build_time_steps
from lazycogs._temporal import _DayGrouper, _MonthGrouper, _FixedDayGrouper


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

    with pytest.raises(Exception) as exc_info:
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
    with patch("rustac.search_sync", return_value=_FAKE_ITEMS_SAME_DAY):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            temporal_grouper=_DayGrouper(),
        )
    assert filter_strings == ["2023-01-15"]
    assert len(time_coords) == 1
    assert time_coords[0] == np.datetime64("2023-01-15", "D")


def test_build_time_steps_day_two_days():
    """Items on two different days produce two time steps."""
    with patch("rustac.search_sync", return_value=_FAKE_ITEMS_TWO_DAYS):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            temporal_grouper=_DayGrouper(),
        )
    assert filter_strings == ["2023-01-15", "2023-01-16"]
    assert len(time_coords) == 2


def test_build_time_steps_month_deduplicates_same_month():
    """Items in the same month collapse to one time step with MonthGrouper."""
    with patch("rustac.search_sync", return_value=_FAKE_ITEMS_SAME_MONTH):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            temporal_grouper=_MonthGrouper(),
        )
    assert filter_strings == ["2023-01-01/2023-01-31"]
    assert len(time_coords) == 1
    assert time_coords[0] == np.datetime64("2023-01-01", "D")


def test_build_time_steps_month_two_months():
    """Items in two different months produce two time steps."""
    with patch("rustac.search_sync", return_value=_FAKE_ITEMS_TWO_MONTHS):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
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
    with patch("rustac.search_sync", return_value=items):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
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
    with patch("rustac.search_sync", return_value=items):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            temporal_grouper=_FixedDayGrouper(16),
        )
    assert len(filter_strings) == 2
    assert time_coords[0] < time_coords[1]


def test_build_time_steps_empty_items():
    """Empty item list returns empty lists."""
    with patch("rustac.search_sync", return_value=[]):
        filter_strings, time_coords = _build_time_steps(
            "fake.parquet",
            temporal_grouper=_DayGrouper(),
        )
    assert filter_strings == []
    assert time_coords == []


def test_build_time_steps_uses_start_datetime_fallback():
    """Items with start_datetime (no datetime) are handled."""
    items = [{"properties": {"start_datetime": "2023-03-10T00:00:00Z"}}]
    with patch("rustac.search_sync", return_value=items):
        filter_strings, _ = _build_time_steps(
            "fake.parquet",
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
    """open() does not raise RuntimeError when called from within a running event loop."""
    import asyncio

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
