"""Tests for _temporal.py: ISO duration parsing and temporal groupers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from stac_cog_xarray._temporal import (
    _DayGrouper,
    _FixedDayGrouper,
    _MonthGrouper,
    _WeekGrouper,
    _YearGrouper,
    _parse_iso_duration,
    grouper_from_period,
)


# ---------------------------------------------------------------------------
# _parse_iso_duration
# ---------------------------------------------------------------------------


class TestParseIsoDuration:
    def test_p1d(self):
        assert _parse_iso_duration("P1D") == (1, "D")

    def test_p16d(self):
        assert _parse_iso_duration("P16D") == (16, "D")

    def test_p5d(self):
        assert _parse_iso_duration("P5D") == (5, "D")

    def test_p1w(self):
        assert _parse_iso_duration("P1W") == (1, "W")

    def test_p1m(self):
        assert _parse_iso_duration("P1M") == (1, "M")

    def test_p1y(self):
        assert _parse_iso_duration("P1Y") == (1, "Y")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported ISO 8601 duration"):
            _parse_iso_duration("1D")

    def test_invalid_time_component_raises(self):
        # PT1H is a valid ISO duration but not supported
        with pytest.raises(ValueError):
            _parse_iso_duration("PT1H")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_iso_duration("")


# ---------------------------------------------------------------------------
# _DayGrouper
# ---------------------------------------------------------------------------


class TestDayGrouper:
    def setup_method(self):
        self.g = _DayGrouper()

    def test_group_key_truncates_to_date(self):
        assert self.g.group_key("2023-01-15T10:30:00Z") == "2023-01-15"

    def test_group_key_bare_date(self):
        assert self.g.group_key("2023-01-15") == "2023-01-15"

    def test_datetime_filter_returns_date(self):
        assert self.g.datetime_filter("2023-01-15") == "2023-01-15"

    def test_to_datetime64(self):
        result = self.g.to_datetime64("2023-01-15")
        assert result == np.datetime64("2023-01-15", "D")

    def test_group_key_same_day_different_times(self):
        key1 = self.g.group_key("2023-01-15T08:00:00Z")
        key2 = self.g.group_key("2023-01-15T22:59:59Z")
        assert key1 == key2

    def test_sort_order(self):
        datetimes = ["2023-03-01T00:00Z", "2023-01-15T00:00Z", "2023-06-30T00:00Z"]
        keys = [self.g.group_key(dt) for dt in datetimes]
        assert sorted(keys) == ["2023-01-15", "2023-03-01", "2023-06-30"]


# ---------------------------------------------------------------------------
# _WeekGrouper
# ---------------------------------------------------------------------------


class TestWeekGrouper:
    def setup_method(self):
        self.g = _WeekGrouper()

    def test_group_key_basic(self):
        # 2023-01-09 is the Monday of ISO week 2023-W02
        assert self.g.group_key("2023-01-09T00:00:00Z") == "2023-W02"

    def test_group_key_year_boundary(self):
        # 2022-01-02 belongs to ISO week 2021-W52, not 2022-W00
        assert self.g.group_key("2022-01-02T00:00:00Z") == "2021-W52"

    def test_group_key_year_boundary_jan3(self):
        # 2022-01-03 is the Monday of ISO week 2022-W01
        assert self.g.group_key("2022-01-03T00:00:00Z") == "2022-W01"

    def test_datetime_filter_basic(self):
        # Week 2023-W02: Mon 2023-01-09 to Sun 2023-01-15
        assert self.g.datetime_filter("2023-W02") == "2023-01-09/2023-01-15"

    def test_datetime_filter_year_boundary(self):
        # ISO week 2021-W52: Mon 2021-12-27 to Sun 2022-01-02
        result = self.g.datetime_filter("2021-W52")
        assert result == "2021-12-27/2022-01-02"

    def test_to_datetime64_returns_monday(self):
        result = self.g.to_datetime64("2023-W02")
        assert result == np.datetime64("2023-01-09", "D")

    def test_sort_order(self):
        keys = ["2023-W10", "2023-W02", "2023-W03"]
        assert sorted(keys) == ["2023-W02", "2023-W03", "2023-W10"]

    def test_same_week_different_days(self):
        # Monday and Friday of the same week map to the same key
        key_mon = self.g.group_key("2023-01-09T00:00:00Z")
        key_fri = self.g.group_key("2023-01-13T00:00:00Z")
        assert key_mon == key_fri

    def test_adjacent_weeks(self):
        key_sun = self.g.group_key("2023-01-08T23:59:59Z")  # Sunday of W01
        key_mon = self.g.group_key("2023-01-09T00:00:00Z")  # Monday of W02
        assert key_sun == "2023-W01"
        assert key_mon == "2023-W02"


# ---------------------------------------------------------------------------
# _MonthGrouper
# ---------------------------------------------------------------------------


class TestMonthGrouper:
    def setup_method(self):
        self.g = _MonthGrouper()

    def test_group_key(self):
        assert self.g.group_key("2023-01-15T10:30:00Z") == "2023-01"

    def test_datetime_filter_january(self):
        assert self.g.datetime_filter("2023-01") == "2023-01-01/2023-01-31"

    def test_datetime_filter_february_non_leap(self):
        assert self.g.datetime_filter("2023-02") == "2023-02-01/2023-02-28"

    def test_datetime_filter_february_leap(self):
        assert self.g.datetime_filter("2024-02") == "2024-02-01/2024-02-29"

    def test_datetime_filter_april(self):
        assert self.g.datetime_filter("2023-04") == "2023-04-01/2023-04-30"

    def test_to_datetime64(self):
        result = self.g.to_datetime64("2023-01")
        assert result == np.datetime64("2023-01-01", "D")

    def test_sort_order(self):
        keys = ["2023-06", "2023-01", "2022-12"]
        assert sorted(keys) == ["2022-12", "2023-01", "2023-06"]

    def test_same_month_different_days(self):
        key1 = self.g.group_key("2023-01-01T00:00Z")
        key2 = self.g.group_key("2023-01-31T23:59Z")
        assert key1 == key2


# ---------------------------------------------------------------------------
# _YearGrouper
# ---------------------------------------------------------------------------


class TestYearGrouper:
    def setup_method(self):
        self.g = _YearGrouper()

    def test_group_key(self):
        assert self.g.group_key("2023-06-15T10:00:00Z") == "2023"

    def test_datetime_filter(self):
        assert self.g.datetime_filter("2023") == "2023-01-01/2023-12-31"

    def test_to_datetime64(self):
        result = self.g.to_datetime64("2023")
        assert result == np.datetime64("2023-01-01", "D")

    def test_sort_order(self):
        keys = ["2024", "2022", "2023"]
        assert sorted(keys) == ["2022", "2023", "2024"]

    def test_same_year_different_months(self):
        key1 = self.g.group_key("2023-01-01T00:00Z")
        key2 = self.g.group_key("2023-12-31T23:59Z")
        assert key1 == key2


# ---------------------------------------------------------------------------
# _FixedDayGrouper
# ---------------------------------------------------------------------------


class TestFixedDayGrouper:
    def test_p16d_same_bucket(self):
        g = _FixedDayGrouper(16)
        # Items a few days apart within the same 16-day window
        key1 = g.group_key("2023-01-10T00:00Z")
        key2 = g.group_key("2023-01-12T00:00Z")
        assert key1 == key2

    def test_p16d_adjacent_buckets(self):
        g = _FixedDayGrouper(16)
        # Derive the start of a bucket from the epoch (2000-01-01)
        # Bucket 0: 2000-01-01 .. 2000-01-16
        key_last_of_bucket = g.group_key("2000-01-16T23:59Z")
        key_first_of_next = g.group_key("2000-01-17T00:00Z")
        assert key_last_of_bucket != key_first_of_next
        assert sorted([key_last_of_bucket, key_first_of_next])[0] == key_last_of_bucket

    def test_datetime_filter_range(self):
        g = _FixedDayGrouper(16)
        key = g.group_key("2000-01-01T00:00Z")
        filt = g.datetime_filter(key)
        start, end = filt.split("/")
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        assert (end_date - start_date).days == 15  # 16-day window (inclusive)

    def test_to_datetime64_returns_bucket_start(self):
        g = _FixedDayGrouper(16)
        key = g.group_key("2000-01-10T00:00Z")
        coord = g.to_datetime64(key)
        # Epoch start 2000-01-01 is the first bucket boundary
        assert coord == np.datetime64("2000-01-01", "D")

    def test_determinism(self):
        # Same period string always produces identical bucket boundaries
        g1 = _FixedDayGrouper(16)
        g2 = _FixedDayGrouper(16)
        dt = "2023-01-15T00:00Z"
        assert g1.group_key(dt) == g2.group_key(dt)
        assert g1.datetime_filter(g1.group_key(dt)) == g2.datetime_filter(
            g2.group_key(dt)
        )

    def test_p5d_sort_order(self):
        g = _FixedDayGrouper(5)
        # Dates must be in chronological order for the assertion to hold.
        dates = ["2023-01-05T00:00Z", "2023-03-01T00:00Z", "2023-06-15T00:00Z"]
        keys = [g.group_key(dt) for dt in dates]
        assert sorted(keys) == keys  # already sorted by date, should sort the same


# ---------------------------------------------------------------------------
# grouper_from_period (factory)
# ---------------------------------------------------------------------------


class TestGrouperFromPeriod:
    def test_p1d_returns_day_grouper(self):
        assert isinstance(grouper_from_period("P1D"), _DayGrouper)

    def test_p16d_returns_fixed_day_grouper(self):
        g = grouper_from_period("P16D")
        assert isinstance(g, _FixedDayGrouper)

    def test_p5d_returns_fixed_day_grouper(self):
        assert isinstance(grouper_from_period("P5D"), _FixedDayGrouper)

    def test_p1w_returns_week_grouper(self):
        assert isinstance(grouper_from_period("P1W"), _WeekGrouper)

    def test_p1m_returns_month_grouper(self):
        assert isinstance(grouper_from_period("P1M"), _MonthGrouper)

    def test_p1y_returns_year_grouper(self):
        assert isinstance(grouper_from_period("P1Y"), _YearGrouper)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            grouper_from_period("invalid")

    def test_pt1h_raises(self):
        with pytest.raises(ValueError):
            grouper_from_period("PT1H")

    def test_p2m_raises(self):
        # P2M (2 months) is not supported
        with pytest.raises(ValueError):
            grouper_from_period("P2M")

    def test_p2y_raises(self):
        with pytest.raises(ValueError):
            grouper_from_period("P2Y")


# ---------------------------------------------------------------------------
# Sort invariant across all groupers
# ---------------------------------------------------------------------------


SAMPLE_DATETIMES = [
    "2022-12-28T00:00:00Z",
    "2023-01-02T00:00:00Z",
    "2023-01-09T00:00:00Z",
    "2023-02-15T00:00:00Z",
    "2023-06-30T00:00:00Z",
    "2023-12-31T23:59:59Z",
    "2024-01-01T00:00:00Z",
]


@pytest.mark.parametrize(
    "grouper",
    [
        _DayGrouper(),
        _WeekGrouper(),
        _MonthGrouper(),
        _YearGrouper(),
        _FixedDayGrouper(16),
        _FixedDayGrouper(5),
    ],
)
def test_sort_invariant(grouper):
    """Sorted group keys must match the temporal order of the input datetimes."""
    # SAMPLE_DATETIMES are already in chronological order.
    keys = [grouper.group_key(dt) for dt in SAMPLE_DATETIMES]
    # After deduplication, sorted order must still be non-decreasing relative
    # to the original chronological order.
    unique_keys = list(dict.fromkeys(keys))  # deduplicate while preserving order
    assert sorted(unique_keys) == unique_keys
