"""Private temporal grouping logic for STAC item time-step bucketing."""

from __future__ import annotations

import calendar
import re
from abc import ABC, abstractmethod
from datetime import date, timedelta

import numpy as np

# Epoch used for epoch-aligned fixed-day-count periods (PnD where n > 1).
_EPOCH = date(2000, 1, 1)

_ISO_DURATION_RE = re.compile(r"^P(\d+)(D|W|M|Y)$")


def _parse_iso_duration(s: str) -> tuple[int, str]:
    """Parse a simple ISO 8601 duration string into ``(count, unit)``.

    Only the ``PnD``, ``PnW``, ``PnM``, and ``PnY`` forms are supported.

    Args:
        s: ISO 8601 duration string, e.g. ``"P1D"``, ``"P16D"``, ``"P1M"``.

    Returns:
        ``(count, unit)`` where *unit* is one of ``"D"``, ``"W"``, ``"M"``,
        ``"Y"``.

    Raises:
        ValueError: If *s* does not match the supported pattern.

    """
    m = _ISO_DURATION_RE.match(s)
    if not m:
        raise ValueError(
            f"Unsupported ISO 8601 duration {s!r}. "
            "Expected PnD, PnW, PnM, or PnY (e.g. 'P1D', 'P1M', 'P16D')."
        )
    return int(m.group(1)), m.group(2)


class _TemporalGrouper(ABC):
    """Abstract base for temporal grouping strategies.

    Each subclass buckets STAC item datetimes into discrete time steps,
    producing a group label (used for sorting and deduplication), a
    ``rustac``-compatible datetime filter string, and a ``numpy.datetime64``
    coordinate value.

    All built-in implementations use ``datetime64[D]`` precision for
    coordinate values to keep the xarray time axis consistent.

    """

    @abstractmethod
    def group_key(self, datetime_str: str) -> str:
        """Map a STAC item datetime string to a group label.

        Labels must sort lexicographically in temporal order.

        Args:
            datetime_str: RFC 3339 datetime string from a STAC item's
                ``properties.datetime`` or ``properties.start_datetime``.

        Returns:
            An opaque string label for the group this item belongs to.

        """
        ...

    @abstractmethod
    def datetime_filter(self, group_key: str) -> str:
        """Return a ``rustac``-compatible datetime filter for a group.

        Args:
            group_key: A label previously returned by :meth:`group_key`.

        Returns:
            An RFC 3339 datetime or range string suitable for passing to
            ``rustac.search_sync(..., datetime=...)``.

        """
        ...

    @abstractmethod
    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Map a group label to an xarray time coordinate value.

        Args:
            group_key: A label previously returned by :meth:`group_key`.

        Returns:
            A ``numpy.datetime64`` value at day precision (``datetime64[D]``).

        """
        ...


class _DayGrouper(_TemporalGrouper):
    """Group items by calendar day (``P1D``).

    This is the default and preserves the existing behaviour of truncating
    each item datetime to ``YYYY-MM-DD``.

    """

    def group_key(self, datetime_str: str) -> str:
        """Return the ``YYYY-MM-DD`` portion of *datetime_str*."""
        return datetime_str[:10]

    def datetime_filter(self, group_key: str) -> str:
        """Return *group_key* unchanged; rustac accepts bare date strings."""
        return group_key

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return ``numpy.datetime64(group_key, "D")``."""
        return np.datetime64(group_key, "D")


class _WeekGrouper(_TemporalGrouper):
    """Group items by ISO 8601 calendar week (``P1W``), anchored on Monday."""

    def group_key(self, datetime_str: str) -> str:
        """Return an ``YYYY-Www`` ISO week label for *datetime_str*."""
        d = date.fromisoformat(datetime_str[:10])
        iso = d.isocalendar()
        # Use iso.year (not d.year) to handle year-boundary weeks correctly:
        # e.g. 2022-01-02 belongs to ISO week 2021-W52, not 2022-W00.
        return f"{iso.year}-W{iso.week:02d}"

    def datetime_filter(self, group_key: str) -> str:
        """Return a ``Monday/Sunday`` RFC 3339 range for *group_key*."""
        monday = self._monday(group_key)
        sunday = monday + timedelta(days=6)
        return f"{monday.isoformat()}/{sunday.isoformat()}"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return the Monday of the ISO week as ``datetime64[D]``."""
        return np.datetime64(self._monday(group_key).isoformat(), "D")

    @staticmethod
    def _monday(group_key: str) -> date:
        """Return the Monday ``date`` for an ``YYYY-Www`` key."""
        year = int(group_key[:4])
        week = int(group_key[6:])
        # ISO week 1 always contains January 4th.
        jan4 = date(year, 1, 4)
        week1_monday = jan4 - timedelta(days=jan4.weekday())
        return week1_monday + timedelta(weeks=week - 1)


class _MonthGrouper(_TemporalGrouper):
    """Group items by calendar month (``P1M``)."""

    def group_key(self, datetime_str: str) -> str:
        """Return the ``YYYY-MM`` portion of *datetime_str*."""
        return datetime_str[:7]

    def datetime_filter(self, group_key: str) -> str:
        """Return a ``YYYY-MM-01/YYYY-MM-DD`` range covering the full month."""
        year, month = int(group_key[:4]), int(group_key[5:7])
        last_day = calendar.monthrange(year, month)[1]
        return f"{group_key}-01/{group_key}-{last_day:02d}"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return the first of the month as ``datetime64[D]``."""
        return np.datetime64(f"{group_key}-01", "D")


class _YearGrouper(_TemporalGrouper):
    """Group items by calendar year (``P1Y``)."""

    def group_key(self, datetime_str: str) -> str:
        """Return the ``YYYY`` portion of *datetime_str*."""
        return datetime_str[:4]

    def datetime_filter(self, group_key: str) -> str:
        """Return a ``YYYY-01-01/YYYY-12-31`` range covering the full year."""
        return f"{group_key}-01-01/{group_key}-12-31"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return January 1st of the year as ``datetime64[D]``."""
        return np.datetime64(f"{group_key}-01-01", "D")


class _FixedDayGrouper(_TemporalGrouper):
    """Group items into fixed-length windows of *n_days* days.

    Windows are aligned to :data:`_EPOCH` (2000-01-01) so that the same
    ``time_period`` always produces identical bucket boundaries regardless of
    which dates happen to appear in a query.

    Args:
        n_days: Window length in days (must be >= 2).

    """

    def __init__(self, n_days: int) -> None:
        """Initialise with a fixed window length."""
        self._n = n_days

    def _bucket(self, datetime_str: str) -> int:
        """Return the zero-based bucket index for *datetime_str*."""
        d = date.fromisoformat(datetime_str[:10])
        return (d - _EPOCH).days // self._n

    def group_key(self, datetime_str: str) -> str:
        """Return a zero-padded decimal bucket index as the group label.

        The index is zero-padded to 6 digits so that lexicographic sort
        matches numeric/temporal order for up to 10^6 buckets.

        """
        return f"{self._bucket(datetime_str):06d}"

    def datetime_filter(self, group_key: str) -> str:
        """Return a ``start/end`` RFC 3339 range for the bucket."""
        bucket = int(group_key)
        start = _EPOCH + timedelta(days=bucket * self._n)
        end = start + timedelta(days=self._n - 1)
        return f"{start.isoformat()}/{end.isoformat()}"

    def to_datetime64(self, group_key: str) -> np.datetime64:
        """Return the start date of the bucket as ``datetime64[D]``."""
        bucket = int(group_key)
        start = _EPOCH + timedelta(days=bucket * self._n)
        return np.datetime64(start.isoformat(), "D")


def grouper_from_period(time_period: str) -> _TemporalGrouper:
    """Return a :class:`_TemporalGrouper` for an ISO 8601 duration string.

    Args:
        time_period: ISO 8601 duration string, e.g. ``"P1D"``, ``"P1M"``,
            ``"P16D"``.  Supported forms: ``PnD``, ``PnW``, ``PnM``, ``PnY``.

    Returns:
        A :class:`_TemporalGrouper` instance appropriate for *time_period*.

    Raises:
        ValueError: If *time_period* is not a recognised duration string.

    """
    count, unit = _parse_iso_duration(time_period)

    if unit == "D" and count == 1:
        return _DayGrouper()
    if unit == "D":
        return _FixedDayGrouper(count)
    if unit == "W" and count == 1:
        return _WeekGrouper()
    if unit == "W":
        return _FixedDayGrouper(count * 7)
    if unit == "M" and count == 1:
        return _MonthGrouper()
    if unit == "Y" and count == 1:
        return _YearGrouper()

    raise ValueError(
        f"Unsupported time_period {time_period!r}. "
        "Supported values: 'P1D', 'PnD' (n>1), 'P1W', 'P1M', 'P1Y'."
    )
