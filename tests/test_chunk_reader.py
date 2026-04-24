"""Tests for _chunk_reader helpers: _select_overview, _native_window, and mosaic functions."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
from affine import Affine
from pyproj import CRS

from lazycogs._chunk_reader import (
    _apply_bands_with_warp_cache,
    _native_window,
    _select_overview,
    async_mosaic_chunk,
)
from lazycogs._mosaic_methods import FirstMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_geotiff(native_res: float, overview_resolutions: list[float]) -> MagicMock:
    """Build a minimal GeoTIFF mock with the given resolutions."""
    geotiff = MagicMock()
    geotiff.transform = Affine(native_res, 0.0, 0.0, 0.0, -native_res, 0.0)

    overviews = []
    for res in overview_resolutions:
        ov = MagicMock()
        ov.transform = Affine(res, 0.0, 0.0, 0.0, -res, 0.0)
        overviews.append(ov)

    geotiff.overviews = overviews
    return geotiff


def _mock_reader(transform: Affine) -> MagicMock:
    """Build a minimal GeoTIFF/Overview mock with the given transform."""
    reader = MagicMock()
    reader.transform = transform
    return reader


# ---------------------------------------------------------------------------
# _select_overview
# ---------------------------------------------------------------------------


def test_select_overview_no_overviews_returns_none():
    """Returns None when the file has no overviews."""
    geotiff = _mock_geotiff(10.0, [])
    assert _select_overview(geotiff, 100.0) is None


def test_select_overview_target_finer_than_native_returns_none():
    """Returns None when the requested resolution is finer than native."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0])
    assert _select_overview(geotiff, 5.0) is None


def test_select_overview_target_equal_to_native_returns_none():
    """Returns None when the requested resolution equals native."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0])
    assert _select_overview(geotiff, 10.0) is None


def test_select_overview_returns_coarsest_non_upsampling_overview():
    """Returns the coarsest overview whose resolution <= target."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0, 80.0])
    ov = _select_overview(geotiff, 30.0)
    # Target is 30 m → coarsest overview <= 30 m is the 20 m one (index 0)
    assert ov is geotiff.overviews[0]


def test_select_overview_target_between_native_and_finest_returns_none():
    """Returns None when target falls between native res and the finest overview."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0, 80.0])
    # 15 m > 10 m native, but 20 m finest overview > 15 m → upsampling; use full res
    assert _select_overview(geotiff, 15.0) is None


def test_select_overview_exact_match():
    """Returns the overview whose resolution exactly matches the target."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0])
    ov = _select_overview(geotiff, 20.0)
    assert ov is geotiff.overviews[0]


def test_select_overview_target_coarser_than_all_overviews():
    """When target is coarser than all overviews, returns the coarsest."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0, 80.0])
    ov = _select_overview(geotiff, 200.0)
    assert ov is geotiff.overviews[-1]


# ---------------------------------------------------------------------------
# _native_window
# ---------------------------------------------------------------------------


def test_native_window_full_coverage():
    """A bbox that covers the full image returns a window matching the image."""
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)  # 4-px tall, any width
    reader = _mock_reader(transform)
    win = _native_window(reader, (0.0, 0.0, 4.0, 4.0), width=4, height=4)
    assert win is not None
    assert win.col_off == 0
    assert win.row_off == 0
    assert win.width == 4
    assert win.height == 4


def test_native_window_sub_region():
    """A bbox covering the bottom-right quadrant returns the correct window."""
    # 8×8 image, 1 m resolution, origin top-left at (0, 8)
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 8.0)
    reader = _mock_reader(transform)
    # Bottom-right quadrant: x=[4,8], y=[0,4]
    win = _native_window(reader, (4.0, 0.0, 8.0, 4.0), width=8, height=8)
    assert win is not None
    assert win.col_off == 4
    assert win.row_off == 4
    assert win.width == 4
    assert win.height == 4


def test_native_window_bbox_outside_returns_none():
    """A bbox entirely outside the image returns None."""
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    reader = _mock_reader(transform)
    # Image covers x=[0,4], bbox is at x=[10,14]
    win = _native_window(reader, (10.0, 0.0, 14.0, 4.0), width=4, height=4)
    assert win is None


def test_native_window_clamped_to_image_bounds():
    """A bbox that extends beyond image edges is clamped to valid pixels."""
    # 4×4 image
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    reader = _mock_reader(transform)
    # Bbox extends 2 pixels beyond the right and bottom edges
    win = _native_window(reader, (2.0, -2.0, 6.0, 2.0), width=4, height=4)
    assert win is not None
    assert win.col_off == 2
    assert win.col_off + win.width <= 4  # clamped at image width
    assert win.row_off + win.height <= 4  # clamped at image height


# ---------------------------------------------------------------------------
# async_mosaic_chunk concurrency
# ---------------------------------------------------------------------------


def test_async_mosaic_chunk_limits_concurrent_reads():
    """No more than max_concurrent_reads items are read concurrently."""
    chunk_width, chunk_height = 4, 4
    chunk_affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    dst_crs = CRS.from_epsg(4326)
    bands = ["B01"]
    n_items = 20
    max_concurrent = 5

    peak_concurrent = [0]
    current_concurrent = [0]

    async def _fake_read_item_band(*args, **kwargs):
        current_concurrent[0] += 1
        peak_concurrent[0] = max(peak_concurrent[0], current_concurrent[0])
        await asyncio.sleep(0)  # yield to allow other coroutines to increment
        current_concurrent[0] -= 1
        return {
            b: (np.ones((1, chunk_height, chunk_width), dtype=np.float32), None)
            for b in bands
        }

    items = [{"id": f"item-{i}", "assets": {}} for i in range(n_items)]

    with patch(
        "lazycogs._chunk_reader._read_item_band",
        side_effect=_fake_read_item_band,
    ):
        asyncio.run(
            async_mosaic_chunk(
                items=items,
                bands=bands,
                chunk_affine=chunk_affine,
                dst_crs=dst_crs,
                chunk_width=chunk_width,
                chunk_height=chunk_height,
                nodata=None,
                max_concurrent_reads=max_concurrent,
            )
        )

    assert peak_concurrent[0] <= max_concurrent


# ---------------------------------------------------------------------------
# _apply_bands_with_warp_cache
# ---------------------------------------------------------------------------


def _make_raster(transform: Affine, value: float, h: int = 4, w: int = 4) -> MagicMock:
    raster = MagicMock()
    raster.transform = transform
    raster.data = np.full((1, h, w), value, dtype=np.float32)
    return raster


def test_apply_bands_with_warp_cache_shared_geometry():
    """Bands with the same transform/CRS share a single warp map computation."""
    crs = CRS.from_epsg(4326)
    # Offset the source transform slightly so it differs from dst_transform and
    # the warp path (rather than the identity fast-path) is exercised.
    transform = Affine(1.0, 0.0, 0.5, 0.0, -1.0, 4.0)
    dst_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)

    raster_a = _make_raster(transform, 1.0)
    raster_b = _make_raster(transform, 2.0)

    warp_map_calls = []

    def _spy_compute_warp_map(*args, **kwargs):
        from lazycogs._reproject import compute_warp_map as _real

        result = _real(*args, **kwargs)
        warp_map_calls.append(True)
        return result

    with patch(
        "lazycogs._chunk_reader.compute_warp_map", side_effect=_spy_compute_warp_map
    ):
        results = _apply_bands_with_warp_cache(
            [("B01", raster_a, crs, None), ("B02", raster_b, crs, None)],
            dst_transform,
            crs,
            dst_width=4,
            dst_height=4,
        )

    # Same transform → warp map computed exactly once.
    assert len(warp_map_calls) == 1
    assert set(results) == {"B01", "B02"}
    np.testing.assert_array_equal(results["B01"][0], 1.0)
    np.testing.assert_array_equal(results["B02"][0], 2.0)


def test_apply_bands_with_warp_cache_different_geometry():
    """Bands with different transforms each compute their own warp map."""
    crs = CRS.from_epsg(4326)
    # Both source transforms differ from dst_transform so the warp path is
    # exercised for each band (fast-path is not triggered).
    transform_a = Affine(1.0, 0.0, 0.5, 0.0, -1.0, 4.0)
    transform_b = Affine(2.0, 0.0, 0.0, 0.0, -2.0, 8.0)
    dst_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)

    raster_a = _make_raster(transform_a, 1.0)
    raster_b = _make_raster(transform_b, 2.0, h=2, w=2)

    warp_map_calls = []

    def _spy_compute_warp_map(*args, **kwargs):
        from lazycogs._reproject import compute_warp_map as _real

        result = _real(*args, **kwargs)
        warp_map_calls.append(True)
        return result

    with patch(
        "lazycogs._chunk_reader.compute_warp_map", side_effect=_spy_compute_warp_map
    ):
        results = _apply_bands_with_warp_cache(
            [("B01", raster_a, crs, None), ("B02", raster_b, crs, None)],
            dst_transform,
            crs,
            dst_width=4,
            dst_height=4,
        )

    # Different transforms → two separate warp map computations.
    assert len(warp_map_calls) == 2
    assert set(results) == {"B01", "B02"}


def test_apply_bands_with_warp_cache_shared_across_calls():
    """A shared external cache reuses warp maps across separate calls (e.g. time steps)."""
    crs = CRS.from_epsg(4326)
    # Source transform offset from dst so the warp path is exercised.
    transform = Affine(1.0, 0.0, 0.5, 0.0, -1.0, 4.0)
    dst_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)

    raster = _make_raster(transform, 1.0)
    shared_cache: dict = {}

    warp_map_calls = []

    def _spy_compute_warp_map(*args, **kwargs):
        from lazycogs._reproject import compute_warp_map as _real

        result = _real(*args, **kwargs)
        warp_map_calls.append(True)
        return result

    with patch(
        "lazycogs._chunk_reader.compute_warp_map", side_effect=_spy_compute_warp_map
    ):
        _apply_bands_with_warp_cache(
            [("B01", raster, crs, None)],
            dst_transform,
            crs,
            dst_width=4,
            dst_height=4,
            warp_cache=shared_cache,
        )
        _apply_bands_with_warp_cache(
            [("B01", raster, crs, None)],
            dst_transform,
            crs,
            dst_width=4,
            dst_height=4,
            warp_cache=shared_cache,
        )

    # Warp map computed only once despite two separate calls.
    assert len(warp_map_calls) == 1
    assert len(shared_cache) == 1


# ---------------------------------------------------------------------------
# async_mosaic_chunk (multi-band)
# ---------------------------------------------------------------------------


def test_async_mosaic_chunk_returns_all_bands():
    """Returns a dict with one entry per requested band."""
    chunk_width, chunk_height = 4, 4
    chunk_affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    dst_crs = CRS.from_epsg(4326)
    bands = ["B01", "B02", "B03"]

    async def _fake_read_item_band(*args, **kwargs):
        return {
            b: (np.ones((1, chunk_height, chunk_width), dtype=np.float32), None)
            for b in bands
        }

    items = [{"id": "item-0", "assets": {}}]

    with patch(
        "lazycogs._chunk_reader._read_item_band", side_effect=_fake_read_item_band
    ):
        result = asyncio.run(
            async_mosaic_chunk(
                items=items,
                bands=bands,
                chunk_affine=chunk_affine,
                dst_crs=dst_crs,
                chunk_width=chunk_width,
                chunk_height=chunk_height,
            )
        )

    assert set(result.keys()) == set(bands)
    for b in bands:
        assert result[b].shape == (1, chunk_height, chunk_width)


def test_async_mosaic_chunk_early_exit():
    """All bands filled on first item → remaining items are skipped."""
    chunk_width, chunk_height = 4, 4
    chunk_affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    dst_crs = CRS.from_epsg(4326)
    bands = ["B01", "B02"]
    reads_executed = [0]
    n_items = 20

    async def _fake_read_item_band(*args, **kwargs):
        reads_executed[0] += 1
        # Yield so the semaphore can throttle and the main coroutine can wake
        # on the first completion instead of processing every task in one
        # synchronous burst.
        await asyncio.sleep(0)
        return {
            b: (np.ones((1, chunk_height, chunk_width), dtype=np.float32), None)
            for b in bands
        }

    items = [{"id": f"item-{i}", "assets": {}} for i in range(n_items)]

    with patch(
        "lazycogs._chunk_reader._read_item_band", side_effect=_fake_read_item_band
    ):
        asyncio.run(
            async_mosaic_chunk(
                items=items,
                bands=bands,
                chunk_affine=chunk_affine,
                dst_crs=dst_crs,
                chunk_width=chunk_width,
                chunk_height=chunk_height,
                mosaic_method_cls=FirstMethod,
                max_concurrent_reads=5,
            )
        )

    # All tasks are launched up front, but the semaphore caps in-flight reads
    # at max_concurrent_reads.  Once every band's FirstMethod is satisfied,
    # pending tasks are cancelled, so strictly fewer than n_items reads execute.
    assert reads_executed[0] < n_items
