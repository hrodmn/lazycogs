"""Tests for the lazycogs._explain module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from rustac import DuckdbClient
import xarray as xr
from affine import Affine
from pyproj import CRS

from lazycogs._backend import StacBackendArray
from lazycogs._explain import (
    ChunkRead,
    CogRead,
    ExplainPlan,
    _compute_chunk_bbox_4326,
    _infer_chunk_sizes,
    _iter_spatial_chunks,
    _roi_pixel_offsets,
)
from lazycogs._mosaic_methods import FirstMethod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wgs84() -> CRS:
    return CRS.from_epsg(4326)


@pytest.fixture
def epsg5070() -> CRS:
    return CRS.from_epsg(5070)


def _make_backend(
    crs: CRS,
    dates: list[str] | None = None,
    parquet_path: str = "/tmp/fake.parquet",
    band: str = "red",
    dst_width: int = 10,
    dst_height: int = 10,
    affine: Affine | None = None,
) -> StacBackendArray:
    """Return a minimal StacBackendArray for unit testing."""
    if dates is None:
        dates = ["2023-01-01/2023-01-01", "2023-01-02/2023-01-02"]
    if affine is None:
        affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    return StacBackendArray(
        parquet_path=parquet_path,
        duckdb_client=DuckdbClient(),
        band=band,
        dates=dates,
        dst_affine=affine,
        dst_crs=crs,
        bbox_4326=[0.0, 0.0, 10.0, 10.0],
        sort_by=None,
        filter=None,
        ids=None,
        dst_width=dst_width,
        dst_height=dst_height,
        dtype=np.dtype("float32"),
        nodata=-9999.0,
        shape=(len(dates), dst_height, dst_width),
        mosaic_method_cls=FirstMethod,
    )


def _make_da_with_backends(
    crs: CRS,
    dates: list[str],
    time_coords: list[np.datetime64],
    bands: list[str],
    width: int = 10,
    height: int = 10,
    affine: Affine | None = None,
) -> xr.DataArray:
    """Return a minimal DataArray with stac_cog explain attrs attached."""
    if affine is None:
        resolution = 1.0
        affine = Affine(resolution, 0.0, 0.0, 0.0, -resolution, float(height))

    backends = [
        _make_backend(
            crs,
            dates=dates,
            band=band,
            dst_width=width,
            dst_height=height,
            affine=affine,
        )
        for band in bands
    ]

    time_coord = np.array(time_coords, dtype="datetime64[D]")
    resolution = affine.a

    # Build coordinates matching the grid convention: x ascending, y ascending
    x_coords = np.array([affine.c + (i + 0.5) * resolution for i in range(width)])
    y_coords = np.array(
        [affine.f + (height - 1 - i + 0.5) * affine.e for i in range(height)]
    )
    # y ascending (south to north)
    y_coords = np.sort(y_coords)

    da = xr.DataArray(
        np.zeros((len(bands), len(dates), height, width), dtype="float32"),
        coords={
            "band": bands,
            "time": time_coord,
            "y": y_coords,
            "x": x_coords,
        },
        dims=("band", "time", "y", "x"),
    )
    da.attrs["_stac_backends"] = backends
    da.attrs["_stac_time_coords"] = time_coord
    return da


# ---------------------------------------------------------------------------
# _iter_spatial_chunks
# ---------------------------------------------------------------------------


def test_iter_spatial_chunks_exact_fit():
    """A 4x4 grid with 2x2 chunks yields 4 tiles."""
    affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    tiles = list(_iter_spatial_chunks(affine, 4, 4, 2, 2))
    assert len(tiles) == 4
    rows = {t[0] for t in tiles}
    cols = {t[1] for t in tiles}
    assert rows == {0, 1}
    assert cols == {0, 1}
    # All tiles have full size
    for _, _, _, w, h in tiles:
        assert w == 2
        assert h == 2


def test_iter_spatial_chunks_edge_tiles():
    """A 10x10 grid with chunk=4 yields 3x3=9 tiles; edge tiles are smaller."""
    affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    tiles = list(_iter_spatial_chunks(affine, 10, 10, 4, 4))
    assert len(tiles) == 9  # ceil(10/4) = 3 in each dimension

    # Collect widths and heights by column/row
    col_widths = {}
    row_heights = {}
    for row, col, _, w, h in tiles:
        col_widths[col] = w
        row_heights[row] = h

    assert col_widths[0] == 4
    assert col_widths[1] == 4
    assert col_widths[2] == 2  # 10 - 8 = 2
    assert row_heights[0] == 4
    assert row_heights[1] == 4
    assert row_heights[2] == 2


def test_iter_spatial_chunks_single_tile():
    """When chunk >= extent, a single tile covering the whole area is yielded."""
    affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
    tiles = list(_iter_spatial_chunks(affine, 5, 5, 100, 100))
    assert len(tiles) == 1
    _, _, _, w, h = tiles[0]
    assert w == 5
    assert h == 5


def test_iter_spatial_chunks_affine_translation():
    """Tile affines are offset correctly from the ROI affine."""
    affine = Affine(2.0, 0.0, 10.0, 0.0, -2.0, 20.0)
    tiles = list(_iter_spatial_chunks(affine, 4, 4, 2, 2))
    assert len(tiles) == 4

    # First tile: same as ROI affine
    _, _, tile_affine_00, _, _ = tiles[0]
    assert tile_affine_00.c == pytest.approx(10.0)
    assert tile_affine_00.f == pytest.approx(20.0)

    # Second tile in x direction: offset by 2 pixels * 2 units/px = 4 units
    _, _, tile_affine_01, _, _ = tiles[1]
    assert tile_affine_01.c == pytest.approx(14.0)
    assert tile_affine_01.f == pytest.approx(20.0)

    # Second tile in y direction: offset by 2 pixels * 2 units/px = 4 units down
    _, _, tile_affine_10, _, _ = tiles[2]
    assert tile_affine_10.c == pytest.approx(10.0)
    assert tile_affine_10.f == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# _compute_chunk_bbox_4326
# ---------------------------------------------------------------------------


def test_compute_chunk_bbox_4326_wgs84(wgs84):
    """In EPSG:4326 the bbox is returned unchanged."""
    affine = Affine(1.0, 0.0, 10.0, 0.0, -1.0, 50.0)
    bbox = _compute_chunk_bbox_4326(affine, 4, 1, wgs84)
    assert bbox == pytest.approx([10.0, 49.0, 14.0, 50.0])


def test_compute_chunk_bbox_4326_projected(epsg5070):
    """Projected CRS results are reprojected to WGS84."""
    affine = Affine(100.0, 0.0, 300000.0, 0.0, -100.0, 2700000.0)
    bbox = _compute_chunk_bbox_4326(affine, 10, 10, epsg5070)
    # Just verify it's a plausible lon/lat range for EPSG:5070 coordinates
    assert len(bbox) == 4
    minx, miny, maxx, maxy = bbox
    assert -180 <= minx < maxx <= 180
    assert -90 <= miny < maxy <= 90


# ---------------------------------------------------------------------------
# _infer_chunk_sizes
# ---------------------------------------------------------------------------


def test_infer_chunk_sizes_no_dask(wgs84):
    """Without dask the full spatial extent is returned as one chunk."""
    da = _make_da_with_backends(
        wgs84,
        dates=["2023-01-01/2023-01-01"],
        time_coords=[np.datetime64("2023-01-01", "D")],
        bands=["red"],
        width=100,
        height=80,
    )
    chunk_h, chunk_w = _infer_chunk_sizes(da)
    assert chunk_h == 80
    assert chunk_w == 100


def test_infer_chunk_sizes_with_dask(wgs84):
    """With dask the first chunk size is used."""
    da = _make_da_with_backends(
        wgs84,
        dates=["2023-01-01/2023-01-01"],
        time_coords=[np.datetime64("2023-01-01", "D")],
        bands=["red"],
        width=100,
        height=80,
    )
    da = da.chunk({"y": 32, "x": 64})
    chunk_h, chunk_w = _infer_chunk_sizes(da)
    assert chunk_h == 32
    assert chunk_w == 64


# ---------------------------------------------------------------------------
# _roi_pixel_offsets
# ---------------------------------------------------------------------------


def test_roi_pixel_offsets_full_extent(wgs84):
    """Full-extent DataArray yields zero offsets and full dimensions."""
    affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    da = _make_da_with_backends(
        wgs84,
        dates=["2023-01-01/2023-01-01"],
        time_coords=[np.datetime64("2023-01-01", "D")],
        bands=["red"],
        width=10,
        height=10,
        affine=affine,
    )
    backend = da.attrs["_stac_backends"][0]
    x_start, y_start_physical, roi_w, roi_h = _roi_pixel_offsets(da, backend)
    assert x_start == 0
    assert y_start_physical == 0
    assert roi_w == 10
    assert roi_h == 10


# ---------------------------------------------------------------------------
# ExplainPlan
# ---------------------------------------------------------------------------


def _make_plan(n_bands: int = 2, n_time: int = 3, n_items: int = 1) -> ExplainPlan:
    """Return a minimal ExplainPlan for display tests."""
    chunk_reads = []
    for band in [f"band{i}" for i in range(n_bands)]:
        for t in range(n_time):
            items = [
                CogRead(
                    item_id=f"item-{t}-{j}",
                    asset_key=band,
                    href=f"s3://bucket/item-{t}-{j}.tif",
                )
                for j in range(n_items)
            ]
            chunk_reads.append(
                ChunkRead(
                    band=band,
                    time_index=t,
                    date_filter=f"2023-01-0{t + 1}/2023-01-0{t + 1}",
                    time_coord=np.datetime64(f"2023-01-0{t + 1}", "D"),
                    chunk_row=0,
                    chunk_col=0,
                    chunk_affine=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
                    chunk_width=10,
                    chunk_height=10,
                    cog_reads=items,
                )
            )
    return ExplainPlan(
        href="/tmp/fake.parquet",
        crs="EPSG:4326",
        resolution=1.0,
        bands=[f"band{i}" for i in range(n_bands)],
        time_coords=[np.datetime64(f"2023-01-0{t + 1}", "D") for t in range(n_time)],
        dst_width=10,
        dst_height=10,
        chunk_width=10,
        chunk_height=10,
        chunk_reads=chunk_reads,
        fetch_headers=False,
    )


def test_explain_plan_repr():
    """__repr__ renders without error and contains key counts."""
    plan = _make_plan(n_bands=2, n_time=3, n_items=1)
    r = repr(plan)
    assert "2 band(s)" in r
    assert "3 time step(s)" in r
    assert "6 chunk read(s)" in r


def test_explain_plan_summary():
    """summary() renders without error and contains expected sections."""
    plan = _make_plan(n_bands=1, n_time=2, n_items=2)
    s = plan.summary()
    assert "ExplainPlan" in s
    assert "EPSG:4326" in s
    assert "band0" in s


def test_explain_plan_summary_with_empty_chunks():
    """summary() correctly counts empty chunks."""
    plan = _make_plan(n_bands=1, n_time=2, n_items=0)
    s = plan.summary()
    assert "2" in s  # 2 empty chunks


def test_explain_plan_summary_fetch_headers():
    """summary() includes overview and window stats when fetch_headers=True."""
    chunk_reads = [
        ChunkRead(
            band="red",
            time_index=0,
            date_filter="2023-01-01/2023-01-01",
            time_coord=np.datetime64("2023-01-01", "D"),
            chunk_row=0,
            chunk_col=0,
            chunk_affine=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
            chunk_width=10,
            chunk_height=10,
            cog_reads=[
                CogRead(
                    item_id="item-0",
                    asset_key="red",
                    href="s3://bucket/item-0.tif",
                    overview_level=1,
                    overview_resolution=20.0,
                    window_col_off=0,
                    window_row_off=0,
                    window_width=256,
                    window_height=256,
                )
            ],
        )
    ]
    plan = ExplainPlan(
        href="/tmp/fake.parquet",
        crs="EPSG:4326",
        resolution=1.0,
        bands=["red"],
        time_coords=[np.datetime64("2023-01-01", "D")],
        dst_width=10,
        dst_height=10,
        chunk_width=10,
        chunk_height=10,
        chunk_reads=chunk_reads,
        fetch_headers=True,
    )
    s = plan.summary()
    assert "ovr 1" in s
    assert "256 x 256" in s
    assert "fetch_headers=True" not in s  # no hint to enable it


def test_explain_plan_to_dataframe():
    """to_dataframe() returns correct columns and row count."""
    pytest.importorskip("pandas")
    plan = _make_plan(n_bands=2, n_time=3, n_items=2)
    df = plan.to_dataframe()
    expected_cols = {
        "band",
        "time_index",
        "date_filter",
        "chunk_row",
        "chunk_col",
        "n_cog_reads",
        "item_id",
        "href",
        "overview_level",
        "window_width",
    }
    assert expected_cols.issubset(df.columns)
    # 2 bands * 3 time steps * 2 items each = 12 rows
    assert len(df) == 12


def test_explain_plan_to_dataframe_empty_chunks():
    """to_dataframe() includes one row per empty chunk."""
    pytest.importorskip("pandas")
    plan = _make_plan(n_bands=1, n_time=2, n_items=0)
    df = plan.to_dataframe()
    assert len(df) == 2
    assert df["item_id"].isna().all()


def test_explain_plan_properties():
    """total_chunk_reads, total_cog_reads, empty_chunk_count are correct."""
    plan = _make_plan(n_bands=2, n_time=3, n_items=1)
    assert plan.total_chunk_reads == 6
    assert plan.total_cog_reads == 6
    assert plan.empty_chunk_count == 0

    empty_plan = _make_plan(n_bands=1, n_time=2, n_items=0)
    assert empty_plan.empty_chunk_count == 2
    assert empty_plan.total_cog_reads == 0


# ---------------------------------------------------------------------------
# StacCogAccessor.explain() via mocked rustac
# ---------------------------------------------------------------------------


def _fake_items(band: str, n: int) -> list[dict]:
    return [
        {
            "id": f"item-{i}",
            "assets": {
                band: {"href": f"s3://bucket/item-{i}.tif"},
            },
        }
        for i in range(n)
    ]


def test_accessor_raises_on_non_stac_da():
    """explain() raises ValueError when the DataArray has no explain metadata."""
    da = xr.DataArray(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="lazycogs.open"):
        da.lazycogs.explain()


def test_accessor_explain_returns_plan(wgs84):
    """explain() returns an ExplainPlan with correct counts."""
    dates = ["2023-01-01/2023-01-01", "2023-01-02/2023-01-02"]
    time_coords = [np.datetime64("2023-01-01", "D"), np.datetime64("2023-01-02", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=4,
        height=4,
    )

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = _fake_items("red", 2)
        plan = da.lazycogs.explain()

    # 1 band * 2 time steps * 1 spatial tile (no chunking) = 2 chunk reads
    assert plan.total_chunk_reads == 2
    assert plan.total_cog_reads == 4  # 2 items per chunk * 2 chunks
    assert plan.bands == ["red"]
    assert len(plan.time_coords) == 2


def test_accessor_explain_empty_results(wgs84):
    """explain() handles chunks with zero matching items gracefully."""
    dates = ["2023-01-01/2023-01-01"]
    time_coords = [np.datetime64("2023-01-01", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=4,
        height=4,
    )

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = []
        plan = da.lazycogs.explain()

    assert plan.total_chunk_reads == 1
    assert plan.total_cog_reads == 0
    assert plan.empty_chunk_count == 1


def test_accessor_explain_with_dask_chunks(wgs84):
    """explain() uses dask chunk sizes when available."""
    dates = ["2023-01-01/2023-01-01"]
    time_coords = [np.datetime64("2023-01-01", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=8,
        height=8,
    )
    da = da.chunk({"y": 4, "x": 4})

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = []
        plan = da.lazycogs.explain()

    # 1 band * 1 time * 4 spatial tiles (2x2 grid from 8px / 4px chunks)
    assert plan.total_chunk_reads == 4
    assert plan.chunk_width == 4
    assert plan.chunk_height == 4


def test_accessor_explain_multiple_bands(wgs84):
    """explain() iterates over all active bands."""
    dates = ["2023-01-01/2023-01-01"]
    time_coords = [np.datetime64("2023-01-01", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red", "green", "blue"],
        width=4,
        height=4,
    )

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = _fake_items("red", 1)
        plan = da.lazycogs.explain()

    # 3 bands * 1 time * 1 spatial tile = 3 chunk reads
    assert plan.total_chunk_reads == 3
    assert set(plan.bands) == {"red", "green", "blue"}


def test_accessor_explain_band_slice(wgs84):
    """explain() on a single-band slice only queries that band's backend."""
    dates = ["2023-01-01/2023-01-01"]
    time_coords = [np.datetime64("2023-01-01", "D")]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red", "green"],
        width=4,
        height=4,
    )
    da_red = da.sel(band="red")

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = []
        plan = da_red.lazycogs.explain()

    assert plan.bands == ["red"]
    assert plan.total_chunk_reads == 1  # only 1 band


def test_accessor_explain_time_slice(wgs84):
    """explain() on a time-sliced DataArray only queries matching time steps."""
    dates = ["2023-01-01/2023-01-01", "2023-01-02/2023-01-02", "2023-01-03/2023-01-03"]
    time_coords = [
        np.datetime64("2023-01-01", "D"),
        np.datetime64("2023-01-02", "D"),
        np.datetime64("2023-01-03", "D"),
    ]
    da = _make_da_with_backends(
        wgs84,
        dates=dates,
        time_coords=time_coords,
        bands=["red"],
        width=4,
        height=4,
    )
    da_sliced = da.isel(time=slice(0, 2))  # first 2 time steps

    with patch("rustac.DuckdbClient.search") as mock_search:
        mock_search.return_value = []
        plan = da_sliced.lazycogs.explain()

    assert len(plan.time_coords) == 2
    assert plan.total_chunk_reads == 2  # 1 band * 2 time steps * 1 spatial tile
