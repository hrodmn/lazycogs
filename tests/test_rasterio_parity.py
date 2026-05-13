"""Rasterio/GDAL reference tests for lazycogs reprojection behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.enums
import rasterio.warp
from affine import Affine
from async_geotiff import GeoTIFF
from pyproj import CRS

from lazycogs._chunk_reader import _native_window, _select_overview
from lazycogs._reproject import (
    ReprojectRequest,
    ResamplingMethod,
    _get_transformer,
    reproject_tile,
)
from lazycogs._store import resolve

_NEAREST_RESOLUTIONS = [
    10,
    14,
    19,
    20,
    21,
    30,
    39,
    40,
    41,
    60,
    79,
    80,
    81,
    120,
    159,
    160,
    161,
    300,
]
_INTERPOLATING_RESOLUTIONS = [14, 21, 41, 81, 161]
_CHUNK_SIZE = 64
_CENTER_UTM_X = 510_240.0
_CENTER_UTM_Y = 5_589_760.0
_INTERPOLATING_METHODS = [
    pytest.param(
        ResamplingMethod.BILINEAR,
        rasterio.enums.Resampling.bilinear,
        id="bilinear",
    ),
    pytest.param(
        ResamplingMethod.CUBIC,
        rasterio.enums.Resampling.cubic,
        id="cubic",
    ),
]


def _chunk_affine(resolution: float, center_x: float, center_y: float) -> Affine:
    half = (_CHUNK_SIZE / 2) * resolution
    return Affine(resolution, 0.0, center_x - half, 0.0, -resolution, center_y + half)


async def _read_lazycogs(
    href: str,
    chunk_affine: Affine,
    dst_crs: CRS,
    *,
    resampling: ResamplingMethod = ResamplingMethod.NEAREST,
    backend: str | None = None,
) -> np.ndarray:
    """Run the lazycogs tile read + reprojection path for one chunk."""
    store, path = resolve(href)
    geotiff = await GeoTIFF.open(path, store=store)
    src_crs = geotiff.crs
    same_crs = dst_crs.equals(src_crs)

    target_res_native = abs(chunk_affine.a)
    t = None
    if not same_crs:
        t = _get_transformer(dst_crs, src_crs)
        cx = chunk_affine.c + (_CHUNK_SIZE / 2) * chunk_affine.a
        cy = chunk_affine.f + (_CHUNK_SIZE / 2) * chunk_affine.e
        x0, _ = t.transform(cx, cy)
        x1, _ = t.transform(cx + chunk_affine.a, cy)
        target_res_native = abs(x1 - x0)

    overview = _select_overview(geotiff, target_res_native)
    reader = overview if overview is not None else geotiff

    chunk_minx = chunk_affine.c
    chunk_maxy = chunk_affine.f
    chunk_maxx = chunk_minx + _CHUNK_SIZE * chunk_affine.a
    chunk_miny = chunk_maxy + _CHUNK_SIZE * chunk_affine.e

    if same_crs or t is None:
        bbox_native = (chunk_minx, chunk_miny, chunk_maxx, chunk_maxy)
    else:
        xs, ys = t.transform(
            [chunk_minx, chunk_maxx, chunk_minx, chunk_maxx],
            [chunk_maxy, chunk_maxy, chunk_miny, chunk_miny],
        )
        bbox_native = (min(xs), min(ys), max(xs), max(ys))

    window = _native_window(reader, bbox_native, reader.width, reader.height)
    assert window is not None, "test chunk must overlap COG extent"

    raster = await reader.read(window=window)
    return reproject_tile(
        ReprojectRequest(
            data=raster.data,
            src_transform=raster.transform,
            src_crs=src_crs,
            dst_transform=chunk_affine,
            dst_crs=dst_crs,
            dst_width=_CHUNK_SIZE,
            dst_height=_CHUNK_SIZE,
            nodata=geotiff.nodata,
            resampling=resampling,
        ),
        backend=backend,
    )


def _odc_overview_level(
    path: Path,
    target_res_native: float,
    native_res: float,
) -> int | None:
    """Replicate odc-stac's pick_overview: coarsest shrink <= read_shrink."""
    read_shrink = int(target_res_native / native_res)
    with rasterio.open(path) as src:
        overviews = src.overviews(1)
    ovr_level: int | None = None
    for i, shrink in enumerate(overviews):
        if shrink <= read_shrink:
            ovr_level = i
        else:
            break
    return ovr_level


def _read_rasterio(
    path: Path,
    chunk_affine: Affine,
    dst_crs: CRS,
    native_res: float,
    *,
    resampling: rasterio.enums.Resampling,
) -> np.ndarray:
    """Run rasterio/GDAL reproject at the odc-stac-selected overview."""
    with rasterio.open(path) as src:
        src_crs_obj = CRS.from_user_input(src.crs.to_wkt())
        src_dtype = np.dtype(src.dtypes[0])
        src_nodata = src.nodata
    same_crs = dst_crs.equals(src_crs_obj)

    target_res_native = abs(chunk_affine.a)
    if not same_crs:
        t = _get_transformer(dst_crs, src_crs_obj)
        cx = chunk_affine.c + (_CHUNK_SIZE / 2) * chunk_affine.a
        cy = chunk_affine.f + (_CHUNK_SIZE / 2) * chunk_affine.e
        x0, _ = t.transform(cx, cy)
        x1, _ = t.transform(cx + chunk_affine.a, cy)
        target_res_native = abs(x1 - x0)

    ovr_level = _odc_overview_level(path, target_res_native, native_res)

    with rasterio.open(path, overview_level=ovr_level) as src:
        out = np.zeros((1, _CHUNK_SIZE, _CHUNK_SIZE), dtype=src_dtype)
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=chunk_affine,
            dst_crs=dst_crs.to_wkt(),
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
        )
    return out


def _href(path: Path) -> str:
    return path.as_uri()


def _assert_parity(
    lazycogs_out: np.ndarray,
    rasterio_out: np.ndarray,
    label: str,
    *,
    max_differing_pixels: int = 0,
    max_abs_diff: int = 0,
) -> None:
    """Assert parity with integer-valued outputs under explicit tolerances."""
    diff = lazycogs_out.astype(np.int32) - rasterio_out.astype(np.int32)
    n_diff = int(np.count_nonzero(diff))
    actual_max = int(np.abs(diff).max()) if n_diff else 0
    msg = (
        f"{label}: {n_diff}/{lazycogs_out.size} pixels differ "
        f"(allowed ≤{max_differing_pixels}); max abs diff = {actual_max} "
        f"(allowed ≤{max_abs_diff})"
    )
    assert n_diff <= max_differing_pixels, msg
    assert actual_max <= max_abs_diff, msg


def _assert_interpolating_reference(
    lazycogs_out: np.ndarray,
    rasterio_out: np.ndarray,
    label: str,
    *,
    nodata: float,
    atol: float,
    max_nodata_mismatch: int = 0,
    rtol: float = 1e-6,
) -> None:
    """Assert float-valued interpolation output stays close to rasterio."""
    lazy_valid = ~np.isclose(lazycogs_out, nodata)
    rasterio_valid = ~np.isclose(rasterio_out, nodata)
    nodata_mismatch = int(np.count_nonzero(lazy_valid != rasterio_valid))
    assert nodata_mismatch <= max_nodata_mismatch, (
        f"{label} had {nodata_mismatch} nodata-mask mismatches "
        f"(allowed ≤{max_nodata_mismatch})"
    )

    shared_valid = lazy_valid & rasterio_valid
    assert np.any(shared_valid), f"{label} produced no overlapping valid pixels"

    try:
        np.testing.assert_allclose(
            lazycogs_out[shared_valid],
            rasterio_out[shared_valid],
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{label} exceeded tolerance atol={atol}, rtol={rtol}",
        ) from exc


@pytest.mark.parametrize("resolution", _NEAREST_RESOLUTIONS)
def test_parity_same_crs(synthetic_cog: Path, resolution: int) -> None:
    """rust-warp nearest matches rasterio for same-CRS reads at overview boundaries."""
    dst_crs = CRS.from_epsg(32632)
    affine = _chunk_affine(resolution, _CENTER_UTM_X, _CENTER_UTM_Y)

    lc_out = asyncio.run(_read_lazycogs(_href(synthetic_cog), affine, dst_crs))
    rio_out = _read_rasterio(
        synthetic_cog,
        affine,
        dst_crs,
        native_res=10.0,
        resampling=rasterio.enums.Resampling.nearest,
    )

    _assert_parity(lc_out, rio_out, f"same_crs res={resolution}")


@pytest.mark.parametrize("resolution", _NEAREST_RESOLUTIONS)
def test_parity_cross_crs(synthetic_cog: Path, resolution: int) -> None:
    """rust-warp nearest stays within tight rasterio tolerances cross-CRS."""
    src_crs = CRS.from_epsg(32632)
    dst_crs = CRS.from_epsg(3035)
    t = _get_transformer(src_crs, dst_crs)
    cx_laea, cy_laea = t.transform(_CENTER_UTM_X, _CENTER_UTM_Y)
    affine = _chunk_affine(float(resolution), cx_laea, cy_laea)

    lc_out = asyncio.run(_read_lazycogs(_href(synthetic_cog), affine, dst_crs))
    rio_out = _read_rasterio(
        synthetic_cog,
        affine,
        dst_crs,
        native_res=10.0,
        resampling=rasterio.enums.Resampling.nearest,
    )

    _assert_parity(
        lc_out,
        rio_out,
        f"cross_crs res={resolution}m",
        max_differing_pixels=3,
        max_abs_diff=2048 * 16 + 1,
    )


@pytest.mark.parametrize("resolution", [20, 60, 160])
def test_nearest_legacy_matches_rust_warp_same_crs(
    synthetic_cog: Path,
    resolution: int,
) -> None:
    """Migration-window A/B checks keep nearest same-CRS behavior aligned."""
    dst_crs = CRS.from_epsg(32632)
    affine = _chunk_affine(float(resolution), _CENTER_UTM_X, _CENTER_UTM_Y)

    legacy_out = asyncio.run(
        _read_lazycogs(
            _href(synthetic_cog),
            affine,
            dst_crs,
            backend="legacy",
        ),
    )
    rust_out = asyncio.run(
        _read_lazycogs(
            _href(synthetic_cog),
            affine,
            dst_crs,
            backend="rust-warp",
        ),
    )

    np.testing.assert_array_equal(rust_out, legacy_out)


@pytest.mark.parametrize("resolution", [20, 60, 160])
def test_nearest_legacy_matches_rust_warp_cross_crs(
    synthetic_cog: Path,
    resolution: int,
) -> None:
    """Migration-window A/B checks keep nearest cross-CRS behavior aligned."""
    src_crs = CRS.from_epsg(32632)
    dst_crs = CRS.from_epsg(3035)
    t = _get_transformer(src_crs, dst_crs)
    cx_laea, cy_laea = t.transform(_CENTER_UTM_X, _CENTER_UTM_Y)
    affine = _chunk_affine(float(resolution), cx_laea, cy_laea)

    legacy_out = asyncio.run(
        _read_lazycogs(
            _href(synthetic_cog),
            affine,
            dst_crs,
            backend="legacy",
        ),
    )
    rust_out = asyncio.run(
        _read_lazycogs(
            _href(synthetic_cog),
            affine,
            dst_crs,
            backend="rust-warp",
        ),
    )

    _assert_parity(
        rust_out,
        legacy_out,
        f"legacy_vs_rust cross_crs res={resolution}m",
        max_differing_pixels=3,
        max_abs_diff=2048 * 16 + 1,
    )


@pytest.mark.parametrize(("resampling", "rasterio_resampling"), _INTERPOLATING_METHODS)
@pytest.mark.parametrize("resolution", _INTERPOLATING_RESOLUTIONS)
def test_interpolating_parity_same_crs(
    continuous_synthetic_cog: Path,
    resampling: ResamplingMethod,
    rasterio_resampling: rasterio.enums.Resampling,
    resolution: int,
) -> None:
    """Interpolating kernels stay close to rasterio on smooth same-CRS data."""
    dst_crs = CRS.from_epsg(32632)
    affine = _chunk_affine(float(resolution), _CENTER_UTM_X, _CENTER_UTM_Y)

    lc_out = asyncio.run(
        _read_lazycogs(
            _href(continuous_synthetic_cog),
            affine,
            dst_crs,
            resampling=resampling,
        ),
    )
    rio_out = _read_rasterio(
        continuous_synthetic_cog,
        affine,
        dst_crs,
        native_res=10.0,
        resampling=rasterio_resampling,
    )

    atol = 1e-3 if resampling is ResamplingMethod.BILINEAR else 2e-1
    _assert_interpolating_reference(
        lc_out,
        rio_out,
        f"same_crs {resampling} res={resolution}",
        nodata=-9999.0,
        atol=atol,
    )


@pytest.mark.parametrize(("resampling", "rasterio_resampling"), _INTERPOLATING_METHODS)
def test_interpolating_parity_cross_crs(
    continuous_synthetic_cog: Path,
    resampling: ResamplingMethod,
    rasterio_resampling: rasterio.enums.Resampling,
) -> None:
    """Interpolating kernels stay close to rasterio on smooth cross-CRS data."""
    src_crs = CRS.from_epsg(32632)
    dst_crs = CRS.from_epsg(3035)
    t = _get_transformer(src_crs, dst_crs)
    cx_laea, cy_laea = t.transform(_CENTER_UTM_X, _CENTER_UTM_Y)
    affine = _chunk_affine(60.0, cx_laea, cy_laea)

    lc_out = asyncio.run(
        _read_lazycogs(
            _href(continuous_synthetic_cog),
            affine,
            dst_crs,
            resampling=resampling,
        ),
    )
    rio_out = _read_rasterio(
        continuous_synthetic_cog,
        affine,
        dst_crs,
        native_res=10.0,
        resampling=rasterio_resampling,
    )

    atol = 1e-2 if resampling is ResamplingMethod.BILINEAR else 2e-1
    _assert_interpolating_reference(
        lc_out,
        rio_out,
        f"cross_crs {resampling}",
        nodata=-9999.0,
        atol=atol,
        max_nodata_mismatch=24,
    )
