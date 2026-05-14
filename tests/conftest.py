"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.enums
import rasterio.shutil
from affine import Affine
from pyproj import CRS


def _write_synthetic_cog(
    cog_path: Path,
    *,
    data: np.ndarray,
    transform: Affine,
    crs: CRS,
    nodata: float,
    overview_resampling: rasterio.enums.Resampling = rasterio.enums.Resampling.nearest,
) -> Path:
    """Write ``data`` to a tiled GeoTIFF with built overviews.

    The file is written with the two-step recipe required by ``async_geotiff``:
    first create a temporary GeoTIFF and build overviews, then copy it to a
    tiled output while preserving the overview IFDs.
    """
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs.to_wkt(),
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(data[np.newaxis])

        with rasterio.open(tmp_path, "r+") as dst:
            dst.build_overviews([2, 4, 8, 16], overview_resampling)
            dst.update_tags(
                ns="rio_overview",
                resampling=overview_resampling.name,
            )

        rasterio.shutil.copy(
            str(tmp_path),
            str(cog_path),
            driver="GTiff",
            copy_src_overviews=True,
            tiled=True,
            blockxsize=64,
            blockysize=64,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return cog_path


@pytest.fixture(scope="session")
def synthetic_cog(tmp_path_factory) -> Path:
    """Write a synthetic nearest-neighbor parity COG to a temp file."""
    cog_path = tmp_path_factory.mktemp("cog") / "synthetic.tif"
    native_res = 10.0
    size = 2048
    minx, maxy = 500_000.0, 5_600_000.0
    transform = Affine(native_res, 0.0, minx, 0.0, -native_res, maxy)
    crs = CRS.from_epsg(32632)

    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    data = ((cols + rows * size) % 65535 + 1).astype(np.uint16)

    return _write_synthetic_cog(
        cog_path,
        data=data,
        transform=transform,
        crs=crs,
        nodata=0,
        overview_resampling=rasterio.enums.Resampling.nearest,
    )


@pytest.fixture(scope="session")
def continuous_synthetic_cog(tmp_path_factory) -> Path:
    """Write a smooth float32 COG for interpolation parity tests."""
    cog_path = tmp_path_factory.mktemp("cog") / "continuous.tif"
    native_res = 10.0
    size = 1024
    minx, maxy = 500_000.0, 5_600_000.0
    transform = Affine(native_res, 0.0, minx, 0.0, -native_res, maxy)
    crs = CRS.from_epsg(32632)

    rows, cols = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        indexing="ij",
    )
    data = (
        cols * np.float32(0.5)
        + rows * np.float32(1.25)
        + np.sin(cols / np.float32(32.0)) * np.float32(5.0)
        + np.cos(rows / np.float32(40.0)) * np.float32(7.0)
        + np.float32(1000.0)
    ).astype(np.float32)

    return _write_synthetic_cog(
        cog_path,
        data=data,
        transform=transform,
        crs=crs,
        nodata=np.float32(-9999.0),
        overview_resampling=rasterio.enums.Resampling.average,
    )
