"""Compute output raster grid parameters from bbox, CRS, and resolution."""

from __future__ import annotations

import numpy as np
from affine import Affine
from pyproj import CRS


def compute_output_grid(
    bbox: tuple[float, float, float, float],
    crs: CRS,
    resolution: float,
) -> tuple[Affine, int, int, np.ndarray, np.ndarray]:
    """Compute the output raster grid from a bounding box and resolution.

    The grid is aligned to the bbox corners, with x increasing left-to-right
    and y increasing bottom-to-top (ascending), so that label-based slicing
    with ``xarray.sel`` works naturally with ``(low, high)`` ranges.

    Args:
        bbox: ``(minx, miny, maxx, maxy)`` in the target CRS.
        crs: Target coordinate reference system (used only for documentation;
            callers are responsible for ensuring bbox is in this CRS).
        resolution: Pixel size in CRS units (assumed square).

    Returns:
        A five-tuple ``(transform, width, height, x_coords, y_coords)`` where
        ``transform`` is the affine mapping from pixel space to CRS space,
        ``width`` and ``height`` are the grid dimensions, and ``x_coords`` /
        ``y_coords`` are 1-D arrays of pixel-centre coordinates.

    """
    minx, miny, maxx, maxy = bbox

    width = max(1, round((maxx - minx) / resolution))
    height = max(1, round((maxy - miny) / resolution))

    # Origin at the top-left corner of the top-left pixel.
    transform = Affine(resolution, 0.0, minx, 0.0, -resolution, maxy)

    x_coords = minx + resolution / 2 + np.arange(width) * resolution
    y_coords = miny + resolution / 2 + np.arange(height) * resolution

    return transform, width, height, x_coords, y_coords
