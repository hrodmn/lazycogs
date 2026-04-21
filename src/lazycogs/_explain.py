"""Dry-run read estimator: explains which COG chunks would be read."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import numpy as np
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer

from lazycogs._backend import StacBackendArray, _run_coroutine
from lazycogs._chunk_reader import _open_and_window

if TYPE_CHECKING:
    from obstore.store import ObjectStore

logger = logging.getLogger(__name__)


@dataclass
class CogRead:
    """Read details for one COG file within one chunk.

    Attributes:
        item_id: STAC item ID.
        asset_key: Asset key (band name) that would be read.
        href: Asset HREF.
        overview_level: Overview level that would be read.  ``None`` means
            full resolution.  Only populated when ``fetch_headers=True``.
        overview_resolution: Pixel size of the selected level in source CRS
            units.  Only populated when ``fetch_headers=True``.
        window_col_off: Column offset of the read window in source pixels.
            Only populated when ``fetch_headers=True``.
        window_row_off: Row offset of the read window in source pixels.
            Only populated when ``fetch_headers=True``.
        window_width: Width of the read window in source pixels.
            Only populated when ``fetch_headers=True``.
        window_height: Height of the read window in source pixels.
            Only populated when ``fetch_headers=True``.

    """

    item_id: str
    asset_key: str
    href: str
    overview_level: int | None = None
    overview_resolution: float | None = None
    window_col_off: int | None = None
    window_row_off: int | None = None
    window_width: int | None = None
    window_height: int | None = None


@dataclass
class ChunkRead:
    """All reads required for one (band, time step, spatial tile).

    Attributes:
        band: Asset key for this chunk.
        time_index: Index of this time step in the full time axis.
        date_filter: ``rustac``-compatible datetime filter string for this
            time step.
        time_coord: Coordinate value for this time step.
        chunk_row: Tile row index within the spatial grid (0-indexed).
        chunk_col: Tile column index within the spatial grid (0-indexed).
        chunk_affine: Affine transform of the tile (top-left origin).
        chunk_width: Tile width in pixels.
        chunk_height: Tile height in pixels.
        cog_reads: Per-COG read details.
        n_cog_reads: Number of COG files matched (derived from ``cog_reads``).

    """

    band: str
    time_index: int
    date_filter: str
    time_coord: np.datetime64
    chunk_row: int
    chunk_col: int
    chunk_affine: Affine
    chunk_width: int
    chunk_height: int
    cog_reads: list[CogRead]
    n_cog_reads: int = field(init=False)

    def __post_init__(self) -> None:
        """Derive n_cog_reads from the cog_reads list."""
        self.n_cog_reads = len(self.cog_reads)


@dataclass
class ExplainPlan:
    """Complete dry-run read plan for a lazycogs query.

    Attributes:
        href: Path to the source geoparquet file.
        crs: String representation of the output CRS.
        resolution: Output pixel size in CRS units.
        bands: Ordered list of band names included in the plan.
        time_coords: Time coordinate values for all explained time steps.
        dst_width: Output grid width in pixels (for the current DataArray extent).
        dst_height: Output grid height in pixels (for the current DataArray extent).
        chunk_width: Spatial chunk width in pixels.
        chunk_height: Spatial chunk height in pixels.
        chunk_reads: One entry per (band, time step, spatial tile).
        fetch_headers: Whether COG headers were opened to populate overview
            and window fields on each :class:`ItemRead`.

    """

    href: str
    crs: str
    resolution: float
    bands: list[str]
    time_coords: list[np.datetime64]
    dst_width: int
    dst_height: int
    chunk_width: int
    chunk_height: int
    chunk_reads: list[ChunkRead]
    fetch_headers: bool

    @property
    def total_chunk_reads(self) -> int:
        """Total number of (band, time, spatial) chunk reads."""
        return len(self.chunk_reads)

    @property
    def total_cog_reads(self) -> int:
        """Total number of COG file reads across all chunks."""
        return sum(c.n_cog_reads for c in self.chunk_reads)

    @property
    def empty_chunk_count(self) -> int:
        """Number of chunks with zero matching COG files."""
        return sum(1 for c in self.chunk_reads if c.n_cog_reads == 0)

    @property
    def _n_tiles(self) -> tuple[int, int]:
        """Number of spatial tiles along (x, y) under the current chunking."""
        return (
            max(1, -(-self.dst_width // self.chunk_width)),
            max(1, -(-self.dst_height // self.chunk_height)),
        )

    def __repr__(self) -> str:
        """Return a compact single-line summary."""
        n_x, n_y = self._n_tiles
        return (
            f"ExplainPlan: {len(self.bands)} band(s) x {len(self.time_coords)} "
            f"time step(s) x {n_x * n_y} spatial chunk(s) "
            f"= {self.total_chunk_reads} chunk read(s)\n"
            f"  Grid: {self.dst_width}x{self.dst_height} px | "
            f"COG reads: {self.total_cog_reads} "
            f"(fetch_headers={self.fetch_headers})"
        )

    def _time_range(self) -> str:
        """Return a human-readable time range or ``"none"``."""
        if not self.time_coords:
            return "none"
        t0 = str(self.time_coords[0])[:10]
        t1 = str(self.time_coords[-1])[:10]
        return f"{t0} - {t1}" if t0 != t1 else t0

    def _header_lines(self) -> list[str]:
        """Return the top section of :meth:`summary` (grid + chunking)."""
        n_x, n_y = self._n_tiles
        return [
            "=== ExplainPlan ===",
            f"Parquet:    {self.href}",
            f"CRS:        {self.crs}  |  Resolution: {self.resolution} units/px  |  Grid: {self.dst_width} x {self.dst_height} px",
            f"Bands ({len(self.bands)}):  {', '.join(self.bands)}",
            f"Time steps: {len(self.time_coords)} ({self._time_range()})",
            f"Chunks:     {self.chunk_width} x {self.chunk_height} px -> {n_x}x{n_y} spatial tiles",
        ]

    def _distribution_lines(self) -> list[str]:
        """Return the chunk-COG distribution section of :meth:`summary`."""
        n_x, n_y = self._n_tiles
        counts = Counter(c.n_cog_reads for c in self.chunk_reads)
        total = len(self.chunk_reads) or 1
        zero = counts.get(0, 0)
        one = counts.get(1, 0)
        two_plus = sum(v for k, v in counts.items() if k >= 2)

        def pct(n: int) -> str:
            return f"({100 * n / total:.1f}%)"

        return [
            f"Total chunk reads:     {self.total_chunk_reads} "
            f"({len(self.bands)} band(s) x {len(self.time_coords)} time step(s) x {n_x * n_y} spatial tile(s))",
            f"Total COG reads:       {self.total_cog_reads}",
            f"Chunks with 0 COGs:    {zero:>4} {pct(zero)}",
            f"Chunks with 1 COG:     {one:>4} {pct(one)}",
            f"Chunks with 2+ COGs:   {two_plus:>4} {pct(two_plus)}",
            f"Max COGs per chunk:    {max((c.n_cog_reads for c in self.chunk_reads), default=0)}",
        ]

    def _header_detail_lines(self) -> list[str]:
        """Return overview/window stats when ``fetch_headers`` is true."""
        if not self.fetch_headers:
            return [
                "(Pass fetch_headers=True to see overview levels and pixel windows.)"
            ]
        all_reads = [r for c in self.chunk_reads for r in c.cog_reads]
        if not all_reads:
            return ["Overview levels: n/a (no matched COG reads)"]
        ov_counts: Counter[str] = Counter()
        for r in all_reads:
            ov_counts[
                "full" if r.overview_level is None else f"ovr {r.overview_level}"
            ] += 1
        widths = [r.window_width for r in all_reads if r.window_width is not None]
        heights = [r.window_height for r in all_reads if r.window_height is not None]
        avg_w = sum(widths) / len(widths) if widths else 0
        avg_h = sum(heights) / len(heights) if heights else 0
        return [
            f"Overview levels:       {'  '.join(f'{k}: {v}' for k, v in sorted(ov_counts.items()))}",
            f"Avg read window:       {avg_w:.0f} x {avg_h:.0f} px",
            "(Use .to_dataframe() for per-item overview and window details.)",
        ]

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the explain plan."""
        return "\n".join(
            [
                *self._header_lines(),
                "",
                *self._distribution_lines(),
                *self._header_detail_lines(),
            ]
        )

    def to_dataframe(self):
        """Return a DataFrame with one row per (chunk x item) combination.

        Empty chunks contribute one row with item fields set to ``None``.
        When ``fetch_headers=False``, the overview and window columns are
        all ``None``.

        Returns:
            A ``pandas.DataFrame`` with columns for chunk metadata, item
            metadata, and (when available) COG header details.

        Raises:
            ImportError: If ``pandas`` is not installed.

        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(); install it with: uv add pandas"
            ) from exc

        rows = []
        for chunk in self.chunk_reads:
            base = {
                "band": chunk.band,
                "time_index": chunk.time_index,
                "date_filter": chunk.date_filter,
                "time_coord": chunk.time_coord,
                "chunk_row": chunk.chunk_row,
                "chunk_col": chunk.chunk_col,
                "chunk_width": chunk.chunk_width,
                "chunk_height": chunk.chunk_height,
                "n_cog_reads": chunk.n_cog_reads,
            }
            if chunk.cog_reads:
                for item in chunk.cog_reads:
                    rows.append(
                        {
                            **base,
                            "item_id": item.item_id,
                            "asset_key": item.asset_key,
                            "href": item.href,
                            "overview_level": item.overview_level,
                            "overview_resolution": item.overview_resolution,
                            "window_col_off": item.window_col_off,
                            "window_row_off": item.window_row_off,
                            "window_width": item.window_width,
                            "window_height": item.window_height,
                        }
                    )
            else:
                rows.append(
                    {
                        **base,
                        "item_id": None,
                        "asset_key": None,
                        "href": None,
                        "overview_level": None,
                        "overview_resolution": None,
                        "window_col_off": None,
                        "window_row_off": None,
                        "window_width": None,
                        "window_height": None,
                    }
                )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_chunk_bbox_4326(
    chunk_affine: Affine,
    chunk_width: int,
    chunk_height: int,
    dst_crs: CRS,
) -> list[float]:
    """Return the bounding box of a chunk in EPSG:4326.

    Args:
        chunk_affine: Affine transform of the chunk (top-left origin).
        chunk_width: Chunk width in pixels.
        chunk_height: Chunk height in pixels.
        dst_crs: CRS of the chunk.

    Returns:
        ``[minx, miny, maxx, maxy]`` in EPSG:4326.

    """
    minx = chunk_affine.c
    maxy = chunk_affine.f
    maxx = minx + chunk_width * chunk_affine.a
    miny = maxy + chunk_height * chunk_affine.e  # e < 0

    epsg_4326 = CRS.from_epsg(4326)
    if dst_crs.equals(epsg_4326):
        return [minx, miny, maxx, maxy]

    transformer = Transformer.from_crs(dst_crs, epsg_4326, always_xy=True)
    xs, ys = transformer.transform(
        [minx, maxx, minx, maxx],
        [maxy, maxy, miny, miny],
    )
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _iter_spatial_chunks(
    roi_affine: Affine,
    roi_width: int,
    roi_height: int,
    chunk_w: int,
    chunk_h: int,
) -> Iterator[tuple[int, int, Affine, int, int]]:
    """Yield spatial tile descriptors for a region of interest.

    Args:
        roi_affine: Affine transform of the ROI top-left corner.
        roi_width: ROI width in pixels.
        roi_height: ROI height in pixels.
        chunk_w: Tile width in pixels (edge tiles may be smaller).
        chunk_h: Tile height in pixels (edge tiles may be smaller).

    Yields:
        ``(chunk_row, chunk_col, tile_affine, actual_width, actual_height)``
        tuples, one per tile.

    """
    y_off = 0
    row = 0
    while y_off < roi_height:
        actual_h = min(chunk_h, roi_height - y_off)
        x_off = 0
        col = 0
        while x_off < roi_width:
            actual_w = min(chunk_w, roi_width - x_off)
            tile_affine = roi_affine * Affine.translation(x_off, y_off)
            yield row, col, tile_affine, actual_w, actual_h
            x_off += chunk_w
            col += 1
        y_off += chunk_h
        row += 1


def _infer_chunk_sizes(da: xr.DataArray) -> tuple[int, int]:
    """Return ``(chunk_height, chunk_width)`` from dask chunks or full extent.

    Args:
        da: DataArray to inspect.

    Returns:
        Tile dimensions in pixels.  When the array is not dask-backed, the
        full spatial extent is returned as a single tile.

    """
    chunksizes = da.chunksizes
    chunk_h = int(chunksizes["y"][0]) if "y" in chunksizes else da.sizes["y"]
    chunk_w = int(chunksizes["x"][0]) if "x" in chunksizes else da.sizes["x"]
    return chunk_h, chunk_w


def _roi_pixel_offsets(
    da: xr.DataArray, backend: StacBackendArray
) -> tuple[int, int, int, int]:
    """Map the DataArray's coordinate extent to pixel offsets in the full grid.

    Args:
        da: DataArray whose spatial extent to map.  Must have ``y`` and ``x``
            dimensions.
        backend: Backend whose ``dst_affine`` defines the full grid.

    Returns:
        ``(x_start, y_start_physical, roi_width, roi_height)`` where
        ``x_start`` is the column offset from the left edge of the full grid,
        ``y_start_physical`` is the row offset from the top (physical, top-down),
        and ``roi_width`` / ``roi_height`` are the dimensions in pixels.

    """
    resolution = backend.dst_affine.a
    affine = backend.dst_affine

    # Pixel center of column `col`: affine.c + (col + 0.5) * resolution
    # Invert: col = (x_center - affine.c) / resolution - 0.5
    x_start = int(round((float(da.x.values[0]) - affine.c) / resolution - 0.5))

    # Physical row center (top-down): affine.f - (row + 0.5) * resolution
    # Invert: row = (affine.f - y_center) / resolution - 0.5
    # da.y is ascending, so da.y.values[-1] is the northernmost (topmost) value
    y_start_physical = int(
        round((affine.f - float(da.y.values[-1])) / resolution - 0.5)
    )

    return (
        max(0, x_start),
        max(0, y_start_physical),
        da.sizes["x"],
        da.sizes["y"],
    )


async def _inspect_item_async(
    item: dict,
    band: str,
    chunk_affine: Affine,
    dst_crs: CRS,
    chunk_width: int,
    chunk_height: int,
    store: ObjectStore | None = None,
) -> CogRead | None:
    """Open a COG header and compute the overview level and read window.

    Does not read any pixel data.

    Args:
        item: STAC item dict.
        band: Asset key to inspect.
        chunk_affine: Affine transform of the destination chunk.
        dst_crs: CRS of the destination chunk.
        chunk_width: Chunk width in pixels.
        chunk_height: Chunk height in pixels.
        store: Optional pre-configured obstore ``ObjectStore``.

    Returns:
        A :class:`CogRead` with all header fields populated, or ``None`` if
        the item has no matching asset or the chunk does not overlap.

    """
    opened = await _open_and_window(
        item, band, chunk_affine, dst_crs, chunk_width, chunk_height, store=store
    )
    if opened is None:
        return None
    geotiff, reader, window, _ = opened

    overview_level = geotiff.overviews.index(reader) if reader is not geotiff else None
    return CogRead(
        item_id=item.get("id", ""),
        asset_key=band,
        href=item["assets"][band]["href"],
        overview_level=overview_level,
        overview_resolution=abs(reader.transform.a),
        window_col_off=window.col_off if window is not None else None,
        window_row_off=window.row_off if window is not None else None,
        window_width=window.width if window is not None else None,
        window_height=window.height if window is not None else None,
    )


async def _explain_async(
    da: xr.DataArray,
    backends: list[StacBackendArray],
    fetch_headers: bool,
) -> ExplainPlan:
    """Run DuckDB queries for all (band, time, spatial chunk) combinations.

    Args:
        da: DataArray whose extent and chunking define the explain scope.
        backends: Full list of :class:`StacBackendArray` instances from attrs.
        fetch_headers: When ``True``, open each matched COG header.

    Returns:
        An :class:`ExplainPlan` with one :class:`ChunkRead` per combination.

    """
    if "y" not in da.sizes or "x" not in da.sizes:
        raise ValueError(
            "DataArray must have 'y' and 'x' dimensions for explain(). "
            "The array may have been reduced to a single pixel."
        )

    # Filter to bands present in the current DataArray.
    if "band" in da.coords:
        current_bands: set[str] = set(
            np.atleast_1d(da.coords["band"].values).astype(str)
        )
        active_backends = [b for b in backends if b.band in current_bands]
    else:
        active_backends = backends

    if not active_backends:
        raise ValueError("No matching bands found in the stored backends.")

    backend = active_backends[0]
    dst_crs = backend.dst_crs

    # Identify which time steps to explain based on current DataArray coords.
    full_time_coords: np.ndarray = np.asarray(
        da.attrs["_stac_time_coords"], dtype="datetime64[D]"
    )
    full_time_filters: list[str] = backend.dates  # same list as filter_strings

    if "time" in da.coords:
        current_times: set[np.datetime64] = set(
            np.atleast_1d(da.coords["time"].values).astype("datetime64[D]")
        )
        time_items = [
            (i, f, tc)
            for i, (f, tc) in enumerate(zip(full_time_filters, full_time_coords))
            if tc.astype("datetime64[D]") in current_times
        ]
    else:
        time_items = [
            (i, f, full_time_coords[i]) for i, f in enumerate(full_time_filters)
        ]

    chunk_h, chunk_w = _infer_chunk_sizes(da)
    chunk_reads: list[ChunkRead] = []

    for band_backend in active_backends:
        x_start, y_start_physical, roi_width, roi_height = _roi_pixel_offsets(
            da, band_backend
        )
        roi_affine = band_backend.dst_affine * Affine.translation(
            x_start, y_start_physical
        )

        for t_idx, date_filter, time_coord in time_items:
            for row, col, tile_affine, actual_w, actual_h in _iter_spatial_chunks(
                roi_affine, roi_width, roi_height, chunk_w, chunk_h
            ):
                chunk_bbox_4326 = _compute_chunk_bbox_4326(
                    tile_affine, actual_w, actual_h, dst_crs
                )
                items = band_backend.duckdb_client.search(
                    band_backend.parquet_path,
                    bbox=chunk_bbox_4326,
                    datetime=date_filter,
                    sortby=band_backend.sort_by,
                    filter=band_backend.filter,
                    ids=band_backend.ids,
                )
                logger.debug(
                    "explain band=%r date=%s chunk=(%d,%d) -> %d items",
                    band_backend.band,
                    date_filter,
                    row,
                    col,
                    len(items),
                )

                if fetch_headers and items:
                    raw_reads = await asyncio.gather(
                        *[
                            _inspect_item_async(
                                item,
                                band_backend.band,
                                tile_affine,
                                dst_crs,
                                actual_w,
                                actual_h,
                                band_backend.store,
                            )
                            for item in items
                        ]
                    )
                    cog_reads = [r for r in raw_reads if r is not None]
                else:
                    cog_reads = [
                        CogRead(
                            item_id=item.get("id", ""),
                            asset_key=band_backend.band,
                            href=item.get("assets", {})
                            .get(band_backend.band, {})
                            .get("href", ""),
                        )
                        for item in items
                    ]

                chunk_reads.append(
                    ChunkRead(
                        band=band_backend.band,
                        time_index=t_idx,
                        date_filter=date_filter,
                        time_coord=time_coord,
                        chunk_row=row,
                        chunk_col=col,
                        chunk_affine=tile_affine,
                        chunk_width=actual_w,
                        chunk_height=actual_h,
                        cog_reads=cog_reads,
                    )
                )

    return ExplainPlan(
        href=backend.parquet_path,
        crs=str(dst_crs),
        resolution=backend.dst_affine.a,
        bands=[b.band for b in active_backends],
        time_coords=[tc for _, _, tc in time_items],
        dst_width=da.sizes["x"],
        dst_height=da.sizes["y"],
        chunk_width=chunk_w,
        chunk_height=chunk_h,
        chunk_reads=chunk_reads,
        fetch_headers=fetch_headers,
    )


# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------


@xr.register_dataarray_accessor("lazycogs")
class StacCogAccessor:
    """xarray accessor adding explain functionality to lazycogs DataArrays.

    Registered as the ``stac_cog`` namespace on all ``xr.DataArray`` objects.
    The :meth:`explain` method is only useful on DataArrays produced by
    :func:`lazycogs.open` or :func:`lazycogs.open_async`.

    """

    def __init__(self, da: xr.DataArray) -> None:
        """Initialise the accessor.

        Args:
            da: The DataArray this accessor is attached to.

        """
        self._da = da

    def explain(self, fetch_headers: bool = False) -> ExplainPlan:
        """Return a dry-run read plan without fetching any pixel data.

        Runs the same DuckDB spatial queries that would fire during
        ``.compute()``, but stops before any COG pixel I/O.  With
        ``fetch_headers=True`` the COG IFD headers are also fetched (one
        small HTTP range request per matched item) to determine which overview
        level and pixel window would be read.

        Args:
            fetch_headers: When ``True``, open each matched COG header to
                populate :attr:`ItemRead.overview_level` and the window
                fields.  Requires network I/O.  Defaults to ``False``.

        Returns:
            An :class:`ExplainPlan` describing all (band, time step, spatial
            tile) reads for the current DataArray extent and chunking.

        Raises:
            ValueError: If the DataArray was not produced by
                ``lazycogs.open()`` (missing explain metadata in
                ``attrs``).

        """
        backends: list[StacBackendArray] | None = self._da.attrs.get("_stac_backends")
        if backends is None:
            raise ValueError(
                "This DataArray does not have stac_cog explain metadata. "
                "Ensure it was created by lazycogs.open() or "
                "lazycogs.open_async()."
            )
        return _run_coroutine(_explain_async(self._da, backends, fetch_headers))
