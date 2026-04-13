import asyncio
import logging
from pathlib import Path

import rustac
from pyproj import Transformer

import stac_cog_xarray

logging.basicConfig(level="WARN")
logging.getLogger("stac_cog_xarray").setLevel("DEBUG")


async def run():
    # define the AOI in/ a projection that is suitable for your analysis
    dst_crs = "epsg:5070"
    dst_bbox = (-150_000, 2_500_000, 600_000, 3_000_000)

    # transform to epsg:4326 for STAC search
    transformer = Transformer.from_crs(dst_crs, "epsg:4326", always_xy=True)
    bbox_4326 = transformer.transform_bounds(*dst_bbox)

    items_parquet = "/tmp/items.parquet"

    if not Path(items_parquet).exists():
        await rustac.search_to(
            items_parquet,
            href="https://earth-search.aws.element84.com/v1",
            collections=["sentinel-2-c1-l2a"],
            datetime="2025-06-01/2025-06-30",
            bbox=list(bbox_4326),
            limit=100,
        )

    da = await stac_cog_xarray.open_async(
        items_parquet,
        crs=dst_crs,
        bbox=dst_bbox,
        resolution=100,
        time_period="P1D",
        bands=["red", "green", "blue"],
        dtype="int16",
    )

    # test reading values from a point
    _ = da.chunk(time=1).sel(x=299965, y=2653947, method="nearest").compute()

    # test loading a larger array
    subset = da.sel(
        x=slice(100_000, 400_000),
        y=slice(2_600_000, 2_800_000),
    )

    _ = subset.isel(time=1).load()


if __name__ == "__main__":
    asyncio.run(run())
