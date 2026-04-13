"""stac-cog-xarray: lazy xarray DataArrays from STAC COG collections."""

from stac_cog_xarray._core import open, open_async
from stac_cog_xarray._mosaic_methods import (
    CountMethod,
    FirstMethod,
    HighestMethod,
    LowestMethod,
    MeanMethod,
    MedianMethod,
    MosaicMethodBase,
    StdevMethod,
)

__all__ = [
    "open",
    "open_async",
    "MosaicMethodBase",
    "FirstMethod",
    "HighestMethod",
    "LowestMethod",
    "MeanMethod",
    "MedianMethod",
    "StdevMethod",
    "CountMethod",
]
