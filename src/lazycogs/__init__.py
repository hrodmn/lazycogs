"""lazycogs: lazy xarray DataArrays from STAC COG collections."""

from lazycogs._core import open, open_async
from lazycogs._executor import set_reproject_workers
from lazycogs._explain import (  # noqa: F401 — registers da.stac_cog accessor
    ChunkRead,
    ExplainPlan,
    CogRead,
    StacCogAccessor,
)
from lazycogs._mosaic_methods import (
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
    "set_reproject_workers",
    "ExplainPlan",
    "ChunkRead",
    "CogRead",
    "MosaicMethodBase",
    "FirstMethod",
    "HighestMethod",
    "LowestMethod",
    "MeanMethod",
    "MedianMethod",
    "StdevMethod",
    "CountMethod",
]
