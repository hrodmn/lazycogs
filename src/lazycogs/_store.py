"""Resolve cloud storage HREFs into store/path pairs and obstore stores."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from obstore.store import from_url
from rustac import DuckdbClient

from lazycogs._storage_ext import _extract_store_kwargs

if TYPE_CHECKING:
    from collections.abc import Callable

    from async_geotiff import Store
    from obstore.store import ObjectStore

logger = logging.getLogger(__name__)

_local = threading.local()


def _cache() -> dict[str, ObjectStore]:
    """Return the thread-local store cache, creating it on first access."""
    if not hasattr(_local, "stores"):
        _local.stores = {}
    return _local.stores


def resolve(
    href: str,
    store: Store | None = None,
    path_fn: Callable[[str], str] | None = None,
) -> tuple[Store, str]:
    """Resolve an HREF into a ``(store, path)`` pair.

    When ``store`` is supplied, it is returned unchanged and only the object
    path is extracted from the HREF. The caller is responsible for ensuring
    the store satisfies the :class:`async_geotiff.Store` read contract
    accepted by ``GeoTIFF.open`` and is rooted at the same ``scheme://netloc`` the HREF
    points to; no introspection is performed on the provided store.

    When ``store`` is ``None``, a store is auto-constructed via
    :func:`obstore.store.from_url` using only the ``scheme://netloc`` portion
    of the HREF and cached per thread. No credential defaults are applied; the
    store is constructed with obstore's own environment-based credential
    discovery. For public buckets, signed URLs, custom endpoints, or
    request-payer buckets, construct the store yourself and pass it via
    ``store`` — see the cloud storage guide for examples.

    Args:
        href: A storage URL supported by :func:`obstore.store.from_url`
            (``s3``, ``s3a``, ``gs``, Azure variants, ``http``, ``https``,
            ``file``, ``memory``).
        store: Optional pre-configured :class:`async_geotiff.Store`
            accepted by ``GeoTIFF.open``.
        path_fn: Optional callable that takes the full HREF and returns the
            object path to use with the store.  When provided, it replaces the
            default ``urlparse``-based path extraction.  Only meaningful when
            combined with a custom ``store`` — without one, the auto-resolved
            store is constructed from the HREF root, and the default path
            extraction is correct for standard cloud URLs.

    Returns:
        A ``(store, path)`` tuple where ``path`` is the object path within
        the store (no leading slash, except for ``file://`` which keeps the
        absolute path).

    """
    parsed = urlparse(href)
    scheme = parsed.scheme.lower()

    if path_fn is not None:
        path = path_fn(href)
    else:
        path = parsed.path if scheme == "file" else parsed.path.lstrip("/")

    if store is not None:
        return store, path

    root_url = f"{scheme}://{parsed.netloc}" if scheme != "file" else "file:///"
    cache = _cache()
    if root_url not in cache:
        cache[root_url] = from_url(root_url)
    return cache[root_url], path


def store_for(
    href: str,
    *,
    asset: str | None = None,
    duckdb_client: DuckdbClient | None = None,
    **kwargs: object,
) -> ObjectStore:
    """Construct an ``ObjectStore`` by inspecting a stac-geoparquet sample asset.

    Reads one sample item from *href*, derives the store root URL from a data
    asset HREF, and constructs an ``ObjectStore`` with obstore's own
    environment-based credential discovery.  If the item carries STAC Storage
    Extension metadata (v1.0.0 or v2.0.0), ``region`` and ``requester_pays``
    are also inferred automatically.

    Caller-supplied *kwargs* override all inferred values; pass
    ``skip_signature=True`` for public buckets that do not require signed
    requests, or supply explicit credentials.

    Args:
        href: Path to a geoparquet file or hive-partitioned parquet directory.
        asset: Asset key to inspect when choosing a representative asset.
            Defaults to the first data asset (role ``"data"`` or media type
            ``"image/tiff"``), falling back to the first asset in the item.
        duckdb_client: Optional ``DuckdbClient`` instance.  When
            ``None`` (default), a plain ``DuckdbClient()`` is used.
            Pass a custom client to query hive-partitioned datasets.
        **kwargs: Forwarded to :func:`obstore.store.from_url`, overriding
            any inferred values.

    Returns:
        A freshly constructed ``ObjectStore`` (not cached).

    Raises:
        ValueError: If no STAC items are found in *href*.
        KeyError: If *asset* is specified but not present in the item.

    """
    if duckdb_client is None:
        duckdb_client = DuckdbClient()
    items = duckdb_client.search(href, max_items=1)
    if not items:
        raise ValueError(f"No STAC items found in {href!r}")
    item = items[0]
    assets_map: dict[str, Any] = item.get("assets", {})

    if asset is not None:
        if asset not in assets_map:
            raise KeyError(f"Asset {asset!r} not found in item {item.get('id')!r}")
        asset_obj = assets_map[asset]
    else:
        data_keys = [
            k
            for k, v in assets_map.items()
            if "data" in v.get("roles", []) or "image/tiff" in v.get("type", "")
        ]
        key = data_keys[0] if data_keys else next(iter(assets_map))
        asset_obj = assets_map[key]

    asset_href: str = asset_obj.get("href", "")
    parsed = urlparse(asset_href)
    scheme = parsed.scheme.lower()
    root_url = f"{scheme}://{parsed.netloc}" if scheme != "file" else "file:///"

    try:
        inferred = _extract_store_kwargs(item, asset_obj)
    except (AttributeError, TypeError):
        logger.warning(
            "Failed to extract storage extension kwargs from %r;"
            " proceeding without them",
            href,
        )
        inferred = {}

    # Build the store once without extra kwargs so we can see which keys obstore
    # already derived from the URL (e.g. region from an HTTPS S3 hostname).
    # Only forward inferred keys that aren't already present, so we never hand
    # obstore a duplicate key (which raises an error).
    base_config = from_url(root_url).config
    filtered_inferred = {k: v for k, v in inferred.items() if k not in base_config}

    return from_url(root_url, **{**filtered_inferred, **kwargs})
