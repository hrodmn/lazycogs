"""Parse cloud storage HREFs into obstore Store instances."""

from __future__ import annotations

import threading
from typing import Any
from urllib.parse import urlparse

from obstore.store import LocalStore, from_url

_local = threading.local()


def _store_cache() -> dict[str, Any]:
    """Return the thread-local store cache, creating it if absent."""
    if not hasattr(_local, "stores"):
        _local.stores = {}
    return _local.stores


_OBJECT_STORE_HOSTS = (
    "amazonaws.com",  # AWS S3 (virtual-hosted and path-style)
    "googleapis.com",  # Google Cloud Storage
    "blob.core.windows.net",  # Azure Blob Storage
    "r2.cloudflarestorage.com",  # Cloudflare R2
)


def _is_object_store(scheme: str, netloc: str) -> bool:
    """Return True if the URL points to a cloud object store.

    Native object store schemes (s3, s3a, gs) always qualify. For https/http,
    check whether the host matches a known object store domain.

    Args:
        scheme: Lowercased URL scheme (e.g. ``"s3"``, ``"https"``).
        netloc: URL netloc component (host + optional port).

    Returns:
        ``True`` if the URL targets a cloud object store.

    """
    if scheme in ("s3", "s3a", "gs"):
        return True
    host = netloc.split(":")[0].lower()
    return any(
        host == domain or host.endswith(f".{domain}") for domain in _OBJECT_STORE_HOSTS
    )


def path_from_href(href: str) -> str:
    """Extract the object path from a cloud storage HREF.

    Returns the path component within the bucket/host (everything after
    ``scheme://netloc/``).  Use this when the store is supplied externally
    and only the in-store path is needed.

    Args:
        href: A cloud storage URL. Supported schemes: ``s3://``, ``s3a://``,
            ``gs://``, ``http://``, ``https://``, ``file://``.

    Returns:
        The object path within the store (no leading slash).

    Raises:
        ValueError: If the URL scheme is not supported.

    """
    parsed = urlparse(href)
    scheme = parsed.scheme.lower()
    if scheme == "file":
        return parsed.path
    if scheme not in ("s3", "s3a", "gs", "http", "https"):
        raise ValueError(
            f"Unsupported URL scheme {scheme!r} in {href!r}. "
            "Expected one of: s3://, s3a://, gs://, http://, https://, file://"
        )
    return parsed.path.lstrip("/")


def store_from_href(href: str) -> tuple[Any, str]:
    """Parse a cloud storage HREF into a (store, path) pair.

    The store is cached per thread per bucket/host to avoid repeated
    connection setup within a single dask task.

    Args:
        href: A cloud storage URL. Supported schemes: ``s3://``, ``s3a://``,
            ``gs://``, ``http://``, ``https://``.

    Returns:
        A ``(store, path)`` tuple where ``store`` is an obstore-compatible
        store instance and ``path`` is the object path within that store.

    Raises:
        ValueError: If the URL scheme is not supported.

    """
    parsed = urlparse(href)
    scheme = parsed.scheme.lower()

    if scheme == "file":
        cache = _store_cache()
        if "file://" not in cache:
            cache["file://"] = LocalStore()
        return cache["file://"], parsed.path

    if scheme in ("s3", "s3a", "gs", "http", "https"):
        root_url = f"{scheme}://{parsed.netloc}"
        path = parsed.path.lstrip("/")
    else:
        raise ValueError(
            f"Unsupported URL scheme {scheme!r} in {href!r}. "
            "Expected one of: s3://, s3a://, gs://, http://, https://, file://"
        )

    cache = _store_cache()
    if root_url not in cache:
        kwargs = (
            {"skip_signature": True} if _is_object_store(scheme, parsed.netloc) else {}
        )
        cache[root_url] = from_url(root_url, **kwargs)
    return cache[root_url], path
