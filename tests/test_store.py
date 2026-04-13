"""Tests for _store.store_from_href."""

import pytest

from stac_cog_xarray._store import store_from_href


@pytest.mark.parametrize(
    "href, expected_path",
    [
        ("s3://my-bucket/path/to/file.tif", "path/to/file.tif"),
        ("s3a://my-bucket/deep/path/file.tif", "deep/path/file.tif"),
        ("gs://my-bucket/data/image.tif", "data/image.tif"),
        ("https://example.com/data/image.tif", "data/image.tif"),
        ("http://localhost:8080/tiles/tile.tif", "tiles/tile.tif"),
    ],
)
def test_path_extraction(href, expected_path):
    """The returned path is the URL path component without a leading slash."""
    _, path = store_from_href(href)
    assert path == expected_path


@pytest.mark.parametrize(
    "href",
    [
        "s3://bucket/a.tif",
        "s3a://bucket/a.tif",
        "gs://bucket/a.tif",
        "https://host.com/a.tif",
        "http://host.com/a.tif",
    ],
)
def test_store_is_not_none(href):
    """A store object is returned for all supported schemes."""
    store, _ = store_from_href(href)
    assert store is not None


def test_unsupported_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        store_from_href("ftp://server/file.tif")


def test_thread_local_cache_same_bucket():
    """Two HREFs in the same bucket return the same store object."""
    store_a, _ = store_from_href("s3://shared-bucket/file1.tif")
    store_b, _ = store_from_href("s3://shared-bucket/file2.tif")
    assert store_a is store_b


def test_thread_local_cache_different_buckets():
    """HREFs in different buckets return distinct store objects."""
    store_a, _ = store_from_href("s3://bucket-one/file.tif")
    store_b, _ = store_from_href("s3://bucket-two/file.tif")
    assert store_a is not store_b


def test_thread_local_cache_same_https_host():
    """Two HTTPS HREFs on the same host share a store."""
    store_a, _ = store_from_href("https://cdn.example.com/img/a.tif")
    store_b, _ = store_from_href("https://cdn.example.com/img/b.tif")
    assert store_a is store_b
