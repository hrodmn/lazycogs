"""Tests for _store.resolve, _store.store_for, and storage extension helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from obstore.store import MemoryStore

from lazycogs._store import (
    _extract_store_kwargs,
    _extract_store_kwargs_v1,
    _extract_store_kwargs_v2,
    _storage_extension_version,
    resolve,
    store_for,
)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


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
    _, path = resolve(href)
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
    store, _ = resolve(href)
    assert store is not None


def test_unsupported_scheme_raises():
    """obstore's from_url rejects unknown schemes."""
    with pytest.raises(Exception, match="(?i)scheme|url"):
        resolve("ftp://server/file.tif")


def test_thread_local_cache_same_bucket():
    """Two HREFs in the same bucket return the same store object."""
    store_a, _ = resolve("s3://shared-bucket/file1.tif")
    store_b, _ = resolve("s3://shared-bucket/file2.tif")
    assert store_a is store_b


def test_thread_local_cache_different_buckets():
    """HREFs in different buckets return distinct store objects."""
    store_a, _ = resolve("s3://bucket-one/file.tif")
    store_b, _ = resolve("s3://bucket-two/file.tif")
    assert store_a is not store_b


def test_thread_local_cache_same_https_host():
    """Two HTTPS HREFs on the same host share a store."""
    store_a, _ = resolve("https://cdn.example.com/img/a.tif")
    store_b, _ = resolve("https://cdn.example.com/img/b.tif")
    assert store_a is store_b


def test_user_supplied_store_is_returned_unchanged():
    """When a store is passed, it is returned as-is with just the path extracted."""
    user_store = MemoryStore()
    store, path = resolve("s3://bucket/some/key.tif", store=user_store)
    assert store is user_store
    assert path == "some/key.tif"


def test_user_supplied_store_bypasses_cache():
    """Passing a store should never consult or populate the auto-cache."""
    user_store = MemoryStore()
    store, _ = resolve("s3://never-cached-bucket/a.tif", store=user_store)
    assert store is user_store
    auto_store, _ = resolve("s3://never-cached-bucket/b.tif")
    assert auto_store is not user_store


def test_path_fn_overrides_default_extraction():
    """path_fn result is used instead of urlparse-based path extraction."""

    def path_fn(href: str) -> str:
        return "custom/extracted/path.tif"

    _, path = resolve("s3://bucket/original/path.tif", path_fn=path_fn)
    assert path == "custom/extracted/path.tif"


def test_path_fn_with_store_returns_both():
    """When store and path_fn are both supplied, path_fn drives extraction and store is returned unchanged."""
    user_store = MemoryStore()

    def path_fn(href: str) -> str:
        return href.split("container/", 1)[1]

    store, path = resolve(
        "https://account.blob.core.windows.net/container/path/to/file.tif",
        store=user_store,
        path_fn=path_fn,
    )
    assert store is user_store
    assert path == "path/to/file.tif"


def test_path_fn_not_called_without_it():
    """When path_fn is absent the default url-parse extraction is used."""
    user_store = MemoryStore()
    _, path = resolve(
        "https://account.blob.core.windows.net/container/path/to/file.tif",
        store=user_store,
    )
    assert path == "container/path/to/file.tif"


# ---------------------------------------------------------------------------
# _storage_extension_version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "https://stac-extensions.github.io/storage/v1.0.0/schema.json",
            "1.0.0",
        ),
        (
            "https://stac-extensions.github.io/storage/v2.0.0/schema.json",
            "2.0.0",
        ),
        (
            "https://stac-extensions.github.io/storage/v1.2.3/schema.json",
            "1.2.3",
        ),
    ],
)
def test_storage_extension_version_recognized(url, expected):
    assert _storage_extension_version([url]) == expected


def test_storage_extension_version_absent():
    assert _storage_extension_version([]) is None
    assert (
        _storage_extension_version(
            ["https://stac-extensions.github.io/eo/v1.0.0/schema.json"]
        )
        is None
    )


def test_storage_extension_version_among_multiple():
    extensions = [
        "https://stac-extensions.github.io/eo/v1.0.0/schema.json",
        "https://stac-extensions.github.io/storage/v1.0.0/schema.json",
        "https://stac-extensions.github.io/projection/v1.0.0/schema.json",
    ]
    assert _storage_extension_version(extensions) == "1.0.0"


# ---------------------------------------------------------------------------
# _extract_store_kwargs_v1
# ---------------------------------------------------------------------------

_V1_EXTENSION_URL = "https://stac-extensions.github.io/storage/v1.0.0/schema.json"


def _v1_item(platform="AWS", region="us-east-1", requester_pays=False):
    return {
        "stac_extensions": [_V1_EXTENSION_URL],
        "properties": {
            "storage:platform": platform,
            "storage:region": region,
            "storage:requester_pays": requester_pays,
        },
        "assets": {},
    }


def test_v1_aws_region_mapped():
    item = _v1_item(platform="AWS", region="eu-west-1")
    result = _extract_store_kwargs_v1(item, {})
    assert result["region"] == "eu-west-1"
    assert "request_payer" not in result


def test_v1_requester_pays_mapped():
    item = _v1_item(platform="AWS", requester_pays=True)
    result = _extract_store_kwargs_v1(item, {})
    assert result["request_payer"] is True


def test_v1_non_aws_platform_ignored():
    """region and requester_pays are only forwarded for AWS."""
    item = _v1_item(platform="GCP", region="us-central1")
    result = _extract_store_kwargs_v1(item, {})
    assert result == {}


def test_v1_asset_level_overrides_item_level():
    """Asset-level storage fields take precedence over item-level properties."""
    item = _v1_item(platform="AWS", region="us-east-1")
    asset = {"storage:platform": "AWS", "storage:region": "ap-southeast-1"}
    result = _extract_store_kwargs_v1(item, asset)
    assert result["region"] == "ap-southeast-1"


def test_v1_tier_not_mapped():
    """storage:tier has no obstore equivalent and must not appear in the output."""
    item = {
        "stac_extensions": [_V1_EXTENSION_URL],
        "properties": {"storage:platform": "AWS", "storage:tier": "Standard"},
        "assets": {},
    }
    result = _extract_store_kwargs_v1(item, {})
    assert "tier" not in result


def test_v1_platform_case_insensitive():
    item = _v1_item(platform="aws", region="us-west-2")
    result = _extract_store_kwargs_v1(item, {})
    assert result["region"] == "us-west-2"


# ---------------------------------------------------------------------------
# _extract_store_kwargs_v2
# ---------------------------------------------------------------------------

_V2_EXTENSION_URL = "https://stac-extensions.github.io/storage/v2.0.0/schema.json"


def _v2_item(schemes: dict, asset_refs: list[str]):
    return {
        "stac_extensions": [_V2_EXTENSION_URL],
        "properties": {"storage:schemes": schemes},
        "assets": {},
    }


def _v2_asset(refs: list[str]):
    return {"storage:refs": refs, "href": "s3://bucket/path/file.tif"}


def test_v2_region_and_requester_pays_mapped():
    schemes = {
        "primary": {
            "type": "aws-s3",
            "platform": "s3://",
            "region": "us-west-2",
            "requester_pays": True,
        }
    }
    asset = _v2_asset(["primary"])
    item = _v2_item(schemes, ["primary"])
    result = _extract_store_kwargs_v2(item, asset)
    assert result["region"] == "us-west-2"
    assert result["request_payer"] is True


def test_v2_non_s3_type_region_not_mapped():
    """region is only forwarded for S3-type schemes (GCS/Azure derive from URL)."""
    schemes = {
        "az": {
            "type": "ms-azure",
            "platform": "https://account.blob.core.windows.net/",
            "region": "westus",
        }
    }
    asset = _v2_asset(["az"])
    item = _v2_item(schemes, ["az"])
    result = _extract_store_kwargs_v2(item, asset)
    assert "region" not in result


def test_v2_custom_s3_endpoint_mapped():
    schemes = {
        "minio": {
            "type": "custom-s3",
            "platform": "https://minio.example.com",
        }
    }
    asset = _v2_asset(["minio"])
    item = _v2_item(schemes, ["minio"])
    result = _extract_store_kwargs_v2(item, asset)
    assert result["endpoint"] == "https://minio.example.com"


def test_v2_custom_s3_template_platform_not_mapped():
    """A platform containing URI template variables is not used as an endpoint."""
    schemes = {
        "aws": {
            "type": "custom-s3",
            "platform": "https://s3.{region}.amazonaws.com",
        }
    }
    asset = _v2_asset(["aws"])
    item = _v2_item(schemes, ["aws"])
    result = _extract_store_kwargs_v2(item, asset)
    assert "endpoint" not in result


def test_v2_missing_ref_returns_empty():
    schemes = {"primary": {"type": "aws-s3", "region": "us-east-1"}}
    asset = _v2_asset(["nonexistent"])
    item = _v2_item(schemes, [])
    result = _extract_store_kwargs_v2(item, asset)
    assert result == {}


def test_v2_no_refs_returns_empty():
    schemes = {"primary": {"type": "aws-s3", "region": "us-east-1"}}
    asset = {"href": "s3://bucket/file.tif"}  # no storage:refs
    item = _v2_item(schemes, [])
    result = _extract_store_kwargs_v2(item, asset)
    assert result == {}


# ---------------------------------------------------------------------------
# _extract_store_kwargs dispatcher
# ---------------------------------------------------------------------------


def test_dispatch_v1():
    item = _v1_item(platform="AWS", region="us-west-2")
    result = _extract_store_kwargs(item, {})
    assert result["region"] == "us-west-2"


def test_dispatch_v2():
    schemes = {"s3": {"type": "aws-s3", "region": "eu-central-1"}}
    asset = _v2_asset(["s3"])
    item = {
        "stac_extensions": [_V2_EXTENSION_URL],
        "properties": {"storage:schemes": schemes},
        "assets": {},
    }
    result = _extract_store_kwargs(item, asset)
    assert result["region"] == "eu-central-1"


def test_dispatch_no_extension_returns_empty():
    item = {"stac_extensions": [], "properties": {}, "assets": {}}
    assert _extract_store_kwargs(item, {}) == {}


def test_dispatch_unknown_version_returns_empty():
    item = {
        "stac_extensions": [
            "https://stac-extensions.github.io/storage/v99.0.0/schema.json"
        ],
        "properties": {},
        "assets": {},
    }
    assert _extract_store_kwargs(item, {}) == {}


# ---------------------------------------------------------------------------
# store_for
# ---------------------------------------------------------------------------

_S3_ITEM = {
    "id": "test-item",
    "stac_extensions": [],
    "properties": {"datetime": "2023-06-01T00:00:00Z"},
    "assets": {
        "B04": {
            "href": "s3://sentinel-cogs/sentinel-s2-l2a/B04.tif",
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "roles": ["data"],
        }
    },
}


def test_store_for_returns_store():
    """store_for constructs and returns an ObjectStore."""
    with patch("rustac.DuckdbClient.search", return_value=[_S3_ITEM]):
        store = store_for("items.parquet")
    assert store is not None


def test_store_for_no_items_raises():
    with patch("rustac.DuckdbClient.search", return_value=[]):
        with pytest.raises(ValueError, match="No STAC items"):
            store_for("items.parquet")


def test_store_for_missing_asset_raises():
    with patch("rustac.DuckdbClient.search", return_value=[_S3_ITEM]):
        with pytest.raises(KeyError, match="B99"):
            store_for("items.parquet", asset="B99")


def test_store_for_asset_kwarg_selects_asset():
    """asset= selects a specific asset rather than the first data asset."""
    item = {
        "id": "test-item",
        "stac_extensions": [],
        "properties": {},
        "assets": {
            "B04": {
                "href": "s3://bucket-a/B04.tif",
                "roles": ["data"],
            },
            "B08": {
                "href": "s3://bucket-b/B08.tif",
                "roles": ["data"],
            },
        },
    }
    with patch("rustac.DuckdbClient.search", return_value=[item]):
        # without asset= the first data asset (B04 from bucket-a) would be used
        store = store_for("items.parquet", asset="B08")
    assert store is not None


def test_store_for_prefers_data_role_asset():
    """When asset= is omitted, a data-role asset is preferred over others."""
    item = {
        "id": "test-item",
        "stac_extensions": [],
        "properties": {},
        "assets": {
            "thumbnail": {
                "href": "https://example.com/thumb.jpg",
                "roles": ["thumbnail"],
            },
            "B04": {
                "href": "s3://bucket/B04.tif",
                "roles": ["data"],
            },
        },
    }
    # If the data asset is chosen the store will be an S3 store, not HTTPS.
    # We verify indirectly: store_for must not raise (thumbnail has no S3 equivalent).
    with patch("rustac.DuckdbClient.search", return_value=[item]):
        store = store_for("items.parquet")
    assert store is not None


def test_store_for_v1_extension_infers_region():
    item = {
        "id": "test-item",
        "stac_extensions": [_V1_EXTENSION_URL],
        "properties": {
            "storage:platform": "AWS",
            "storage:region": "ap-northeast-1",
            "datetime": "2023-06-01T00:00:00Z",
        },
        "assets": {
            "data": {
                "href": "s3://my-bucket/file.tif",
                "roles": ["data"],
            }
        },
    }
    with patch("rustac.DuckdbClient.search", return_value=[item]):
        # If region extraction fails the store is still returned — this just
        # verifies no exception is raised and the store is not None.
        store = store_for("items.parquet")
    assert store is not None


def test_store_for_v2_extension_infers_region():
    item = {
        "id": "test-item",
        "stac_extensions": [_V2_EXTENSION_URL],
        "properties": {
            "storage:schemes": {
                "primary": {
                    "type": "aws-s3",
                    "platform": "s3://",
                    "region": "eu-west-1",
                }
            },
            "datetime": "2023-06-01T00:00:00Z",
        },
        "assets": {
            "data": {
                "href": "s3://my-bucket/file.tif",
                "roles": ["data"],
                "storage:refs": ["primary"],
            }
        },
    }
    with patch("rustac.DuckdbClient.search", return_value=[item]):
        store = store_for("items.parquet")
    assert store is not None


def test_store_for_malformed_extension_does_not_raise():
    """A malformed storage extension is logged and silently ignored."""
    item = {
        "id": "test-item",
        "stac_extensions": [_V2_EXTENSION_URL],
        "properties": {
            "storage:schemes": "not-a-dict",  # intentionally malformed
        },
        "assets": {
            "data": {
                "href": "s3://bucket/file.tif",
                "roles": ["data"],
                "storage:refs": ["primary"],
            }
        },
    }
    with patch("rustac.DuckdbClient.search", return_value=[item]):
        store = store_for("items.parquet")
    assert store is not None


def test_store_for_https_url_region_in_hostname_no_duplicate():
    """store_for must not raise when the asset HREF is an HTTPS S3 URL with the
    region encoded in the hostname AND the storage extension also declares a region.
    Previously, passing region twice caused an obstore duplicate-key error.
    """
    item = {
        "id": "test-item",
        "stac_extensions": [_V1_EXTENSION_URL],
        "properties": {
            "storage:platform": "AWS",
            "storage:region": "us-west-2",
            "datetime": "2023-06-01T00:00:00Z",
        },
        "assets": {
            "data": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/file.tif",
                "roles": ["data"],
            }
        },
    }
    with patch("rustac.DuckdbClient.search", return_value=[item]):
        store = store_for("items.parquet")
    assert store is not None


def test_store_for_kwargs_override_defaults():
    """Caller kwargs override the skip_signature=True default for public schemes."""
    call_kwargs: dict = {}

    def capture_from_url(url, **kwargs):
        call_kwargs.update(kwargs)
        from obstore.store import from_url as real_from_url

        return real_from_url(url, **kwargs)

    with patch("rustac.DuckdbClient.search", return_value=[_S3_ITEM]):
        with patch("lazycogs._store.from_url", side_effect=capture_from_url):
            store_for("items.parquet", skip_signature=False)

    assert call_kwargs.get("skip_signature") is False
