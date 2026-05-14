"""Tests for the benchmark data preparation script."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_benchmark_data.py"
_SPEC = importlib.util.spec_from_file_location("prepare_benchmark_data", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
prepare_benchmark_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_benchmark_data)


def test_main_rewrites_expanded_parquet_without_overwrite(
    tmp_path,
    monkeypatch,
) -> None:
    """Running the script refreshes expanded parquet HREFs even when it exists."""
    monkeypatch.setattr(prepare_benchmark_data, "DATA_DIR", tmp_path)

    raw_parquet = tmp_path / "raw_items.parquet"
    raw_parquet.write_text("placeholder")
    expanded_parquet = tmp_path / "expanded_benchmark_items.parquet"
    expanded_parquet.write_text("stale")

    source_items = [
        {
            "id": "item-001",
            "assets": {
                "red": {"href": "https://example.com/red.tif"},
                "nir08": {"href": "https://example.com/nir08.tif"},
            },
            "properties": {"datetime": "2025-07-04T00:00:00Z"},
        },
    ]
    writes: dict[str, list[dict]] = {}

    async def _fake_search_to(*args, **kwargs) -> None:
        raise AssertionError(
            "search_to should not run when raw_items.parquet already exists",
        )

    def _fake_search_sync(path: str, **kwargs) -> list[dict]:
        assert path == str(raw_parquet)
        assert kwargs == {"use_duckdb": True}
        return source_items

    async def _fake_download(href: str, dest: Path, **kwargs) -> None:
        await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(dest.write_text, f"downloaded from {href}")

    def _fake_write_sync(path: str, items: list[dict]) -> None:
        writes[path] = items
        Path(path).write_text("rewritten")

    monkeypatch.setattr(prepare_benchmark_data.rustac, "search_to", _fake_search_to)
    monkeypatch.setattr(
        prepare_benchmark_data.rustac,
        "search_sync",
        _fake_search_sync,
    )
    monkeypatch.setattr(prepare_benchmark_data, "_download", _fake_download)
    monkeypatch.setattr(prepare_benchmark_data.rustac, "write_sync", _fake_write_sync)

    asyncio.run(prepare_benchmark_data.main(overwrite=False))

    benchmark_parquet = tmp_path / "benchmark_items.parquet"
    assert str(benchmark_parquet) in writes
    assert str(expanded_parquet) in writes

    expanded_items = writes[str(expanded_parquet)]
    assert len(expanded_items) == len(prepare_benchmark_data.SYNTHETIC_DATES)
    assert expanded_items[0]["id"] == "synthetic-0000"
    assert expanded_items[0]["properties"]["datetime"] == "2024-01-15T12:00:00Z"

    expected_red_href = (tmp_path / "cogs" / "item-001" / "red.tif").as_uri()
    assert (
        writes[str(benchmark_parquet)][0]["assets"]["red"]["href"] == expected_red_href
    )
    assert expanded_items[0]["assets"]["red"]["href"] == expected_red_href
