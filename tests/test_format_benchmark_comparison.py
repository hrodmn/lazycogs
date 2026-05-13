"""Tests for the benchmark comparison formatter script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "format_benchmark_comparison.py"
_SPEC = importlib.util.spec_from_file_location(
    "format_benchmark_comparison",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
format_benchmark_comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(format_benchmark_comparison)


def test_generate_report_includes_sections_for_shared_new_and_missing() -> None:
    """The report should surface changed, added, and removed benchmarks."""
    baseline = {
        "test_open_overhead": {"mean": 0.010},
        "test_small_window_reprojection_modes[nearest]": {"mean": 0.002},
        "test_removed_benchmark": {"mean": 0.005},
    }
    pr = {
        "test_open_overhead": {"mean": 0.012},
        "test_small_window_reprojection_modes[nearest]": {"mean": 0.001},
        "test_small_window_reprojection_modes[cubic]": {"mean": 0.003},
        "test_new_end_to_end_benchmark": {"mean": 0.020},
    }

    report = format_benchmark_comparison.generate_report(baseline, pr)

    assert "## Benchmark Comparison" in report
    assert "### End-to-end benchmarks" in report
    assert "### Small-window reprojection microbenchmarks" in report
    assert "| `test_open_overhead` | 10.0 | 12.0 | +20.0% :warning: |" in report
    assert (
        "| `test_small_window_reprojection_modes[nearest]` | 2.0 | 1.0 | -50.0% |"
        in report
    )
    assert "## New benchmarks in PR" in report
    assert "`test_small_window_reprojection_modes[cubic]`" in report
    assert "`test_new_end_to_end_benchmark`" in report
    assert "## Benchmarks missing from PR" in report
    assert "`test_removed_benchmark`" in report


def test_generate_report_handles_empty_shared_benchmarks() -> None:
    """The report should still render when only added or removed tests exist."""
    report = format_benchmark_comparison.generate_report(
        baseline={"test_removed": {"mean": 0.005}},
        pr={"test_added": {"mean": 0.007}},
    )

    assert "No benchmarks were present in both runs." in report
    assert "`test_added`" in report
    assert "`test_removed`" in report
