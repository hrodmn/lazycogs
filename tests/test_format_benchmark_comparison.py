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
    """The report should surface changed, added, removed, and resource-profile data."""
    baseline = {
        "test_open_overhead": {
            "stats": {"mean": 0.010},
            "extra_info": {
                "profile_peak_rss_mb": 100.0,
                "profile_cpu_total_s": 1.5,
                "profile_cpu_per_wall": 1.2,
            },
        },
        "test_small_window_reprojection_modes[nearest]": {
            "stats": {"mean": 0.002},
            "extra_info": {
                "profile_peak_rss_mb": 50.0,
                "profile_cpu_total_s": 0.2,
                "profile_cpu_per_wall": 0.8,
            },
        },
        "test_removed_benchmark": {
            "stats": {"mean": 0.005},
            "extra_info": {},
        },
    }
    pr = {
        "test_open_overhead": {
            "stats": {"mean": 0.012},
            "extra_info": {
                "profile_peak_rss_mb": 120.0,
                "profile_cpu_total_s": 1.1,
                "profile_cpu_per_wall": 0.9,
            },
        },
        "test_small_window_reprojection_modes[nearest]": {
            "stats": {"mean": 0.001},
            "extra_info": {
                "profile_peak_rss_mb": 40.0,
                "profile_cpu_total_s": 0.1,
                "profile_cpu_per_wall": 0.4,
            },
        },
        "test_small_window_reprojection_modes[cubic]": {
            "stats": {"mean": 0.003},
            "extra_info": {},
        },
        "test_new_end_to_end_benchmark": {
            "stats": {"mean": 0.020},
            "extra_info": {},
        },
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
    assert "## Resource Profile Comparison" in report
    assert "#### Resource profile: Peak RSS (MB)" in report
    assert "| `test_open_overhead` | 100.0 | 120.0 | +20.0% |" in report
    assert "#### Resource profile: CPU total (s)" in report
    assert "| `test_open_overhead` | 1.5 | 1.1 | -26.7% |" in report
    assert "#### Resource profile: CPU/wall" in report
    assert (
        "| `test_small_window_reprojection_modes[nearest]` | 0.8 | 0.4 | -50.0% |"
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
        baseline={"test_removed": {"stats": {"mean": 0.005}, "extra_info": {}}},
        pr={"test_added": {"stats": {"mean": 0.007}, "extra_info": {}}},
    )

    assert "No benchmarks were present in both runs." in report
    assert "`test_added`" in report
    assert "`test_removed`" in report


def test_load_benchmarks_keeps_stats_and_extra_info(tmp_path: Path) -> None:
    """The loader should preserve profiling metadata from pytest-benchmark JSON."""
    benchmark_json = tmp_path / "bench.json"
    benchmark_json.write_text(
        """
{
  "benchmarks": [
    {
      "name": "test_full_compute",
      "stats": {"mean": 0.123},
      "extra_info": {"profile_peak_rss_mb": 256.0}
    }
  ]
}
""".strip(),
    )

    loaded = format_benchmark_comparison.load_benchmarks(benchmark_json)

    assert loaded == {
        "test_full_compute": {
            "stats": {"mean": 0.123},
            "extra_info": {"profile_peak_rss_mb": 256.0},
        },
    }
