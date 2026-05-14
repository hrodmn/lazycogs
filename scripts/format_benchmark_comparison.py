#!/usr/bin/env python3
"""Format a pytest-benchmark comparison as GitHub-flavored markdown.

Usage:
    uv run python scripts/format_benchmark_comparison.py \
        --baseline '.benchmarks/**/*_main.json' \
        --pr '.benchmarks/**/*_pr-*.json'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REGRESSION_THRESHOLD_PCT = 10
_SMALL_WINDOW_LABEL = "Small-window reprojection microbenchmarks"
_END_TO_END_LABEL = "End-to-end benchmarks"
_RESOURCE_FIELDS = (
    ("profile_peak_rss_mb", "Peak RSS (MB)"),
    ("profile_cpu_total_s", "CPU total (s)"),
    ("profile_cpu_per_wall", "CPU/wall"),
)


def find_file(pattern: str) -> Path:
    """Find the most recently modified file matching the given glob pattern."""
    matches = list(Path().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match: {pattern!r}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_benchmarks(path: Path) -> dict[str, dict]:
    """Load benchmark records keyed by test name from a pytest-benchmark JSON file."""
    with path.open() as f:
        data = json.load(f)
    return {
        benchmark["name"]: {
            "stats": benchmark["stats"],
            "extra_info": benchmark.get("extra_info", {}),
        }
        for benchmark in data["benchmarks"]
    }


def _ms(seconds: float) -> str:
    """Format a duration in seconds as a millisecond string."""
    return f"{seconds * 1000:.1f}"


def _classify_benchmark(name: str) -> str:
    """Return the report section label for a benchmark name."""
    if "small_window" in name:
        return _SMALL_WINDOW_LABEL
    return _END_TO_END_LABEL


def _change_display(baseline_value: float, pr_value: float) -> str:
    """Return a signed percent-change display string."""
    if baseline_value == 0:
        return "n/a"
    pct = (pr_value - baseline_value) / baseline_value * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _comparison_row(name: str, baseline_record: dict, pr_record: dict) -> str:
    """Return one markdown table row for a benchmark present in both runs."""
    base_mean = baseline_record["stats"]["mean"]
    pr_mean = pr_record["stats"]["mean"]

    if base_mean == 0:
        pct_display = "n/a"
        flag = ""
    else:
        pct = (pr_mean - base_mean) / base_mean * 100
        pct_display = _change_display(base_mean, pr_mean)
        flag = " :warning:" if pct > REGRESSION_THRESHOLD_PCT else ""

    base_ms, pr_ms = _ms(base_mean), _ms(pr_mean)
    return f"| `{name}` | {base_ms} | {pr_ms} | {pct_display}{flag} |"


def _resource_row(
    name: str,
    field: str,
    baseline_record: dict,
    pr_record: dict,
) -> str | None:
    """Return one markdown row for a shared resource metric, if present."""
    baseline_extra = baseline_record.get("extra_info", {})
    pr_extra = pr_record.get("extra_info", {})
    if field not in baseline_extra or field not in pr_extra:
        return None

    baseline_value = baseline_extra[field]
    pr_value = pr_extra[field]
    if not isinstance(baseline_value, int | float) or not isinstance(
        pr_value,
        int | float,
    ):
        return None

    return (
        f"| `{name}` | {baseline_value:.1f} | {pr_value:.1f} | "
        f"{_change_display(float(baseline_value), float(pr_value))} |"
    )


def _render_comparison_section(
    heading: str,
    names: list[str],
    baseline: dict[str, dict],
    pr: dict[str, dict],
) -> str:
    """Render one benchmark comparison table section."""
    rows = [_comparison_row(name, baseline[name], pr[name]) for name in names]
    return "\n".join(
        [
            f"### {heading}",
            "",
            "| Test | Baseline (ms) | PR (ms) | Change |",
            "|------|:-------------:|:-------:|-------:|",
            *rows,
        ],
    )


def _render_resource_section(
    heading: str,
    names: list[str],
    baseline: dict[str, dict],
    pr: dict[str, dict],
) -> str | None:
    """Render resource-profile tables for one benchmark section."""
    field_sections: list[str] = []
    for field, label in _RESOURCE_FIELDS:
        rows = [
            row
            for name in names
            if (row := _resource_row(name, field, baseline[name], pr[name])) is not None
        ]
        if not rows:
            continue
        field_sections.append(
            "\n".join(
                [
                    f"#### Resource profile: {label}",
                    "",
                    f"| Test | Baseline {label} | PR {label} | Change |",
                    "|------|------------------:|-----------:|-------:|",
                    *rows,
                ],
            ),
        )

    if not field_sections:
        return None

    return "\n\n".join([f"### {heading} resource profile", *field_sections])


def _render_name_list_section(heading: str, names: list[str]) -> str:
    """Render a markdown bullet list of benchmark names."""
    lines = [f"## {heading}", "", *[f"- `{name}`" for name in names]]
    return "\n".join(lines)


def generate_report(baseline: dict[str, dict], pr: dict[str, dict]) -> str:
    """Generate a markdown benchmark comparison report."""
    shared_names = sorted(set(baseline) & set(pr))
    new_names = sorted(set(pr) - set(baseline))
    missing_names = sorted(set(baseline) - set(pr))

    shared_sections: list[str] = []
    resource_sections: list[str] = []
    for heading in (_END_TO_END_LABEL, _SMALL_WINDOW_LABEL):
        names = [name for name in shared_names if _classify_benchmark(name) == heading]
        if names:
            shared_sections.append(
                _render_comparison_section(heading, names, baseline, pr),
            )
            resource_section = _render_resource_section(heading, names, baseline, pr)
            if resource_section is not None:
                resource_sections.append(resource_section)

    body_parts = [
        "<!-- lazycogs-benchmark-comparison -->",
        "## Benchmark Comparison",
        "",
    ]
    if shared_sections:
        body_parts.extend(shared_sections)
    else:
        body_parts.append("No benchmarks were present in both runs.")

    if resource_sections:
        body_parts.extend(
            ["", "## Resource Profile Comparison", "", *resource_sections],
        )

    if new_names:
        body_parts.extend(
            ["", _render_name_list_section("New benchmarks in PR", new_names)],
        )
    if missing_names:
        body_parts.extend(
            [
                "",
                _render_name_list_section(
                    "Benchmarks missing from PR",
                    missing_names,
                ),
            ],
        )

    body_parts.append("")
    return "\n".join(body_parts)


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        help="Glob pattern for the baseline JSON",
    )
    parser.add_argument("--pr", required=True, help="Glob pattern for the PR JSON")
    args = parser.parse_args()

    try:
        baseline_path = find_file(args.baseline)
        pr_path = find_file(args.pr)
    except FileNotFoundError:
        logger.exception("Benchmark file not found")
        sys.exit(1)

    logger.info("Baseline: %s", baseline_path)
    logger.info("PR:       %s", pr_path)

    report = generate_report(
        load_benchmarks(baseline_path),
        load_benchmarks(pr_path),
    )
    sys.stdout.write(report)


if __name__ == "__main__":
    main()
