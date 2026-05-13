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


def find_file(pattern: str) -> Path:
    """Find the most recently modified file matching the given glob pattern."""
    matches = list(Path().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match: {pattern!r}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_benchmarks(path: Path) -> dict[str, dict]:
    """Load benchmark stats keyed by test name from a pytest-benchmark JSON file."""
    with path.open() as f:
        data = json.load(f)
    return {b["name"]: b["stats"] for b in data["benchmarks"]}


def _ms(seconds: float) -> str:
    """Format a duration in seconds as a millisecond string."""
    return f"{seconds * 1000:.1f}"


def _classify_benchmark(name: str) -> str:
    """Return the report section label for a benchmark name."""
    if "small_window" in name:
        return _SMALL_WINDOW_LABEL
    return _END_TO_END_LABEL


def _comparison_row(name: str, baseline_stats: dict, pr_stats: dict) -> str:
    """Return one markdown table row for a benchmark present in both runs."""
    base_mean = baseline_stats["mean"]
    pr_mean = pr_stats["mean"]

    if base_mean == 0:
        pct_display = "n/a"
        flag = ""
    else:
        pct = (pr_mean - base_mean) / base_mean * 100
        sign = "+" if pct >= 0 else ""
        pct_display = f"{sign}{pct:.1f}%"
        flag = " :warning:" if pct > REGRESSION_THRESHOLD_PCT else ""

    base_ms, pr_ms = _ms(base_mean), _ms(pr_mean)
    return f"| `{name}` | {base_ms} | {pr_ms} | {pct_display}{flag} |"


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
    for heading in (_END_TO_END_LABEL, _SMALL_WINDOW_LABEL):
        names = [name for name in shared_names if _classify_benchmark(name) == heading]
        if names:
            shared_sections.append(
                _render_comparison_section(heading, names, baseline, pr),
            )

    body_parts = [
        "<!-- lazycogs-benchmark-comparison -->",
        "## Benchmark Comparison",
        "",
    ]
    if shared_sections:
        body_parts.extend(shared_sections)
    else:
        body_parts.append("No benchmarks were present in both runs.")

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
