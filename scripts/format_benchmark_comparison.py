#!/usr/bin/env python3
"""Format a pytest-benchmark comparison as a GitHub-flavored markdown table.

Usage:
    uv run python scripts/format_benchmark_comparison.py \
        --baseline '.benchmarks/**/*_main.json' \
        --pr '.benchmarks/**/*_pr-*.json'
"""

import argparse
import glob
import json
import logging
import sys
from pathlib import Path


def find_file(pattern: str) -> Path:
    """Find the most recently modified file matching the given glob pattern."""
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No files match: {pattern!r}")
    return max((Path(m) for m in matches), key=lambda p: p.stat().st_mtime)


def load_benchmarks(path: Path) -> dict[str, dict]:
    """Load benchmark stats keyed by test name from a pytest-benchmark JSON file."""
    with path.open() as f:
        data = json.load(f)
    return {b["name"]: b["stats"] for b in data["benchmarks"]}


def _ms(seconds: float) -> str:
    """Format a duration in seconds as a millisecond string."""
    return f"{seconds * 1000:.1f}"


def generate_report(baseline: dict[str, dict], pr: dict[str, dict]) -> str:
    """Generate a markdown benchmark comparison table."""
    rows = []
    for name in sorted(baseline):
        if name not in pr:
            continue
        base_mean = baseline[name]["mean"]
        pr_mean = pr[name]["mean"]
        pct = (pr_mean - base_mean) / base_mean * 100
        sign = "+" if pct >= 0 else ""
        flag = " :warning:" if pct > 10 else ""
        rows.append(
            f"| `{name}` | {_ms(base_mean)} | {_ms(pr_mean)} | {sign}{pct:.1f}%{flag} |"
        )

    table = "\n".join(
        [
            "| Test | Baseline (ms) | PR (ms) | Change |",
            "|------|:-------------:|:-------:|-------:|",
            *rows,
        ]
    )
    return (
        f"<!-- lazycogs-benchmark-comparison -->\n## Benchmark Comparison\n\n{table}\n"
    )


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", required=True, help="Glob pattern for the baseline JSON"
    )
    parser.add_argument("--pr", required=True, help="Glob pattern for the PR JSON")
    args = parser.parse_args()

    try:
        baseline_path = find_file(args.baseline)
        pr_path = find_file(args.pr)
    except FileNotFoundError as e:
        logging.error("%s", e)
        sys.exit(1)

    logging.info("Baseline: %s", baseline_path)
    logging.info("PR:       %s", pr_path)

    print(generate_report(load_benchmarks(baseline_path), load_benchmarks(pr_path)))


if __name__ == "__main__":
    main()
