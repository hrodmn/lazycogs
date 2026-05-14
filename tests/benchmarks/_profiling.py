"""Resource profiling helpers for benchmark tests."""

from __future__ import annotations

import os
import resource
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass
class ResourceProfile:
    """Resource profile captured during one benchmark run."""

    wall_time_s: float
    cpu_user_s: float
    cpu_system_s: float
    cpu_total_s: float
    peak_rss_mb: float
    rss_before_mb: float
    rss_after_mb: float
    max_cpu_pct_of_wall: float
    sample_count: int


class ResourceSampler:
    """Sample process RSS and CPU usage while a block runs."""

    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._peak_rss_mb = 0.0
        self._max_cpu_pct = 0.0
        self._sample_count = 0
        self._rss_before_mb = 0.0
        self._rss_after_mb = 0.0
        self._wall_start = 0.0
        self._wall_end = 0.0
        self._cpu_start: os.times_result | None = None
        self._cpu_end: os.times_result | None = None

    def __enter__(self) -> Self:
        self._rss_before_mb = _rss_mb()
        self._wall_start = time.perf_counter()
        self._cpu_start = os.times()
        self._peak_rss_mb = self._rss_before_mb
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join()
        self._cpu_end = os.times()
        self._wall_end = time.perf_counter()
        self._rss_after_mb = _rss_mb()
        self._peak_rss_mb = max(self._peak_rss_mb, self._rss_after_mb, _peak_rss_mb())

    def _run(self) -> None:
        last_wall = time.perf_counter()
        last_cpu = os.times()
        while not self._stop.wait(self.interval_s):
            now_wall = time.perf_counter()
            now_cpu = os.times()
            self._sample_count += 1
            self._peak_rss_mb = max(self._peak_rss_mb, _rss_mb(), _peak_rss_mb())
            wall_delta = now_wall - last_wall
            cpu_delta = (now_cpu.user - last_cpu.user) + (
                now_cpu.system - last_cpu.system
            )
            if wall_delta > 0:
                self._max_cpu_pct = max(
                    self._max_cpu_pct,
                    100.0 * cpu_delta / wall_delta,
                )
            last_wall = now_wall
            last_cpu = now_cpu

    def profile(self) -> ResourceProfile:
        """Return the aggregated resource profile."""
        if self._cpu_start is None or self._cpu_end is None:
            raise RuntimeError(
                "ResourceSampler.profile() called before sampling finished.",
            )
        cpu_user_s = self._cpu_end.user - self._cpu_start.user
        cpu_system_s = self._cpu_end.system - self._cpu_start.system
        wall_time_s = self._wall_end - self._wall_start
        return ResourceProfile(
            wall_time_s=wall_time_s,
            cpu_user_s=cpu_user_s,
            cpu_system_s=cpu_system_s,
            cpu_total_s=cpu_user_s + cpu_system_s,
            peak_rss_mb=self._peak_rss_mb,
            rss_before_mb=self._rss_before_mb,
            rss_after_mb=self._rss_after_mb,
            max_cpu_pct_of_wall=self._max_cpu_pct,
            sample_count=self._sample_count,
        )


def _rss_mb() -> float:
    """Return current RSS of this process in MB on Linux."""
    with Path("/proc/self/status").open() as file_handle:
        for line in file_handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _peak_rss_mb() -> float:
    """Return peak RSS of this process in MB on Linux."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024


def add_resource_profile(benchmark, run_once: Callable[[], object]) -> None:
    """Attach a one-shot resource profile to ``benchmark.extra_info``."""
    with ResourceSampler() as resource_sampler:
        run_once()
    profile = resource_sampler.profile()
    benchmark.extra_info.update(
        {
            "profile_wall_s": round(profile.wall_time_s, 4),
            "profile_cpu_user_s": round(profile.cpu_user_s, 4),
            "profile_cpu_system_s": round(profile.cpu_system_s, 4),
            "profile_cpu_total_s": round(profile.cpu_total_s, 4),
            "profile_cpu_per_wall": round(
                (
                    profile.cpu_total_s / profile.wall_time_s
                    if profile.wall_time_s
                    else 0.0
                ),
                4,
            ),
            "profile_peak_rss_mb": round(profile.peak_rss_mb, 1),
            "profile_rss_before_mb": round(profile.rss_before_mb, 1),
            "profile_rss_after_mb": round(profile.rss_after_mb, 1),
            "profile_peak_cpu_pct": round(profile.max_cpu_pct_of_wall, 1),
            "profile_sample_count": profile.sample_count,
        },
    )
