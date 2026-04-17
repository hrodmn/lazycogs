"""Configure the per-event-loop thread pool size for CPU-bound reprojection work."""

from __future__ import annotations

import os

_MAX_WORKERS: int | None = None


def _default_workers() -> int:
    """Return the default worker count: CPUs up to a cap of 4.

    Reprojection (pyproj + numpy) is memory-bandwidth-bound, not compute-bound.
    Benchmarks show diminishing returns beyond 4 concurrent threads because they
    saturate the memory bus rather than adding CPU throughput. Keep the default
    conservative.
    """
    return min(os.cpu_count() or 4, 4)


def get_max_workers() -> int:
    """Return the configured worker count, or the default if not set.

    Returns:
        Number of reprojection threads each event loop will use.

    """
    return _MAX_WORKERS if _MAX_WORKERS is not None else _default_workers()


def set_reproject_workers(n: int) -> None:
    """Set the number of threads each chunk's event loop uses for reprojection.

    Each chunk read creates a fresh asyncio event loop with its own dedicated
    ``ThreadPoolExecutor`` bounded to ``n`` workers. Dask tasks do not compete
    for a shared pool — each task gets ``n`` independent reprojection threads.
    Total reprojection threads at any moment is at most
    ``n × dask_worker_count``.

    Reprojection is memory-bandwidth-bound rather than compute-bound, so values
    above 4 typically offer no benefit and can hurt throughput due to memory
    contention. The default is ``min(os.cpu_count(), 4)``.

    To improve overall throughput, prefer adding time or band parallelism via
    dask (``chunks={"time": 1}``) over raising this value.

    Args:
        n: Number of worker threads per event loop.  Must be >= 1.

    Raises:
        ValueError: If ``n`` is less than 1.

    """
    global _MAX_WORKERS
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    _MAX_WORKERS = n
