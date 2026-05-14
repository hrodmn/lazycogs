"""Thread pool and event loop configuration for reprojection and DuckDB work."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Coroutine

config: dict[str, int | None] = {"max_workers": None}

_tls = threading.local()

# DuckDB queries serialise on a single connection, so a small pool is enough.
# Kept separate from the reprojection executor so a long reprojection cannot
# starve a queued query within the same chunk read.
_DUCKDB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="lazycogs-duckdb",
)


def _default_workers() -> int:
    """Return the default worker count: CPUs up to a cap of 4.

    Reprojection is memory-bandwidth-bound, not compute-bound.
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
    val = config["max_workers"]
    return val if val is not None else _default_workers()


def set_reproject_workers(n: int) -> None:
    """Set the number of threads each thread's event loop uses for reprojection.

    Each thread (dask worker, Jupyter kernel callback thread, etc.) gets one
    persistent background event loop with one bounded reprojection
    ``ThreadPoolExecutor``.  All chunk reads on that thread share the same loop
    and executor.  Dask tasks on different threads do not compete for a shared
    pool.  Total reprojection threads at any moment is at most
    ``n x active_thread_count``.

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
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    config["max_workers"] = n


def _get_or_create_background_loop() -> asyncio.AbstractEventLoop:
    """Return the persistent background event loop for the current thread.

    Creates the loop, its bounded reprojection executor, and its daemon runner
    thread on first call from a given thread.  Subsequent calls on the same
    thread return the cached loop immediately.

    The loop is stored on ``threading.local`` so each thread (each dask worker,
    each Jupyter kernel callback thread) has its own independent loop and
    executor — tasks on different threads do not share a pool.

    Using a persistent loop (rather than a fresh one per call) ensures that
    any in-flight callbacks from ``async_geotiff``/``obstore`` background
    threads can always be delivered, avoiding ``RuntimeError: Event loop is
    closed`` errors from callbacks that fire after a fresh loop is torn down.
    """
    loop: asyncio.AbstractEventLoop | None = getattr(_tls, "loop", None)
    if loop is not None and loop.is_running():
        return loop

    loop = asyncio.new_event_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=get_max_workers(),
            thread_name_prefix="lazycogs-reproject",
        ),
    )
    t = threading.Thread(target=loop.run_forever, daemon=True, name="lazycogs-loop")
    t.start()
    _tls.loop = loop
    return loop


def _run_coroutine[T](coro: Coroutine[object, object, T]) -> T:
    """Run an async coroutine from sync code.

    Submits the coroutine to a persistent per-thread background event loop,
    blocking until it completes.  The background loop is created on the first
    call from a given thread (dask worker, Jupyter kernel thread, etc.) and
    reused for all subsequent calls on that same thread.

    Args:
        coro: The coroutine to execute.

    Returns:
        The return value of the coroutine.

    """
    loop = _get_or_create_background_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()
