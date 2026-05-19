"""Event loop and executor ownership for lazycogs background work."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import weakref
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

_REPROJECT_WORKERS_ENV = "LAZYCOGS_REPROJECT_WORKERS"
_DUCKDB_MAX_WORKERS = 1
_DUCKDB_MAX_SUBMISSIONS = 2

_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_THREAD: threading.Thread | None = None
_REPROJECT_POOL: concurrent.futures.ThreadPoolExecutor | None = None
_DUCKDB_POOL: concurrent.futures.ThreadPoolExecutor | None = None
_DUCKDB_SUBMISSION_GATES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    asyncio.Semaphore,
] = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


def _default_workers() -> int:
    """Return the default worker count: CPUs up to a cap of 4.

    Reprojection (pyproj + numpy) is memory-bandwidth-bound, not compute-bound.
    Benchmarks show diminishing returns beyond 4 concurrent threads because they
    saturate the memory bus rather than adding CPU throughput. Keep the default
    conservative.
    """
    return min(os.cpu_count() or 1, 4)


def _reproject_worker_count() -> int:
    """Return the configured reprojection worker count."""
    value = os.getenv(_REPROJECT_WORKERS_ENV)
    if value is None:
        return _default_workers()

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{_REPROJECT_WORKERS_ENV} must be an integer, got {value!r}",
        ) from exc

    if parsed < 1:
        raise ValueError(f"worker count must be >= 1, got {parsed!r}")
    return parsed


def _start_background_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Create and start the shared background event loop."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True, name="lazycogs-loop")
    thread.start()
    ready.wait()
    return loop, thread


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, starting it lazily."""
    global _LOOP, _LOOP_THREAD

    with _LOCK:
        if (
            _LOOP is not None
            and _LOOP_THREAD is not None
            and _LOOP_THREAD.is_alive()
            and _LOOP.is_running()
            and not _LOOP.is_closed()
        ):
            return _LOOP

        _LOOP, _LOOP_THREAD = _start_background_loop()
        return _LOOP


def get_reproject_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared bounded reprojection executor."""
    global _REPROJECT_POOL  # noqa: PLW0603

    with _LOCK:
        if _REPROJECT_POOL is None:
            _REPROJECT_POOL = concurrent.futures.ThreadPoolExecutor(
                max_workers=_reproject_worker_count(),
                thread_name_prefix="lazycogs-reproject",
            )
        return _REPROJECT_POOL


def get_duckdb_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared bounded DuckDB executor."""
    global _DUCKDB_POOL  # noqa: PLW0603

    with _LOCK:
        if _DUCKDB_POOL is None:
            _DUCKDB_POOL = concurrent.futures.ThreadPoolExecutor(
                max_workers=_DUCKDB_MAX_WORKERS,
                thread_name_prefix="lazycogs-duckdb",
            )
        return _DUCKDB_POOL


def _duckdb_submission_gate() -> asyncio.Semaphore:
    """Return the private per-loop gate for DuckDB submissions."""
    loop = asyncio.get_running_loop()
    with _LOCK:
        gate = _DUCKDB_SUBMISSION_GATES.get(loop)
        if gate is None:
            gate = asyncio.Semaphore(_DUCKDB_MAX_SUBMISSIONS)
            _DUCKDB_SUBMISSION_GATES[loop] = gate
        return gate


async def run_duckdb[**P, T](
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run blocking DuckDB work on the shared bounded executor.

    Submission is gated per event loop to keep the executor queue small while
    still yielding the caller's loop during the blocking query.
    """
    loop = asyncio.get_running_loop()
    async with _duckdb_submission_gate():
        return await loop.run_in_executor(
            get_duckdb_pool(),
            partial(func, *args, **kwargs),
        )


def _submit_to_loop[T](
    coro: Coroutine[object, object, T],
) -> concurrent.futures.Future[T]:
    """Submit a coroutine to the shared background loop."""
    loop = _ensure_loop()
    thread = _LOOP_THREAD
    if thread is None or not thread.is_alive() or not loop.is_running():
        coro.close()
        raise RuntimeError("lazycogs background event loop is not running")
    if thread.ident == threading.get_ident():
        coro.close()
        raise RuntimeError(
            "Cannot call sync lazycogs bridge from the lazycogs event loop thread. "
            "Await the async API directly instead.",
        )
    return asyncio.run_coroutine_threadsafe(coro, loop)


def run_on_loop[T](coro: Coroutine[object, object, T]) -> T:
    """Run ``coro`` on the shared lazycogs event loop and return its result.

    This is the supported helper for sync code that must execute a coroutine on
    the lazycogs background loop, including callers that need to construct
    loop-bound resources on that loop.
    """
    return _submit_to_loop(coro).result()
