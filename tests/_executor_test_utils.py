"""Test-only helpers for resetting lazycogs executor state."""

from __future__ import annotations

import asyncio

from lazycogs import _executor


async def _cancel_loop_tasks() -> None:
    """Cancel outstanding tasks on the current event loop."""
    current = asyncio.current_task()
    tasks = [task for task in asyncio.all_tasks() if task is not current]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def reset_executor_state_for_tests() -> None:
    """Reset lazycogs executor singletons between tests."""
    with _executor._LOCK:
        loop = _executor._LOOP
        thread = _executor._LOOP_THREAD
        reproject_pool = _executor._REPROJECT_POOL
        duckdb_pool = _executor._DUCKDB_POOL
        _executor._LOOP = None
        _executor._LOOP_THREAD = None
        _executor._REPROJECT_POOL = None
        _executor._DUCKDB_POOL = None
        _executor._DUCKDB_SUBMISSION_GATES.clear()

    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_cancel_loop_tasks(), loop)
        try:
            future.result(timeout=5)
        except Exception:
            future.cancel()
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread.is_alive():
        thread.join(timeout=5)
    if loop is not None and not loop.is_closed():
        loop.close()
    if reproject_pool is not None:
        reproject_pool.shutdown(wait=True, cancel_futures=True)
    if duckdb_pool is not None:
        duckdb_pool.shutdown(wait=True, cancel_futures=True)
