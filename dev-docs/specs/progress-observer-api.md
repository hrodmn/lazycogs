# Spec: Interoperable progress observer API for lazy reads

## Context

lazycogs currently exposes read-planning via `da.lazycogs.explain()` and execution via xarray materialization calls such as `da.load()`, `da.compute()`, and `await da.load_async()`, but it does not expose execution progress. A recent implementation plan narrowed the immediate UX target to notebook-friendly progress, but the preferred direction is a future-proof primitive that broader ecosystem tools can consume without making `tqdm`, dask, or any specific renderer part of lazycogs' core contract.

This spec defines that primitive. It treats `da.lazycogs.explain()` as the existing planning API for computing total work up front, then adds a small callback-based observer contract for execution-time progress.

Related artifact:
- `dev-docs/plans/2026-05-16-002-feat-progress-indicator-plan.md`

## Goals

- Expose an opt-in, renderer-agnostic progress API for lazycogs reads.
- Reuse `da.lazycogs.explain()` to determine total work up front whenever possible.
- Define a stable progress unit that matches real lazycogs execution.
- Support both unchunked reads and dask-backed chunked reads created by `open(..., chunks=...)`.
- Preserve existing behavior and overhead when progress is not enabled.
- Make `tqdm` the first adapter, not the core abstraction.

### Non-goals

- Byte-accurate HTTP range request progress.
- Distributed cross-process progress aggregation.
- A plugin/event-bus framework.
- Renderer-specific core APIs for `tqdm`, `rich`, widgets, or dask.
- Per-COG or per-band completion events in v1.

## Constraints and Assumptions

- `tqdm` must remain an optional user dependency.
- The public API should be usable without dask installed.
- The progress denominator must be known before execution starts when `total="auto"` is used.
- Progress must work for sync (`load()`, `compute()`) and async (`load_async()`) materialization paths.
- Local threaded dask execution is in scope; distributed progress aggregation is not.
- The denominator must align with actual execution units, not with `ExplainPlan.total_chunk_reads`, which is band-expanded and therefore does not match runtime advancement.

## Architecture Overview

The design has three layers:

1. **Planning layer** — `da.lazycogs.explain()` remains the source of truth for preflight work estimation. `ExplainPlan` is extended with a progress-oriented count that matches runtime work units.
2. **Binding layer** — a request-scoped progress session binds a callback to one DataArray materialization request.
3. **Runtime layer** — the backend emits lifecycle events as each execution unit completes.

### Progress unit definition

The canonical progress unit in v1 is:

> one completed `(time step, spatial tile)` materialization request across all selected bands

This matches lazycogs' runtime behavior:
- one DuckDB query per `(time step, spatial tile)`
- one mosaic/read operation across all selected bands for that query result
- zero or more matched COGs inside that unit

This deliberately does **not** count:
- individual COG files
- individual bands
- bytes transferred

### Why this unit

It is the narrowest stable unit that:
- can be counted up front from `explain()`
- can be emitted during execution without deep instrumentation in `_chunk_reader.py`
- is consistent across unchunked and dask-backed reads
- does not double-count multi-band reads

## API or Interface Design

### Public types

```python
@dataclass(frozen=True)
class ProgressEvent:
    type: Literal["start", "advance", "error", "finish"]
    request_id: str
    completed: int
    total: int | None
    unit: Literal["time_spatial_read"]
    metadata: Mapping[str, Any]
    error: Exception | None = None
```

```python
ProgressCallback = Callable[[ProgressEvent], None]
```

### Public binding API

Top-level primitive:

```python
bind_progress(
    da,
    callback: ProgressCallback,
    *,
    total: int | Literal["auto"] = "auto",
    metadata: Mapping[str, Any] | None = None,
) -> ContextManager[None]
```

Accessor sugar:

```python
da.lazycogs.progress(
    callback: ProgressCallback,
    *,
    total: int | Literal["auto"] = "auto",
    metadata: Mapping[str, Any] | None = None,
) -> ContextManager[None]
```

Semantics:
- binds progress for exactly one surrounding materialization scope
- `total="auto"` computes the denominator from `da.lazycogs.explain()`
- explicit `total=<int>` bypasses the preflight explain call
- emits no events outside the active context manager
- nested bindings on the same backend are unsupported in v1 and should raise a clear `RuntimeError`

### Explain integration

`ExplainPlan` gains:

```python
@property
 def total_progress_units(self) -> int:
     ...
```

Semantics:
- returns the number of `(time step, spatial tile)` units for the current DataArray view
- equals `len(time_items) * len(spatial_chunks)` in the current explain implementation
- does not multiply by band count

This property becomes the default denominator for `total="auto"`.

### Optional first-party adapter

v1 may include a tiny adapter helper:

```python
tqdm_callback(bar) -> ProgressCallback
```

Semantics:
- accepts a user-supplied tqdm-like object
- on `start`, sets `bar.total` when available
- on `advance`, calls `bar.update(...)`
- on `finish`, refreshes but does not force-close the bar
- requires no import of `tqdm` inside lazycogs core unless the helper is explicitly used

This helper is convenience only. The stable integration contract is still `ProgressCallback` + `ProgressEvent`.

## Data Model

### Progress session

Internal only.

```python
@dataclass
class _ProgressSession:
    request_id: str
    callback: ProgressCallback
    total: int | None
    completed: int = 0
    unit: str = "time_spatial_read"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)
```

Responsibilities:
- own mutable per-request state
- serialize callback emission from concurrent worker threads / async tasks
- guarantee monotonic `completed`
- centralize start/advance/error/finish event construction

### Runtime metadata

The `metadata` payload should be small and stable in v1. Include only fields that are both useful and cheap to produce:

- `band_count`
- `time_count`
- `chunk_width`
- `chunk_height`
- `date` on per-unit events
- `time_index` on per-unit events
- `estimated_cog_reads` on start when `explain()` was used and that value is cheaply available

Do not include highly unstable implementation details or per-band payloads in v1.

## Integration Points

### `src/lazycogs/_explain.py`

- Add `ExplainPlan.total_progress_units`.
- Optionally update `summary()` and `repr()` to mention progress units separately from band-expanded chunk reads.
- Keep `total_chunk_reads` unchanged for explain/reporting compatibility.

### `src/lazycogs/_backend.py`

- Add an optional internal progress-session field to `MultiBandStacBackendArray`.
- Move runtime advancement to the same granularity as `ExplainPlan.total_progress_units`.
- Emit `advance` after each `_run_one_date(...)` completion, not after the whole `_async_getitem()` call.
- Emit `error` before re-raising if any time-step read fails.
- Emit `finish` once the bound request completes.

### `src/lazycogs/_progress.py`

Owns:
- `ProgressEvent`
- `ProgressCallback`
- session lifecycle helpers
- `bind_progress(...)`
- optional `tqdm_callback(...)`

### `src/lazycogs/_explain.py` accessor

Add `StacCogAccessor.progress(...)` as thin sugar over top-level `bind_progress(...)`.

### `src/lazycogs/__init__.py`

Export only intentionally public progress surface.

## Runtime event semantics

### Start

Emit exactly once when the binding context is entered and the total is known.

Requirements:
- `completed == 0`
- `total == supplied total` or `ExplainPlan.total_progress_units`
- metadata includes aggregate request shape

### Advance

Emit once per completed `(time step, spatial tile)` unit.

Requirements:
- `completed` increments by one each time
- event order is monotonic by `completed`, even if underlying async tasks finish out of order
- per-event metadata may include the finished `time_index` and `date`

### Error

Emit at most once per request, immediately before the exception escapes the lazycogs read path.

Requirements:
- `completed` reflects the last successful unit
- original exception is preserved and re-raised

### Finish

Emit exactly once after all units complete successfully.

Requirements:
- `completed == total` when total is known
- no extra advance events after finish

## Migration Path

This is a backwards-compatible additive change.

1. Extend explain with `total_progress_units`.
2. Add the internal progress session and public callback types.
3. Add the binding API.
4. Instrument runtime advancement at per-unit granularity.
5. Add optional tqdm helper and docs.

No existing public signatures need to change.

## Testing Strategy

### Unit tests

- `ExplainPlan.total_progress_units` matches the number of `(time, tile)` queries for unchunked and chunked arrays.
- `bind_progress(..., total="auto")` calls `explain()` once and emits a `start` event with the derived total.
- explicit `total=<int>` skips `explain()`.
- nested bindings on the same backend raise a clear error.
- `tqdm_callback(...)` updates a fake bar correctly without importing `tqdm`.

### Backend tests

- unchunked multi-time read emits one `advance` per time step, not one for the whole `_async_getitem()` call.
- chunked dask-backed read emits one `advance` per scheduled `(time, spatial tile)` unit.
- empty-result time steps still advance because the unit completed, even though zero COGs matched.
- error path emits `error` and re-raises the original exception.
- concurrent completions do not corrupt `completed`.

### Integration tests

- `with da.lazycogs.progress(callback=...)` works with `da.load()`.
- `with da.lazycogs.progress(callback=...)` works with `await da.load_async()`.
- docs example using a fake tqdm-like bar produces the expected total and final count.

## Decision Log

| Decision | Options Considered | Rationale |
|----------|--------------------|-----------|
| Use callback events as the core public primitive | `tqdm`-specific API; observer class only; event bus | Smallest interoperable contract across renderers and external tools |
| Reuse `da.lazycogs.explain()` for totals | New planner API; runtime-only unknown totals | Reuses existing public API and avoids duplicate planning logic |
| Define progress in `(time step, spatial tile)` units | Per-COG; per-band; per-`_async_getitem()` | Matches real execution and avoids band double-counting |
| Advance at `_run_one_date` granularity | Advance only at `_async_getitem`; deep `_chunk_reader.py` instrumentation | Fine enough for useful percent without invasive lower-level hooks |
| Keep `tqdm` as an adapter | Make `tqdm` the primary API | Preserves interoperability and optional dependencies |

## Open Questions

- Should `tqdm_callback(...)` ship in v1, or should docs show how to write the adapter inline and add the helper later?
- Should `ExplainPlan.summary()` surface `total_progress_units` immediately, or should that remain a property-only addition in v1?
- Do we want to expose a `request_id` in docs/examples, or keep it primarily for advanced integrations and tests?

## References

- `dev-docs/plans/2026-05-16-002-feat-progress-indicator-plan.md`
- `src/lazycogs/_backend.py`
- `src/lazycogs/_explain.py`
- `ARCHITECTURE.md`
