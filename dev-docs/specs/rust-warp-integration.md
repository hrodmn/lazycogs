# Spec: Replacing lazycogs' numpy/pyproj Warp Engine with rust-warp

## Context

lazycogs currently reprojects source COG windows with a small in-repo engine in `src/lazycogs/_reproject.py`. That engine:

- uses `pyproj.Transformer` to map destination pixel centers into the source CRS
- converts projected coordinates to source pixel coordinates with the inverse affine
- floors to integer indices
- samples with numpy fancy indexing

This design is simple and fast enough for nearest-neighbor, but it has hard limits:

- only nearest-neighbor resampling exists
- the warp math and sampling code are maintained locally
- performance is constrained by Python/numpy memory traffic
- the implementation is tightly specialized to today's warp path

[rust-warp](https://github.com/jakenotjay/rust-warp) is a GDAL-free Rust/Python warp engine that already provides:

- inverse-mapping reprojection
- multiple resampling kernels
- GIL-free compute
- pure-Rust projection math for several common CRSes, with `proj4rs` fallback
- Python bindings that accept affine transforms and CRS strings

The question is not "can we call rust-warp somewhere". We obviously can. The question is whether it cleanly replaces lazycogs' current per-window reprojection path without breaking the parts lazycogs is already good at: per-chunk DuckDB search, async COG window reads, overview selection, and mosaic ordering.

This spec defines the architecture for that replacement.

## Goals

- Replace the internal warp implementation in `src/lazycogs/_reproject.py` with a rust-warp-backed adapter.
- Preserve lazycogs' existing compute-time pipeline: DuckDB search, async-geotiff window reads, overview selection, and mosaic methods.
- Enable multiple reprojection resampling methods through lazycogs' public API.
- Preserve the current fast path for same-grid reads where reprojection is unnecessary.
- Keep the migration incremental so behavior can be parity-tested against the current implementation before full cutover.

## Non-goals

- Replacing `rustac`, DuckDB search, or `async-geotiff`.
- Replacing the search-time `pyproj` usage that converts request bboxes to EPSG:4326. That is a separate concern.
- Adopting rust-warp's dask graph builder or xarray accessor. lazycogs already owns those layers.
- Re-architecting mosaic methods.
- Promising bit-for-bit parity with current `pyproj` output across every CRS.

## Constraints and Assumptions

- lazycogs remains a Python-first package with Rust dependencies only through installed wheels.
- The replacement must work with per-item window reads produced by `async-geotiff`; rust-warp does not become the source reader.
- lazycogs must continue to support chunk-local reprojection of small source windows, not only full-scene arrays.
- The current same-CRS same-affine fast path should remain in Python because it avoids any warp call at all.
- rust-warp's useful integration surface is its low-level array reprojection API, not its xarray or dask APIs.
- rust-warp currently documents 2D low-level warps. lazycogs must therefore continue to iterate over band planes in Python unless upstream adds a true 3D band-aware kernel.
- rust-warp supports a limited dtype set. lazycogs must validate or cast unsupported dtypes explicitly.

## Why the high-level rust-warp APIs are the wrong fit

lazycogs already has the hard parts that matter for its product shape:

- chunk-local STAC search
- overview selection before read
- selective source window reads
- per-item mosaicking in caller-defined order
- xarray backend integration

rust-warp's dask planner and xarray accessor solve a different problem: lazy reprojection of arrays you already have. lazycogs does not already have the full source array. It discovers and reads tiny windows on demand. Replacing lazycogs' chunk orchestration with rust-warp's planner would be a step backward.

Therefore the correct integration point is the low-level warp kernel:

- keep lazycogs' orchestration
- replace only the local reprojection core

## Architecture Overview

```text
open(..., resampling=...)
  -> MultiBandStacBackendArray
  -> _async_getitem(...)
  -> _read_chunk_all_dates(...)
  -> read_chunk_async(...)
  -> _read_item_band(...)
     -> GeoTIFF.open / overview select / window read
     -> rust-warp adapter
     -> mosaic method feed
```

### What stays in lazycogs

- `src/lazycogs/_backend.py`
- `src/lazycogs/_chunk_reader.py` item/window/overview logic
- mosaic methods in `src/lazycogs/_mosaic_methods.py`
- grid construction in `src/lazycogs/_grid.py`
- all chunk/time concurrency behavior

### What changes

- `src/lazycogs/_reproject.py` stops implementing warp math itself
- lazycogs adds a thin adapter around `rust_warp.reproject_array`
- lazycogs adds public resampling selection
- caching shifts from cached integer `WarpMap`s to a simpler geometry/argument cache only if benchmarking proves it worthwhile

## Proposed Integration Design

### 1. Introduce a backend-neutral reprojection interface

`src/lazycogs/_reproject.py` should stop exposing "warp map" as the primary abstraction. That is an implementation detail of the current engine, and it leaks too much.

Instead define a narrow operation-level interface:

```python
@dataclass(frozen=True)
class ReprojectRequest:
    data: np.ndarray            # shape (bands, src_h, src_w)
    src_transform: Affine
    src_crs: CRS
    dst_transform: Affine
    dst_crs: CRS
    dst_width: int
    dst_height: int
    nodata: float | None
    resampling: str


def reproject_tile(request: ReprojectRequest) -> np.ndarray:
    ...
```

Initial implementation paths:

- `same-grid fast path`: return input unchanged
- `python-legacy engine`: existing implementation, temporarily retained for rollout
- `rust-warp engine`: new default path once validated

This lets us swap implementations without rewriting `_chunk_reader.py` again.

### 2. Implement a rust-warp adapter, not a direct scatter of API calls

Add a new private module, e.g. `src/lazycogs/_rust_warp.py`, responsible for:

- converting `Affine` to the 6-tuple rust-warp expects
- converting `pyproj.CRS` objects to CRS strings accepted by rust-warp
- handling per-band iteration for `(bands, h, w)` arrays
- validating dtypes and nodata compatibility
- mapping lazycogs resampling names to rust-warp resampling names

Proposed function:

```python
def reproject_array_rust_warp(
    data: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    dst_transform: Affine,
    dst_crs: CRS,
    dst_width: int,
    dst_height: int,
    nodata: float | None = None,
    resampling: str = "nearest",
) -> np.ndarray:
    ...
```

Implementation sketch:

1. Fast-return if source and destination grid already match.
2. Normalize CRS strings.
3. For each band plane:
   - call `rust_warp.reproject_array(...)`
4. Stack band outputs back to `(bands, dst_h, dst_w)`.

This is the right first integration even if it feels slightly boring. It replaces the risky part and leaves working infrastructure alone.

### 3. Stop centering the design on warp-map caching

The current implementation caches `WarpMap` objects because computing destination-to-source coordinate maps in Python is expensive and repeated. Once the warp loop moves into Rust, that specific cache may become unnecessary or even counterproductive.

Decision:

- remove `WarpMap` from the long-term design
- keep the `warp_cache` plumbing during migration only if needed for A/B comparison
- benchmark before reintroducing any geometry-plan cache

Rationale:

- caching tied to old internals makes the new design harder to reason about
- rust-warp already has its own optimization strategy
- a bad cache can waste memory across concurrent chunk reads

If post-migration benchmarks show repeated geometry setup is still expensive, reintroduce a new cache based on a backend-neutral key such as `(src_transform, src_crs, dst_transform, dst_crs, dst_shape, resampling)`.

### 4. Add public resampling selection now that the backend can support it

`lazycogs.open()` should gain:

```python
resampling: Literal["nearest", "bilinear", "cubic", "lanczos", "average"] = "nearest"
```

This parameter must flow through:

- `_core.open()`
- `MultiBandStacBackendArray`
- `_async_getitem()`
- `read_chunk_async()` / `read_chunk()`
- `_read_item_band()`
- `reproject_tile()`

Notes:

- `average` should be documented as downsampling-oriented.
- `nearest` remains the default for backward compatibility.
- If a later decision is made to expose only a smaller supported subset, the interface can still start with the rust-warp names and reject unsupported ones centrally.

### 5. Keep search-time bbox reprojection on pyproj for now

Do not try to delete `pyproj` entirely in this change.

Current uses of `pyproj` fall into two buckets:

1. search/grid plumbing
   - bbox to EPSG:4326 for STAC queries
   - output grid and metadata helpers
2. per-pixel warp math
   - current `_reproject.py`

This spec replaces only bucket 2.

Reasons:

- rust-warp is solving raster reprojection, not all CRS concerns in lazycogs
- bbox transforms are not a performance bottleneck worth destabilizing
- widening scope makes it harder to debug accuracy differences

After migration, `pyproj` may still remain a dependency. That is acceptable.

## API or Interface Design

### Public API

```python
def open(
    href: str,
    *,
    ...,
    resampling: str = "nearest",
    ...,
) -> xr.DataArray:
    ...
```

Validation rules:

- accepted values initially: `nearest`, `bilinear`, `cubic`, `lanczos`, `average`
- unknown values raise `ValueError`
- docs must state that quality and performance differ by method

### Internal API

```python
@dataclass(frozen=True)
class ReprojectRequest:
    data: np.ndarray
    src_transform: Affine
    src_crs: CRS
    dst_transform: Affine
    dst_crs: CRS
    dst_width: int
    dst_height: int
    nodata: float | None
    resampling: str


def reproject_tile(request: ReprojectRequest) -> np.ndarray:
    """Reproject one `(bands, y, x)` source tile onto the destination chunk grid."""
```

### Optional migration hook

During rollout only:

```python
reproject_engine: Literal["legacy", "rust-warp"] = "rust-warp"
```

This should be private or test-only, not a documented permanent user-facing API.

## Data Model

No external data model changes.

Internal changes:

- `WarpMap` becomes deprecated and then removable.
- `MultiBandStacBackendArray` gains a `resampling: str` field.
- `_ChunkContext` gains `resampling: str`.

## Detailed Behavior

### Same-grid fast path

If all of the following are true:

- `src_crs.equals(dst_crs)`
- `raster.transform == dst_transform`
- source width/height match destination width/height

then lazycogs must return the source data unchanged without calling rust-warp.

This preserves the existing zero-overhead case.

### Band handling

rust-warp's documented low-level API is 2D. lazycogs reads `(bands, h, w)` windows. Therefore the adapter must:

- iterate over band axis in Python
- preserve input band order
- preserve output shape `(bands, dst_h, dst_w)`

Future optimization:

- if upstream rust-warp adds multi-band low-level kernels, lazycogs can switch internally without changing public API

### CRS normalization

lazycogs currently carries `pyproj.CRS` objects. rust-warp wants EPSG or PROJ strings.

Adapter rules:

1. Prefer `CRS.to_epsg()` when available.
2. If EPSG exists, pass `f"EPSG:{epsg}"`.
3. Otherwise pass `CRS.to_proj4()`.
4. If neither produces a usable value, raise a clear error.

Do not pass WKT unless benchmarking and compatibility work proves it is necessary. rust-warp's own docs say WKT handling depends on `pyproj` assistance and is not its strongest path.

### Dtype handling

rust-warp README says low-level support includes:

- `float32`
- `float64`
- `int8`
- `uint8`
- `uint16`
- `int16`

Before calling rust-warp, lazycogs must:

- allow these dtypes directly
- explicitly reject or cast unsupported dtypes

Recommended initial policy:

- direct pass-through for supported dtypes
- raise `TypeError` for unsupported dtypes during migration

Reason: silent casting is how you ship a bug and only discover it six weeks later in someone's science pipeline.

A later follow-up can add an explicit casting policy if needed.

### Nodata behavior

lazycogs currently fills out-of-bounds pixels with `nodata` or zero.

Required behavior with rust-warp:

- preserve current caller-facing semantics
- pass explicit `nodata` whenever known
- keep lazycogs' existing `effective_nodata` logic per band asset

Open detail:

- verify how rust-warp treats integer arrays when `nodata=None`
- verify whether NaN propagation for float arrays matches lazycogs expectations

This must be locked down in tests before cutover.

## Integration Points

### `src/lazycogs/_core.py`

- add `resampling` parameter to `open()`
- store on backend object
- validate once at open time

### `src/lazycogs/_backend.py`

- propagate `resampling` through chunk reads
- no change to indexing model

### `src/lazycogs/_chunk_reader.py`

Current `_apply_bands_with_warp_cache()` should be replaced or renamed to reflect the new responsibility. Suggested shape:

```python
def _reproject_band_rasters(
    band_rasters: list[tuple[str, RasterArray, CRS, float | None]],
    dst_transform: Affine,
    dst_crs: CRS,
    dst_width: int,
    dst_height: int,
    resampling: str,
) -> dict[str, tuple[np.ndarray, float | None]]:
    ...
```

Responsibilities:

- preserve same-grid fast path
- call `reproject_tile()` for actual warps
- stop exposing backend-specific cache mechanics to callers

### `src/lazycogs/_reproject.py`

Migration plan:

- stage 1: keep legacy implementation under renamed helpers
- stage 2: add backend-neutral `reproject_tile()` dispatcher
- stage 3: make rust-warp the default implementation
- stage 4: delete legacy warp-map code if benchmarks and parity are acceptable

### Dependencies

Add `rust-warp` as a dependency if licensing, wheel availability, and platform coverage are acceptable.

Before adding it, verify:

- Python versions supported by lazycogs
- Linux/macOS wheel availability for CI and target users
- whether `maturin`-built wheels include everything needed for downstream installs

This is a release engineering issue, not just a code issue.

## Migration Path

### Phase 1: Adapter and hidden A/B mode

- Add dependency.
- Add `reproject_tile()` abstraction.
- Keep legacy engine in place.
- Add rust-warp adapter behind an internal switch.
- Add parity tests that run both engines on the same cases.

Exit criteria:

- supported dtypes work
- same-grid fast path preserved
- nearest parity acceptable on representative CRS pairs

### Phase 2: Public resampling API

- Add `resampling=` to `open()`.
- Route `nearest` through rust-warp too in test environments.
- Add tests for `bilinear`, `cubic`, `lanczos`, `average` where appropriate.
- Benchmark representative workloads.

Exit criteria:

- API stable
- docs updated
- no obvious regressions in common nearest-neighbor path

### Phase 3: Default cutover

- Make rust-warp the default backend.
- Retain legacy engine only as short-lived fallback if needed.

Exit criteria:

- CI green
- benchmark deltas understood
- accuracy deltas documented

### Phase 4: Cleanup

- Remove `WarpMap`, `compute_warp_map`, `apply_warp_map`, and related cache plumbing if no longer needed.
- Simplify docs and architecture notes.

## Testing Strategy

### Unit tests

Add tests for:

- affine tuple conversion
- CRS normalization to EPSG/PROJ strings
- supported dtype pass-through
- unsupported dtype rejection
- same-grid fast path bypassing rust-warp
- output shape and dtype preservation

### Parity tests vs legacy engine

For `nearest` only, compare legacy and rust-warp on:

- identical-grid reads
- partial overlap
- out-of-bounds nodata fill
- common CRS pairs in docs/tests
- multi-band windows

Comparison rule:

- exact equality for identity and same-CRS simple cases
- exact or near-exact equality for common reprojection cases, depending on measured behavior

### Parity tests vs rasterio/GDAL reference

Keep and extend the existing raster parity suite to compare:

- `nearest`
- `bilinear`
- `cubic`
- maybe `average` for downsampling cases

This matters because switching away from `pyproj` changes the projection engine, not just the resampler.

### Integration tests

Exercise full lazycogs chunk reads with:

- multiple overlapping items
- same-CRS overview reads
- cross-CRS reads
- preserved mosaic ordering with `FirstMethod`

### Benchmark tests

Benchmark at least these scenarios:

1. same-grid fast path
2. nearest reprojection, same dtype as current tests
3. bilinear reprojection on continuous data
4. many small chunk reads across dates
5. overview-backed read vs full-resolution read

The benchmark goal is not "rust is always faster". The goal is to confirm the whole lazycogs pipeline gets better or at least does not regress in the dominant workloads.

## Decision Log

| Decision | Options Considered | Rationale |
|----------|--------------------|-----------|
| Integrate at low-level warp API only | Use rust-warp xarray/dask APIs; replace only `_reproject.py` | lazycogs already owns orchestration and selective reads; replacing more would duplicate working logic |
| Keep `pyproj` for search-time CRS work | Remove `pyproj` entirely; replace only per-pixel warp math | search-time transforms are not the problem and do not justify widening scope |
| Add backend-neutral `reproject_tile()` | Keep `compute_warp_map` API and swap internals | current API is overfit to the old implementation |
| Iterate over band planes in Python | Wait for upstream multi-band support; try to coerce 3D input now | documented low-level API is 2D; a thin Python loop is the safest integration |
| Validate dtypes explicitly | Implicit casts; best-effort support | silent casts are risky for scientific output |
| Retain same-grid fast path | Always call rust-warp | avoiding unnecessary warp calls preserves a proven optimization |

## Risks

### 1. CRS accuracy differences

rust-warp uses native Rust projections for common CRSes and `proj4rs` fallback elsewhere. That means output may differ slightly from current `pyproj`-backed behavior.

Mitigation:

- document expected differences
- benchmark and parity-test common CRS pairs used by lazycogs
- prefer EPSG normalization so common CRSes route to rust-warp's native implementations

### 2. Packaging risk

A compiled extension dependency can break installation in environments where pure Python currently works.

Mitigation:

- verify wheel availability before merge
- test install in CI on supported platforms
- do not remove legacy path until packaging is proven

### 3. Small-window overhead

lazycogs often reprojects many small windows, not one giant array. If rust-warp has high fixed call overhead, theoretical wins may not show up in real workloads.

Mitigation:

- benchmark realistic lazycogs chunk shapes, not only synthetic full-image cases

### 4. Nodata/resampling semantics drift

Different kernels and nodata rules can subtly change mosaic outcomes.

Mitigation:

- lock behavior down with integration tests
- document that `average` is for downsampling, not categorical data

## Open Questions

- Does rust-warp's per-call overhead remain favorable for the small window sizes lazycogs reads most often?
- Which lazycogs-supported datasets use dtypes outside rust-warp's current low-level support matrix?
- Are there important lazycogs CRS workflows that would hit `proj4rs` fallback instead of rust-warp native implementations?
- Do we want a temporary environment-variable fallback to the legacy engine during rollout?
- Should `average` ship in the initial public API, or only after dedicated downsampling tests?

## Recommended Implementation Order

1. Add adapter module and backend-neutral `reproject_tile()`.
2. Thread `resampling` through internal call sites.
3. Add hidden rust-warp path and parity tests.
4. Benchmark realistic workloads.
5. Expose public resampling API.
6. Cut over default backend.
7. Delete legacy warp-map code.

## References

- `src/lazycogs/_reproject.py`
- `src/lazycogs/_chunk_reader.py`
- `tests/test_reproject.py`
- `README.md`
- `ARCHITECTURE.md`
- https://github.com/jakenotjay/rust-warp
- https://github.com/jakenotjay/rust-warp/blob/main/docs/architecture.md
- https://github.com/jakenotjay/rust-warp/blob/main/docs/proj4rs-differences.md
