# Changelog

## [0.3.1](https://github.com/developmentseed/lazycogs/compare/v0.3.0...v0.3.1) (2026-05-15)


### Bug Fixes

* eagerly load spatial coords to avoid mismatch in sel operations ([#57](https://github.com/developmentseed/lazycogs/issues/57)) ([1f66381](https://github.com/developmentseed/lazycogs/commit/1f66381582b13a73084fbe24f35709aecfc8ee3e))

## [0.3.0](https://github.com/developmentseed/lazycogs/compare/v0.2.0...v0.3.0) (2026-05-07)


### Features

* implement xarray async capability ([#46](https://github.com/developmentseed/lazycogs/issues/46)) ([e7a501c](https://github.com/developmentseed/lazycogs/commit/e7a501caeecee7f9cc2c0fb104fd7af2a03172ad))
* use rasterix to handle x/y dimensions ([#52](https://github.com/developmentseed/lazycogs/issues/52)) ([0f02cc2](https://github.com/developmentseed/lazycogs/commit/0f02cc24f43bc3c9028d9ce33c1f595ce4c9de94))

## [0.2.0](https://github.com/developmentseed/lazycogs/compare/v0.1.2...v0.2.0) (2026-05-04)


### Features

* add run_chunk_async and run_chunk for direct non-xarray access pattern ([#45](https://github.com/developmentseed/lazycogs/issues/45)) ([31e1952](https://github.com/developmentseed/lazycogs/commit/31e195206f2bb8490766373694afd8ee9e8cc7b6))


### Bug Fixes

* remove unnecessary lock on duckdb searches ([#39](https://github.com/developmentseed/lazycogs/issues/39)) ([dede83c](https://github.com/developmentseed/lazycogs/commit/dede83cff9594576e04e47a08e943ce9a3527dab))

## [0.1.2](https://github.com/developmentseed/lazycogs/compare/v0.1.1...v0.1.2) (2026-04-29)


### Bug Fixes

* rip out fake-async open_async ([#37](https://github.com/developmentseed/lazycogs/issues/37)) ([cf07318](https://github.com/developmentseed/lazycogs/commit/cf07318c0c2414bf47251de72fde9c2189263d31)), closes [#26](https://github.com/developmentseed/lazycogs/issues/26)

## [0.1.1](https://github.com/developmentseed/lazycogs/compare/v0.1.0...v0.1.1) (2026-04-27)


### Documentation

* update installation instructions ([48c0a82](https://github.com/developmentseed/lazycogs/commit/48c0a826aa38f1ab90154a0951176bbdf9578d48))
* update links in README ([d522ba6](https://github.com/developmentseed/lazycogs/commit/d522ba6ab70dcb588318868261f025719c3b487a))

## [0.1.0](https://github.com/developmentseed/lazycogs/compare/v0.0.1...v0.1.0) (2026-04-27)


### Features

* initial release ([4b5241d](https://github.com/developmentseed/lazycogs/commit/4b5241d24653a28791f07759a4660d7582f2330c))
