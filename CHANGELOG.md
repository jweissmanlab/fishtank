# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

### Changed

### Fixes

## [0.3.7]

### Fixes

- Fix `IndexError` when reading ImageJ hyperstacks stored as a single physical TIFF page (e.g. contiguous multi-frame files from some microscope acquisitions). Falls back to `tif.asarray()` when `tif.pages` has fewer entries than the requested frame index.

## [0.1.1]

### Added

### Changed

-  Strategy is now optional for decode_spots

### Fixes

- aggregate_polygons now works when no FOVs overlap

## [0.1.0]

### Added

-   Support for Zhuang lab XML format
-   Switched to Cellpose>=4.0.0 to use SAM model

### Changed

-  "nuclei" and "cyto" models are no longer supported, use "cpsam" instead

### Fixes

## [0.0.1]

### Added

-   Basic tool, preprocessing and plotting functions
