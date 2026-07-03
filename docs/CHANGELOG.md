# Changelog

## [Unreleased]

### Added
- Unified TF-IDF/vectorization capabilities:
  - Added weighted n-gram blending (`NGramBlendOptions`) to combine unigram/bigram/trigram features in one run.
  - Added weighting schemes: raw TF-IDF, sublinear TF-IDF, and BM25.
  - Added IDF smoothing options: `STANDARD_SMOOTH`, `PROBABILISTIC_IDF`, `BM25_IDF`.
  - Added document-frequency cutoffs: `minDocumentFrequency` and `maxDocumentFrequency`.
  - Added class-balanced discriminative feature selection options:
    - Chi-square and information-gain selection.
    - Macro-averaging support.
    - Optional class-prior weighting for minority-class retention.
- Reproducibility and persistence:
  - Added persisted `VocabularyState` capturing vocabulary, DF stats, blend weights, smoothing strategy, and DF cutoffs.
  - Added explicit persisted-state compatibility policy (V2 current + V1 migration defaults).
- Vector storage optimization:
  - Added dense-vector persistence formats: `FLOAT32`, `FLOAT16`, and `INT8`.
  - Added load-time reconstruction/dequantization support for all persisted formats.
- Quality guardrails:
  - Added cross-backend TF-IDF parity and latency guardrail assertions in benchmark-oriented tests.

### Changed
- Consolidated TF-IDF computation into a single shared `TfIdfAlgorithms` core used by CPU/OpenCL/CUDA/ROCm paths to reduce backend drift.
- Updated roadmap tracking in docs to explicitly mark delivered feature-engineering hardening milestones as complete.

### Fixed
- Hardened Mermaid diagram labels/content for better GitHub renderer compatibility and fewer rich-display parse failures.
- Improved deterministic test safety around vectorization/caching behavior through expanded edge-case coverage.

### Docs
- Expanded `README.md` with explicit "what changed / why / benefits" coverage for new vectorization features.
- Added richer operator guidance: workload decision matrix, backend trade-off tables, rollout checklist, and clearer CPU-vs-GPU parallelization explanations.
- Added deeper sections on architecture internals, algorithm/formula rationale, and collapsible API reference blocks.
- Added additional visuals and references to improve onboarding and release communication.

## [2.0.0] - 2025-06-26
- Upgrade to OpenNLP 2.5.4
- Refactored for new OpenNLP APIs
- Native build system improvements (CMake, ROCm, CUDA)
- Test suite refactor and stabilization
- Documentation updates

## [1.0.0] - Initial Release
- Initial GPU acceleration for OpenNLP
- CUDA support
- Basic test suite
