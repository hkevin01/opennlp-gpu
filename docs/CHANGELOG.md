# Changelog

## [Unreleased]
- Advanced TF-IDF and vectorization pipeline overhaul:
	- Consolidated TF-IDF computation into a single reusable core (`TfIdfAlgorithms`) used by CPU/OpenCL/CUDA/ROCm paths.
	- Added weighted n-gram blending (`NGramBlendOptions`) to combine unigram/bigram/trigram signals in one vectorizer run.
	- Added configurable weighting schemes: raw TF-IDF, sublinear TF-IDF, and BM25.
	- Added configurable IDF smoothing strategies (`STANDARD_SMOOTH`, `PROBABILISTIC_IDF`, `BM25_IDF`).
	- Added document-frequency cutoffs (`minDocumentFrequency`, `maxDocumentFrequency`) for robust vocabulary pruning.
	- Added class-balanced discriminative feature selection options:
		- Chi-square and information-gain selection paths.
		- Macro-averaging support.
		- Optional class-prior weighting to better represent minority classes.
- Reproducibility and persistence improvements:
	- Added persisted `VocabularyState` that captures vocabulary, DF statistics, blend weights, IDF smoothing, and DF cutoffs.
	- Added explicit compatibility policy for persisted state:
		- V2 current format.
		- V1 migration path with safe defaults.
		- Clear failure on unsupported future versions.
- Vector storage compression:
	- Added dense-vector persistence with optional quantization/compression:
		- `FLOAT32` (baseline)
		- `FLOAT16` (reduced storage, moderate precision loss)
		- `INT8` (high compression with per-row scale factors)
	- Added load-time dequantization support for all formats.
- Quality and benchmark guardrails:
	- Added cross-backend parity + latency guardrail assertions for TF-IDF benchmark tests.
	- Expanded TF-IDF unit coverage for smoothing, DF cutoffs, class balancing behavior, cache determinism, persistence roundtrip, quantization roundtrip, and numerical stability stress.

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
