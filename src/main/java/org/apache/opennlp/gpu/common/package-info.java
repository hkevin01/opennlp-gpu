/**

 * Requirement: Package org.apache.opennlp.gpu.common must provide all shared
 *              interfaces, configuration, logging, and memory management types
 *              required by every GPU and CPU compute provider.
 * Purpose: Foundation package for the GPU acceleration framework. Contains:
 *          ComputeProvider (compute backend contract), GpuConfig (configuration),
 *          MatrixOperation (common math interface), MemoryPool, MemoryManager,
 *          ResourceManager, NativeLibraryLoader, GpuLogger, and provider stubs
 *          (CpuComputeProvider, CudaComputeProvider, OpenClComputeProvider, RocmComputeProvider).
 * Rationale: Placing foundation types in a dedicated package avoids circular imports
 *            between the compute, ml, monitoring, and integration sub-packages.
 * Assumptions: All public types in this package are part of the stable public API.
 * Constraints: No dependency on compute.*, ml.*, monitoring.*, or integration.*.
 * Failure Modes: N/A — package descriptor; no executable code.
 * References: ARCHITECTURE_OVERVIEW.md; Apache OpenNLP 2.5.8 API contract.
 */
package org.apache.opennlp.gpu.common;
