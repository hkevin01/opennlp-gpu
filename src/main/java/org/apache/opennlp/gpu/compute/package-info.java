/**

 * Requirement: Package org.apache.opennlp.gpu.compute must provide concrete
 *              implementations of MatrixOperation and FeatureExtractionOperation
 *              for every supported backend: CPU, CUDA, OpenCL, ROCm, and cloud accelerators.
 * Purpose: Compute implementation layer. Contains:
 *          CpuMatrixOperation / CpuFeatureExtractionOperation (CPU reference),
 *          CudaMatrixOperation / CudaFeatureExtractionOperation (NVIDIA CUDA),
 *          OpenClMatrixOperation / OpenClFeatureExtractionOperation (cross-vendor OpenCL),
 *          RocmMatrixOperation / RocmFeatureExtractionOperation (AMD ROCm/HIP),
 *          GpuMatrixOperation (runtime-selection dispatcher), GpuMemoryManager,
 *          MatrixOperationConfig, OperationFactory, and cloud sub-package providers.
 * Rationale: Separating concrete implementations from the common interface package
 *            allows vendor-specific code to be compiled conditionally and loaded
 *            only when the corresponding native library is available.
 * Assumptions: Concrete classes instantiated via OperationFactory or ComputeProviderFactory;
 *              not directly constructed by callers outside unit tests.
 * Constraints: May depend on common.*; must not depend on ml.*, monitoring.*, or integration.*.
 * Failure Modes: N/A — package descriptor; no executable code.
 * References: ARCHITECTURE_OVERVIEW.md; gpu_config.h; OpenCL 1.2 spec; CUDA Toolkit docs.
 */
package org.apache.opennlp.gpu.compute;
