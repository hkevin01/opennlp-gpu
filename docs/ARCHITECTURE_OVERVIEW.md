# Architecture Overview

## High-Level Design
- Layered architecture: API → GPU Wrapper → Native Runtime
- Supports CUDA, ROCm, OpenCL
- CPU fallback for all operations

## Core Components
- GpuConfig: Central configuration
- ComputeProvider: Abstracts hardware
- GpuMatrixOperation: GPU matrix ops
- GpuMaxentModel, GpuPerceptronModel: Accelerated models
- GpuFeatureExtractor: Fast feature extraction

## Native Integration
- JNI bridges Java and C++/CUDA/HIP/OpenCL
- CMake-based native build

## Extensibility
- Modular design for new GPU platforms
- Easy to add new models or kernels 