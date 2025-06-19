# GPU Prerequisites Guide

## Overview

OpenNLP GPU Acceleration requires proper GPU drivers and runtime environments. This guide explains how to check and set up your system.

## Quick Checks

### 1. Instant Prerequisites Check (No Build Required)

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/opennlp-gpu/main/scripts/check_gpu_prerequisites.sh | bash
```

**What it checks:**
- GPU hardware detection (NVIDIA, AMD, Intel, Apple Silicon)
- Driver installation status
- Runtime environments (CUDA, ROCm, OpenCL)
- Java environment compatibility

### 2. Comprehensive Diagnostics (After Building)

```bash
git clone https://github.com/yourusername/opennlp-gpu.git
cd opennlp-gpu
mvn clean compile
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"
```

**Additional checks:**
- Detailed GPU specifications
- Memory analysis
- Performance baseline tests
- OpenNLP integration validation

### 3. VS Code Task (In IDE)

If you're using VS Code, run the GPU diagnostics task:
- `Ctrl+Shift+P` → "Tasks: Run Task" → "🔍 GPU Diagnostics Check"

## Supported GPU Platforms

### NVIDIA GPUs
**Requirements:**
- NVIDIA driver (535+ recommended)
- CUDA Toolkit (12.0+ recommended)

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit

# Verify
nvidia-smi
nvcc --version
```

### AMD GPUs  
**Requirements:**
- ROCm platform (5.0+ recommended)
- ROCm drivers

**Installation:**
```bash
# Ubuntu/Debian
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms

# Verify
rocm-smi
```

### Intel GPUs
**Requirements:**
- Intel GPU drivers
- Intel OpenCL runtime

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install intel-gpu-tools intel-opencl-icd

# Verify
intel_gpu_top
clinfo
```

### Apple Silicon (macOS)
**Requirements:**
- macOS with Apple Silicon (M1/M2/M3)
- Metal Performance Shaders (built-in)

**Verification:**
```bash
system_profiler SPHardwareDataType | grep "Chip"
```

## Troubleshooting

### Permission Issues
```bash
# Add user to GPU groups
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Logout and login to apply
```

### OpenCL Not Found
```bash
# Install OpenCL headers and runtime
sudo apt install ocl-icd-opencl-dev clinfo

# Test OpenCL
clinfo
```

### CUDA Issues
```bash
# Check CUDA installation
ls /usr/local/cuda/

# Add to PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## CPU Fallback

**No GPU? No Problem!**

If no compatible GPU is detected, OpenNLP GPU Acceleration automatically uses:
- Optimized CPU implementations
- SIMD vectorization
- Multi-threading
- Batch processing optimizations

You still get performance improvements even without a GPU!

## Performance Expectations

| GPU Type                              | Expected Speedup | Recommended For                        |
| ------------------------------------- | ---------------- | -------------------------------------- |
| **High-end** (RTX 4090, MI250)        | 15-50x           | Large-scale training, batch processing |
| **Mid-range** (RTX 3060, RX 6700)     | 5-15x            | Development, medium workloads          |
| **Entry-level** (GTX 1660, RX 580)    | 3-8x             | Testing, small models                  |
| **Integrated** (Intel Iris, Apple M1) | 2-5x             | Lightweight tasks                      |
| **CPU only**                          | 1-3x             | Optimized implementations              |

## Next Steps

1. **Run prerequisites check** using one of the methods above
2. **Install missing drivers/runtimes** if needed
3. **Build the project**: `mvn clean compile`
4. **Run demos**: See `docs/getting_started.md`
5. **Integrate into your project**: See main `README.md`
