#!/bin/bash

# GPU Prerequisites Check Script for OpenNLP GPU Acceleration
# This script checks if your system has the necessary GPU drivers and SDKs

# Source cross-platform compatibility library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cross_platform_lib.sh"

echo "🔍 OpenNLP GPU Prerequisites Check"
echo "=================================="
echo

# Show system info first
OS=$(detect_os)
ARCH=$(detect_arch)
DISTRO=$(detect_distro)

echo "🖥️ System: $OS ($ARCH) - $DISTRO"
echo

GPU_FOUND=false
DRIVER_OK=false
RUNTIME_OK=false

# Check for NVIDIA GPU and drivers
echo "📊 Checking NVIDIA Setup..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
    GPU_FOUND=true
    DRIVER_OK=true
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        echo "✅ CUDA Toolkit: $(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')"
        RUNTIME_OK=true
    else
        echo "⚠️  CUDA Toolkit not found. Install with: sudo apt install nvidia-cuda-toolkit"
    fi
else
    echo "ℹ️  NVIDIA GPU not detected or drivers not installed"
fi

echo

# Check for AMD GPU and drivers  
echo "📊 Checking AMD Setup..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ AMD GPU with ROCm detected:"
    rocm-smi --showproductname | head -5
    GPU_FOUND=true
    DRIVER_OK=true
    RUNTIME_OK=true
elif lspci | grep -qi amd | grep -qi vga; then
    echo "⚠️  AMD GPU detected but ROCm not installed"
    echo "   Install with: curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -"
    echo "                sudo apt install rocm-dkms"
    GPU_FOUND=true
else
    echo "ℹ️  AMD GPU not detected"
fi

echo

# Check for Intel GPU
echo "📊 Checking Intel Setup..."
if command -v intel_gpu_top &> /dev/null; then
    echo "✅ Intel GPU tools detected"
    GPU_FOUND=true
    DRIVER_OK=true
    
    if dpkg -l | grep -q intel-opencl-icd; then
        echo "✅ Intel OpenCL runtime installed"
        RUNTIME_OK=true
    else
        echo "⚠️  Intel OpenCL runtime not found. Install with: sudo apt install intel-opencl-icd"
    fi
elif lspci | grep -qi intel | grep -qi vga; then
    echo "⚠️  Intel GPU detected but tools not installed"
    echo "   Install with: sudo apt install intel-gpu-tools intel-opencl-icd"
    GPU_FOUND=true
else
    echo "ℹ️  Intel GPU not detected"
fi

echo

# Check for Apple Silicon (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📊 Checking Apple Silicon Setup..."
    if system_profiler SPHardwareDataType | grep -q "Apple"; then
        echo "✅ Apple Silicon detected - Metal Performance Shaders available"
        GPU_FOUND=true
        DRIVER_OK=true
        RUNTIME_OK=true
    fi
    echo
fi

# Check OpenCL generic support
echo "📊 Checking OpenCL Support..."
if command -v clinfo &> /dev/null; then
    OPENCL_DEVICES=$(clinfo -l 2>/dev/null | grep -c "Platform")
    if [ "$OPENCL_DEVICES" -gt 0 ]; then
        echo "✅ OpenCL runtime detected with $OPENCL_DEVICES platform(s)"
        RUNTIME_OK=true
    else
        echo "⚠️  OpenCL not working properly"
    fi
else
    echo "ℹ️  OpenCL info tool not available. Install with: sudo apt install clinfo"
fi

echo

# Check Java environment
echo "📊 Checking Java Environment..."
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | grep version | cut -d'"' -f2 | cut -d'.' -f1)
    if [ "$JAVA_VERSION" -ge 11 ]; then
        echo "✅ Java $JAVA_VERSION - Compatible"
    else
        echo "⚠️  Java $JAVA_VERSION - Upgrade to Java 11+ recommended"
    fi
else
    echo "❌ Java not found. Install with: sudo apt install openjdk-17-jdk"
fi

echo

# Summary
echo "📋 Summary"
echo "=========="
if [ "$GPU_FOUND" = true ] && [ "$DRIVER_OK" = true ] && [ "$RUNTIME_OK" = true ]; then
    echo "🎉 GPU acceleration ready! Your system is properly configured."
    echo
    echo "Next steps:"
    echo "1. Clone the project: git clone https://github.com/yourusername/opennlp-gpu.git"
    echo "2. Build and test: mvn clean compile"
    echo "3. Run full diagnostics: mvn exec:java -Dexec.mainClass=org.apache.opennlp.gpu.tools.GpuDiagnostics"
    echo "4. Try the demos: mvn exec:java -Dexec.mainClass=org.apache.opennlp.gpu.demo.GpuDemoApplication"
elif [ "$GPU_FOUND" = true ]; then
    echo "⚠️  GPU hardware detected but drivers/runtime incomplete."
    echo "   See installation recommendations above."
    echo
    echo "✅ CPU fallback will work - you can still use the project!"
else
    echo "ℹ️  No GPU detected. Project will use optimized CPU implementations."
    echo
    echo "CPU-only mode still provides:"
    echo "• Optimized batch processing"
    echo "• Advanced neural network features"  
    echo "• Production monitoring tools"
fi

echo
echo "For detailed analysis, run the comprehensive diagnostics after building:"
echo "mvn exec:java -Dexec.mainClass=org.apache.opennlp.gpu.tools.GpuDiagnostics"
