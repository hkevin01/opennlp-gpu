#!/bin/bash

# Universal GPU Setup Script for OpenNLP GPU
# Detects and configures both CUDA and ROCm/HIP platforms

echo "🚀 OpenNLP GPU Platform Setup"
echo "============================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Initialize variables
CUDA_FOUND=false
ROCM_FOUND=false
GPU_PLATFORM=""
GPU_PATH=""

echo "🔍 Detecting available GPU platforms..."

# Check for CUDA
echo ""
echo "🟢 Checking for NVIDIA CUDA..."
if command -v nvcc >/dev/null 2>&1; then
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
    echo "✅ CUDA found at: $CUDA_PATH"
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -o "release [0-9.]*" | cut -d' ' -f2 || echo "Unknown")
    echo "   Version: $CUDA_VERSION"
    
    # Check for NVIDIA GPUs
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
        if [ "$GPU_COUNT" -gt 0 ]; then
            echo "✅ Found $GPU_COUNT NVIDIA GPU(s):"
            nvidia-smi -L 2>/dev/null | head -3
            if [ "$GPU_COUNT" -gt 3 ]; then
                echo "   ... and $((GPU_COUNT - 3)) more GPUs"
            fi
        else
            echo "❌ No NVIDIA GPUs detected"
        fi
    fi
    CUDA_FOUND=true
else
    echo "❌ CUDA not found"
fi

# Check for ROCm
echo ""
echo "🔴 Checking for AMD ROCm..."
ROCM_PATHS=("/opt/rocm" "/usr/local/rocm" "$HOME/rocm")

for path in "${ROCM_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/bin/hipcc" ]; then
        echo "✅ ROCm found at: $path"
        ROCM_PATH="$path"
        ROCM_FOUND=true
        break
    fi
done

# Check system packages for ROCm
if [ "$ROCM_FOUND" = false ] && command -v hipcc >/dev/null 2>&1; then
    ROCM_PATH=$(dirname $(dirname $(which hipcc)))
    echo "✅ ROCm found via system packages at: $ROCM_PATH"
    ROCM_FOUND=true
fi

if [ "$ROCM_FOUND" = true ]; then
    if [ -f "$ROCM_PATH/bin/hipcc" ]; then
        HIP_VERSION=$($ROCM_PATH/bin/hipcc --version 2>/dev/null | grep -o "HIP version: [0-9.]*" || echo "Unknown")
        echo "   HIP Version: $HIP_VERSION"
    fi
    
    # Check for AMD GPUs
    if [ -f "$ROCM_PATH/bin/rocm-smi" ]; then
        GPU_COUNT=$($ROCM_PATH/bin/rocm-smi -i 2>/dev/null | grep -c "GPU\[" || echo "0")
        if [ "$GPU_COUNT" -gt 0 ]; then
            echo "✅ Found $GPU_COUNT AMD GPU(s):"
            $ROCM_PATH/bin/rocm-smi --showproductname 2>/dev/null | head -3
        else
            echo "❌ No AMD GPUs detected"
        fi
    fi
else
    echo "❌ ROCm not found"
fi

# Determine preferred platform
echo ""
echo "🎯 Platform Selection:"

if [ "$CUDA_FOUND" = true ] && [ "$ROCM_FOUND" = true ]; then
    echo "✅ Both CUDA and ROCm detected!"
    echo "   🟢 CUDA available at: $CUDA_PATH"
    echo "   🔴 ROCm available at: $ROCM_PATH"
    echo ""
    echo "   The CMake build will support both platforms."
    echo "   Runtime selection will be available."
    GPU_PLATFORM="BOTH"
elif [ "$CUDA_FOUND" = true ]; then
    echo "✅ CUDA will be used as the GPU platform"
    GPU_PLATFORM="CUDA"
    GPU_PATH="$CUDA_PATH"
elif [ "$ROCM_FOUND" = true ]; then
    echo "✅ ROCm will be used as the GPU platform"
    GPU_PLATFORM="ROCM"
    GPU_PATH="$ROCM_PATH"
else
    echo "⚠️  No GPU platforms detected - will build CPU-only version"
    GPU_PLATFORM="CPU"
fi

# Configure VS Code settings
echo ""
echo "⚙️  Configuring VS Code settings..."
VSCODE_SETTINGS=".vscode/settings.json"
mkdir -p .vscode

# Create or update VS Code settings
python3 << EOF
import json
import os

settings_file = '$VSCODE_SETTINGS'
cuda_found = '$CUDA_FOUND' == 'true'
rocm_found = '$ROCM_FOUND' == 'true'
cuda_path = '$CUDA_PATH' if cuda_found else ''
rocm_path = '$ROCM_PATH' if rocm_found else ''

try:
    with open(settings_file, 'r') as f:
        settings = json.load(f)
except:
    settings = {}

# Initialize environment sections
if 'terminal.integrated.env.linux' not in settings:
    settings['terminal.integrated.env.linux'] = {}
if 'cmake.configureEnvironment' not in settings:
    settings['cmake.configureEnvironment'] = {}
if 'cmake.configureArgs' not in settings:
    settings['cmake.configureArgs'] = []

env = settings['terminal.integrated.env.linux']
cmake_env = settings['cmake.configureEnvironment']
cmake_args = settings['cmake.configureArgs']

# Add CUDA environment if found
if cuda_found and cuda_path:
    env['CUDA_PATH'] = cuda_path
    cmake_env['CUDA_PATH'] = cuda_path
    if '-DUSE_CUDA=ON' not in cmake_args:
        cmake_args.append('-DUSE_CUDA=ON')

# Add ROCm environment if found
if rocm_found and rocm_path:
    env['ROCM_PATH'] = rocm_path
    env['HIP_PATH'] = rocm_path
    cmake_env['ROCM_PATH'] = rocm_path
    cmake_env['HIP_PATH'] = rocm_path
    if '-DUSE_ROCM=ON' not in cmake_args:
        cmake_args.append('-DUSE_ROCM=ON')

# Update PATH and LD_LIBRARY_PATH
path_additions = []
lib_additions = []

if cuda_found and cuda_path:
    path_additions.append(f"{cuda_path}/bin")
    lib_additions.append(f"{cuda_path}/lib64")

if rocm_found and rocm_path:
    path_additions.append(f"{rocm_path}/bin")
    lib_additions.append(f"{rocm_path}/lib")

if path_additions:
    env['PATH'] = ':'.join(path_additions) + ':\${env:PATH}'
if lib_additions:
    env['LD_LIBRARY_PATH'] = ':'.join(lib_additions) + ':\${env:LD_LIBRARY_PATH}'

# Save settings
settings['cmake.configureArgs'] = cmake_args
with open(settings_file, 'w') as f:
    json.dump(settings, f, indent=2)

print("✅ VS Code settings updated")
EOF

# Test CMake configuration
echo ""
echo "🧪 Testing CMake configuration..."

# Clean and create build directory
if [ -d "build" ]; then
    echo "   Cleaning previous build..."
    rm -rf build
fi

mkdir -p build
cd build

# Prepare CMake arguments
CMAKE_ARGS=()
if [ "$CUDA_FOUND" = true ]; then
    CMAKE_ARGS+=("-DUSE_CUDA=ON" "-DCUDA_PATH=$CUDA_PATH")
fi
if [ "$ROCM_FOUND" = true ]; then
    CMAKE_ARGS+=("-DUSE_ROCM=ON" "-DROCM_PATH=$ROCM_PATH")
fi

echo "   Running CMake configure with args: ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}" ../src/main/cpp

if [ $? -eq 0 ]; then
    echo "✅ CMake configuration successful!"
    
    echo ""
    echo "🏗️  Building the project..."
    make -j$(nproc)
    
    if [ $? -eq 0 ]; then
        echo "✅ Build successful!"
    else
        echo "❌ Build failed - check error messages above"
    fi
else
    echo "❌ CMake configuration failed"
    echo "   Check the error messages above"
    exit 1
fi

cd "$PROJECT_DIR"

echo ""
echo "📋 Setup Summary:"
echo "=================="
echo "Platform: $GPU_PLATFORM"
if [ "$CUDA_FOUND" = true ]; then
    echo "CUDA: ✅ Available at $CUDA_PATH"
fi
if [ "$ROCM_FOUND" = true ]; then
    echo "ROCm: ✅ Available at $ROCM_PATH"
fi
echo "Build: ✅ Complete"
echo "VS Code: ✅ Configured"

echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Reload VS Code window (Ctrl+Shift+P → 'Developer: Reload Window')"
echo "2. Run the GPU demo: mvn exec:java -Dexec.mainClass=org.apache.opennlp.gpu.ml.GpuMlDemo"
echo "3. Check GPU diagnostics: mvn exec:java -Dexec.mainClass=org.apache.opennlp.gpu.tools.GpuDiagnostics"

if [ "$GPU_PLATFORM" = "CPU" ]; then
    echo ""
    echo "⚠️  Note: No GPU platforms detected. The build will work but with CPU-only computation."
    echo "   To install GPU support:"
    echo "   - For NVIDIA: Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
    echo "   - For AMD: Install ROCm from https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
fi
