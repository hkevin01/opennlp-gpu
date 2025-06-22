#!/bin/bash

# Quick Fix Script for OpenNLP GPU Extension Compilation Errors
# This script fixes the immediate C++ compilation issues identified

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "🔧 Fixing OpenNLP GPU Extension Compilation Issues..."

# Fix 1: Ensure clean build environment
echo "1. Cleaning build environment..."
cd "${SCRIPT_DIR}/src/main/cpp"
rm -rf CMakeCache.txt CMakeFiles/ Makefile build/ *.so *.dylib 2>/dev/null || true

# Fix 2: Verify the math library includes are present
echo "2. Verifying math library includes..."
if ! grep -q "#include <cmath>" jni/GpuOperationsJNI.cpp; then
    echo "   Adding missing math includes to GpuOperationsJNI.cpp..."
    sed -i '/#include <vector>/a #include <cmath>\n#include <algorithm>' jni/GpuOperationsJNI.cpp
fi

# Fix 3: Check CMakeLists.txt for math library linking
echo "3. Checking CMakeLists.txt for math library..."
if ! grep -q "target_link_libraries.*m" CMakeLists.txt; then
    echo "   Math library link may be missing - this should be fixed in the main file"
fi

# Fix 4: Set proper environment variables for compilation
echo "4. Setting compilation environment..."
export CXXFLAGS="-O2 -fPIC -D_GNU_SOURCE"
export LDFLAGS="-lm"

# Fix 5: Try a clean build with verbose output
echo "5. Attempting clean build..."
cmake . -DUSE_CPU_ONLY=ON -DCMAKE_VERBOSE_MAKEFILE=ON

echo "6. Building with make (verbose)..."
make VERBOSE=1

if [ -f "libopennlp_gpu.so" ] || [ -f "libopennlp_gpu.dylib" ]; then
    echo "✅ Build successful! Native library created."
    
    # Copy to resources if possible
    mkdir -p "${SCRIPT_DIR}/src/main/resources/native/linux/x86_64" 2>/dev/null || true
    cp libopennlp_gpu.* "${SCRIPT_DIR}/src/main/resources/native/linux/x86_64/" 2>/dev/null || \
    cp libopennlp_gpu.* "${SCRIPT_DIR}/src/main/resources/" 2>/dev/null || \
    echo "   Note: Could not copy to resources directory (may need manual copy)"
    
    echo "✅ Fix completed successfully!"
else
    echo "❌ Build still failing. Check the verbose output above for specific errors."
    echo "   Common issues:"
    echo "   - Missing development packages (build-essential, cmake, etc.)"
    echo "   - Incompatible compiler version"
    echo "   - Missing system headers"
    exit 1
fi
