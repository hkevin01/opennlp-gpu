#!/bin/bash
set -e

echo "🔥 Testing GPU Kernel Optimization"
echo "================================="

# Make the script executable
chmod +x "$0"

# Compile the project
echo "📦 Compiling project..."
mvn clean compile test-compile -q

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful"
else
    echo "❌ Compilation failed"
    exit 1
fi

# Run kernel performance tests
echo "🧪 Running kernel performance tests..."
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.performance.KernelPerformanceTest" -q

# Verify GPU configuration
echo "🔍 Verifying GPU configuration..."
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics" -q

echo "✅ Kernel optimization testing completed!"