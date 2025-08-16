#!/bin/bash

# Gecho "🚀 OpenNLP GPU Extension - Realistic Performance Demo"
echo "===================================================="
echo "📊 Demonstrating practical 2-5x performance improvements"
echo "for production NLP workflows"
echo
echo "1. System Diagnostics:"
java -cp "${CLASSPATH}" \
     -Djava.library.path="${PROJECT_ROOT}/src/main/cpp/build" \
     org.apache.opennlp.gpu.tools.GpuDiagnostics
echo
echo "2. Realistic Benchmarks:"
echo "   📈 Batch Document Classification (10K documents)"
echo "   📈 Large-Scale Feature Extraction (sparse matrices)"
echo "   📈 High-Throughput Named Entity Recognition"
echo "   📈 Concurrent Multi-Model Processing"
echo
java -cp "${CLASSPATH}" \
     -Djava.library.path="${PROJECT_ROOT}/src/main/cpp/build" \
     org.apache.opennlp.gpu.ml.RealisticBenchmarkDemo
echo
echo "3. Industry-Specific Scenarios:"
echo "   🏥 Healthcare: Clinical text processing"
echo "   💰 Finance: Risk assessment & compliance"
echo "   ⚖️  Legal: Document review & discovery"
echo "   🛒 E-Commerce: Customer support automation"
echo
java -cp "${CLASSPATH}" \
     -Djava.library.path="${PROJECT_ROOT}/src/main/cpp/build" \
     org.apache.opennlp.gpu.examples.IndustrySpecificDemo
echo
echo "4. Batch Processing & Streaming:"
echo "   📦 High-volume batch processing"
echo "   🌊 Real-time streaming pipelines"
echo "   🔀 Concurrent multi-model execution"
echo
java -cp "${CLASSPATH}" \
     -Djava.library.path="${PROJECT_ROOT}/src/main/cpp/build" \
     org.apache.opennlp.gpu.examples.BatchProcessingDemo
echo
echo "5. Cost-Efficiency Analysis:"
echo "   💰 Cloud instance cost comparison"
echo "   ⚡ Energy consumption analysis"
echo "   📊 Memory efficiency metrics"
echo
java -cp "${CLASSPATH}" \
     -Djava.library.path="${PROJECT_ROOT}/src/main/cpp/build" \
     org.apache.opennlp.gpu.analysis.CostEfficiencyDemoct root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set up classpath with Maven to ensure all dependencies
cd "${PROJECT_ROOT}"
CLASSPATH=$(mvn -q exec:exec -Dexec.executable=echo -Dexec.args="%classpath")

# Ensure dependencies are available
if [ ! -d "${PROJECT_ROOT}/target/dependency" ]; then
    echo "� Setting up dependencies..."
    cd "${PROJECT_ROOT}" && mvn dependency:copy-dependencies
fi

echo "�🚀 Running OpenNLP GPU Extension Demo"
echo "======================================"
echo
echo "1. GPU Diagnostics:"
java -cp "${CLASSPATH}" \
     -Djava.library.path="${PROJECT_ROOT}/src/main/cpp/build" \
     org.apache.opennlp.gpu.tools.GpuDiagnostics
echo
echo "2. GPU ML Demo:"
java -cp "${CLASSPATH}" \
     -Djava.library.path="${PROJECT_ROOT}/src/main/cpp/build" \
     org.apache.opennlp.gpu.ml.GpuMlDemo
