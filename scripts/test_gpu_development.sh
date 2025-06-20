#!/bin/bash
set -e

echo "🚀 Comprehensive GPU Development Test Suite"
echo "==========================================="

# Make the script executable
chmod +x "$0"

# Make sure we're in the right directory
cd "$(dirname "$0")/.."

# Compile everything first
echo "📦 Compiling project..."
mvn clean compile test-compile -q

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful"
else
    echo "❌ Compilation failed"
    exit 1
fi

echo ""
echo "🧪 Running GPU Development Tests..."
echo "-----------------------------------"

# Test 1: GPU Kernel Optimization
echo ""
echo "1️⃣ GPU Kernel Performance Test"
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.performance.KernelPerformanceTest" -q

# Test 2: Enhanced Performance Benchmarking
echo ""
echo "2️⃣ Enhanced Performance Benchmark"
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.benchmark.EnhancedPerformanceBenchmark" -q

# Test 3: OpenNLP Integration Test
echo ""
echo "3️⃣ OpenNLP Integration Test"
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.integration.OpenNlpIntegrationTest" -q

# Test 4: GPU Diagnostics
echo ""
echo "4️⃣ GPU Diagnostics Check"
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics" -q

# Test 5: Working Examples Verification
echo ""
echo "5️⃣ Examples Verification"
echo "Running sentiment analysis example..."
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.sentiment_analysis.GpuSentimentAnalysis" -Dexec.args="--test-mode" -q

echo "Running NER example..."
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.ner.GpuNamedEntityRecognition" -Dexec.args="--test-mode" -q

echo "Running classification example..."
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.classification.GpuDocumentClassification" -Dexec.args="--test-mode" -q

echo ""
echo "📊 Test Summary"
echo "==============="
echo "✅ Kernel optimization tests completed"
echo "✅ Performance benchmarking enhanced"
echo "✅ OpenNLP integration tested"
echo "✅ GPU diagnostics verified"
echo "✅ Working examples validated"

echo ""
echo "🎯 Next Development Phase Ready!"
echo "Project ready for:"
echo "  • Advanced GPU kernel development"
echo "  • Real OpenNLP model integration"
echo "  • Production optimization"
echo "  • Apache community contribution"

echo ""
echo "🏆 GPU Development Test Suite Completed Successfully!"