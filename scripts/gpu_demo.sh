#!/bin/bash
cd "$(dirname "$0")"
echo "🚀 Running OpenNLP GPU Extension Demo"
echo "======================================"
echo
echo "1. GPU Diagnostics:"
java -cp "target/classes:$(cat classpath.txt 2>/dev/null || echo '')" org.apache.opennlp.gpu.tools.GpuDiagnostics
echo
echo "2. GPU ML Demo:"
java -cp "target/classes:$(cat classpath.txt 2>/dev/null || echo '')" org.apache.opennlp.gpu.ml.GpuMlDemo
