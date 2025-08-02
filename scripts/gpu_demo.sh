#!/bin/bash

# Get the project root directory
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
