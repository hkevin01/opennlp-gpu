#!/bin/bash

# test_docker_examples_simple.sh
# Simple script to test all examples in one Docker environment
# This is a faster test to validate our setup before running full multi-platform tests

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_OUTPUT_DIR="${PROJECT_ROOT}/test-output/docker-simple"
DOCKER_IMAGE="ubuntu:22.04"
CONTAINER_NAME="opennlp-test-simple"

# Examples to test
EXAMPLES=(
    "sentiment_analysis:GpuSentimentAnalysis"
    "ner:GpuNamedEntityRecognition" 
    "classification:GpuDocumentClassification"
    "language_detection:GpuLanguageDetection"
    "question_answering:GpuQuestionAnswering"
)

# Create output directory
mkdir -p "${TEST_OUTPUT_DIR}"

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Cleanup function
cleanup_container() {
    if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        log_info "Cleaning up container: ${CONTAINER_NAME}"
        docker rm -f "${CONTAINER_NAME}" &> /dev/null || true
    fi
}

# Setup container
setup_container() {
    log_info "Setting up Docker container"
    
    # Cleanup any existing container
    cleanup_container
    
    # Create and start container
    docker run -d --name "${CONTAINER_NAME}" \
        -v "${PROJECT_ROOT}:/app" \
        "${DOCKER_IMAGE}" \
        tail -f /dev/null
    
    # Install Java and Maven
    log_info "Installing Java and Maven in container"
    docker exec "${CONTAINER_NAME}" bash -c "
        apt-get update -qq && 
        apt-get install -y -qq openjdk-21-jdk maven && 
        export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
    "
    
    log_success "Container setup completed"
}

# Test example in container
test_example() {
    local example_dir="$1"
    local example_class="$2"
    
    log_info "Testing ${example_class}"
    
    local log_file="${TEST_OUTPUT_DIR}/${example_class}.log"
    
    # Run the example
    if docker exec "${CONTAINER_NAME}" bash -c "
        cd /app && 
        export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 && 
        export PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin:\$PATH &&
        timeout 60 mvn compile exec:java -Dexec.mainClass=\"${example_class}\" -Dexec.args=\"--test-mode --quick-test\"
    " > "${log_file}" 2>&1; then
        
        # Check for success indicators
        if grep -qE "(SUCCESS|PASSED|✅|Test completed successfully)" "${log_file}"; then
            log_success "${example_class} PASSED"
            return 0
        else
            log_error "${example_class} FAILED (no success indicator)"
            return 1
        fi
    else
        log_error "${example_class} FAILED (execution failed)"
        return 1
    fi
}

# Main execution
main() {
    log_header "Simple Docker Examples Test"
    
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Docker image: ${DOCKER_IMAGE}"
    log_info "Output directory: ${TEST_OUTPUT_DIR}"
    
    # Setup cleanup on exit
    trap cleanup_container EXIT
    
    # Check Docker
    if ! docker info &> /dev/null; then
        log_error "Docker is not available"
        exit 1
    fi
    
    # Setup container
    setup_container
    
    # Compile project in container
    log_info "Building project in container"
    if ! docker exec "${CONTAINER_NAME}" bash -c "
        cd /app && 
        export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 && 
        export PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin:\$PATH &&
        mvn clean compile
    " > "${TEST_OUTPUT_DIR}/build.log" 2>&1; then
        log_error "Project build failed"
        log_error "Check build log: ${TEST_OUTPUT_DIR}/build.log"
        exit 1
    fi
    log_success "Project build completed"
    
    # Test all examples
    local passed=0
    local total=0
    
    for example in "${EXAMPLES[@]}"; do
        IFS=':' read -r example_dir example_class <<< "${example}"
        total=$((total + 1))
        
        if test_example "${example_dir}" "${example_class}"; then
            passed=$((passed + 1))
        fi
    done
    
    # Summary
    log_header "Test Results"
    log_info "Total: ${total}"
    log_success "Passed: ${passed}"
    log_error "Failed: $((total - passed))"
    
    local success_rate=$((passed * 100 / total))
    if [[ ${success_rate} -eq 100 ]]; then
        log_success "🎉 All examples work in Docker!"
        exit 0
    else
        log_error "❌ Some examples failed. Check logs in ${TEST_OUTPUT_DIR}"
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
