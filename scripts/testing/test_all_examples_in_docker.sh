#!/bin/bash

# test_all_examples_in_docker.sh
# Comprehensive script to test all OpenNLP GPU examples in all Docker environments
# This verifies that all examples work correctly in every supported Docker platform

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"
TEST_OUTPUT_DIR="${PROJECT_ROOT}/test-output/docker-examples"
TIMEOUT_SECONDS=300  # 5 minutes per test

# Docker images to test
LINUX_IMAGES=(
    "opennlp-test-ubuntu22.04"
    "opennlp-test-ubuntu20.04"
    "opennlp-test-centos8"
    "opennlp-test-fedora38"
    "opennlp-test-debian11"
    "opennlp-test-alpine"
    "opennlp-test-amazonlinux2"
)

WINDOWS_IMAGES=(
    "opennlp-test-windows"
    "opennlp-test-windows-nano"
)

# Examples to test
EXAMPLES=(
    "sentiment_analysis:GpuSentimentAnalysis"
    "ner:GpuNamedEntityRecognition" 
    "classification:GpuDocumentClassification"
    "language_detection:GpuLanguageDetection"
    "question_answering:GpuQuestionAnswering"
)

# Test results tracking
declare -A TEST_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Create output directory
mkdir -p "${TEST_OUTPUT_DIR}"

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
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

# Initialize test report
initialize_test_report() {
    local report_file="${TEST_OUTPUT_DIR}/docker_examples_test_report.md"
    cat > "${report_file}" << EOF
# Docker Examples Test Report

**Generated**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
**Project**: OpenNLP GPU Acceleration
**Test Suite**: All Examples in All Docker Environments

## Test Overview

This report shows the results of running all OpenNLP GPU examples in every supported Docker environment.

### Test Matrix

| Environment | Examples Tested | Status |
|-------------|----------------|--------|
EOF
    echo "${report_file}"
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or not accessible"
        exit 1
    fi
    
    log_success "Docker is available and running"
}

# Build Docker image if it doesn't exist
build_docker_image() {
    local dockerfile="$1"
    local image_name="$2"
    local platform="$3"
    
    log_info "Building Docker image: ${image_name}"
    
    local build_args=""
    if [[ "${platform}" == "windows" ]]; then
        build_args="--platform windows/amd64"
    fi
    
    if docker build ${build_args} -f "${DOCKER_DIR}/${dockerfile}" -t "${image_name}" "${PROJECT_ROOT}" &> "${TEST_OUTPUT_DIR}/build_${image_name}.log"; then
        log_success "Built ${image_name}"
        return 0
    else
        log_error "Failed to build ${image_name}"
        log_error "Check build log: ${TEST_OUTPUT_DIR}/build_${image_name}.log"
        return 1
    fi
}

# Test example in Docker container
test_example_in_container() {
    local image_name="$1"
    local example_dir="$2"
    local example_class="$3"
    local platform="$4"
    
    local test_name="${image_name}_${example_dir}"
    local log_file="${TEST_OUTPUT_DIR}/${test_name}.log"
    
    log_info "Testing ${example_class} in ${image_name}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Prepare test command based on platform
    local test_cmd=""
    if [[ "${platform}" == "windows" ]]; then
        test_cmd="powershell -Command \"cd C:\\app; mvn compile exec:java -Dexec.mainClass=\\\"${example_class}\\\" -Dexec.args=\\\"--test-mode\\\"\""
    else
        test_cmd="cd /app && mvn compile exec:java -Dexec.mainClass=\"${example_class}\" -Dexec.args=\"--test-mode\""
    fi
    
    # Run test with timeout
    local docker_run_args=""
    if [[ "${platform}" == "windows" ]]; then
        docker_run_args="--platform windows/amd64 --isolation process"
    fi
    
    if timeout "${TIMEOUT_SECONDS}" docker run --rm ${docker_run_args} \
        -v "${PROJECT_ROOT}:/app" \
        "${image_name}" \
        sh -c "${test_cmd}" &> "${log_file}"; then
        
        # Check if the test actually passed by looking for success indicators
        if grep -q "SUCCESS\|PASSED\|✅" "${log_file}" && ! grep -q "ERROR\|FAILED\|❌" "${log_file}"; then
            log_success "✅ ${test_name} PASSED"
            TEST_RESULTS["${test_name}"]="PASSED"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            log_error "❌ ${test_name} FAILED (execution completed but test failed)"
            TEST_RESULTS["${test_name}"]="FAILED"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        local exit_code=$?
        if [[ ${exit_code} -eq 124 ]]; then
            log_error "❌ ${test_name} TIMEOUT (exceeded ${TIMEOUT_SECONDS}s)"
            TEST_RESULTS["${test_name}"]="TIMEOUT"
        else
            log_error "❌ ${test_name} FAILED (exit code: ${exit_code})"
            TEST_RESULTS["${test_name}"]="FAILED"
        fi
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Test all examples in a specific Docker image
test_all_examples_in_image() {
    local image_name="$1"
    local platform="$2"
    
    log_header "Testing All Examples in ${image_name}"
    
    local image_passed=0
    local image_total=0
    
    for example in "${EXAMPLES[@]}"; do
        IFS=':' read -r example_dir example_class <<< "${example}"
        image_total=$((image_total + 1))
        
        if test_example_in_container "${image_name}" "${example_dir}" "${example_class}" "${platform}"; then
            image_passed=$((image_passed + 1))
        fi
    done
    
    log_info "Image ${image_name}: ${image_passed}/${image_total} examples passed"
    return $((image_total - image_passed))
}

# Test Linux Docker images
test_linux_images() {
    log_header "Testing Linux Docker Images"
    
    local linux_failed=0
    
    for image in "${LINUX_IMAGES[@]}"; do
        local dockerfile="test-$(echo "${image}" | sed 's/opennlp-test-//'}.Dockerfile"
        
        if [[ -f "${DOCKER_DIR}/${dockerfile}" ]]; then
            if build_docker_image "${dockerfile}" "${image}" "linux"; then
                if ! test_all_examples_in_image "${image}" "linux"; then
                    linux_failed=$((linux_failed + 1))
                fi
            else
                log_error "Skipping tests for ${image} due to build failure"
                linux_failed=$((linux_failed + 1))
            fi
        else
            log_warning "Dockerfile not found: ${dockerfile}, skipping ${image}"
        fi
    done
    
    return ${linux_failed}
}

# Test Windows Docker images
test_windows_images() {
    log_header "Testing Windows Docker Images"
    
    # Check if Windows containers are supported
    if ! docker system info | grep -q "OSType.*windows" 2>/dev/null; then
        log_warning "Windows containers not supported on this Docker host, skipping Windows tests"
        return 0
    fi
    
    local windows_failed=0
    
    for image in "${WINDOWS_IMAGES[@]}"; do
        local dockerfile="test-$(echo "${image}" | sed 's/opennlp-test-//').Dockerfile"
        
        if [[ -f "${DOCKER_DIR}/${dockerfile}" ]]; then
            if build_docker_image "${dockerfile}" "${image}" "windows"; then
                if ! test_all_examples_in_image "${image}" "windows"; then
                    windows_failed=$((windows_failed + 1))
                fi
            else
                log_error "Skipping tests for ${image} due to build failure"
                windows_failed=$((windows_failed + 1))
            fi
        else
            log_warning "Dockerfile not found: ${dockerfile}, skipping ${image}"
        fi
    done
    
    return ${windows_failed}
}

# Generate final test report
generate_final_report() {
    local report_file="${TEST_OUTPUT_DIR}/docker_examples_test_report.md"
    
    cat >> "${report_file}" << EOF

## Test Results Summary

- **Total Tests**: ${TOTAL_TESTS}
- **Passed**: ${PASSED_TESTS}
- **Failed**: ${FAILED_TESTS}
- **Success Rate**: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

## Detailed Results

| Test | Status | Log File |
|------|--------|----------|
EOF

    for test_name in $(printf '%s\n' "${!TEST_RESULTS[@]}" | sort); do
        local status="${TEST_RESULTS[${test_name}]}"
        local status_emoji=""
        case "${status}" in
            "PASSED") status_emoji="✅" ;;
            "FAILED") status_emoji="❌" ;;
            "TIMEOUT") status_emoji="⏰" ;;
            *) status_emoji="❓" ;;
        esac
        
        echo "| ${test_name} | ${status_emoji} ${status} | ${test_name}.log |" >> "${report_file}"
    done
    
    cat >> "${report_file}" << EOF

## Test Environment

- **Test Date**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
- **Docker Version**: $(docker --version)
- **Host OS**: $(uname -a)
- **Test Timeout**: ${TIMEOUT_SECONDS} seconds per test

## Example Test Coverage

EOF

    for example in "${EXAMPLES[@]}"; do
        IFS=':' read -r example_dir example_class <<< "${example}"
        echo "- **${example_class}**: tests/${example_dir}/" >> "${report_file}"
    done
    
    cat >> "${report_file}" << EOF

## Notes

- All examples are tested with \`--test-mode\` flag for faster execution
- Tests automatically fall back to CPU mode if GPU is not available in container
- Windows container tests require Docker Desktop with Windows container support
- Log files are available in \`test-output/docker-examples/\` directory

EOF

    log_success "Test report generated: ${report_file}"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up test artifacts..."
    
    # Remove test containers (if any are still running)
    docker ps -a --format "table {{.Names}}" | grep -E "opennlp-test-" | xargs -r docker rm -f
    
    # Optionally remove test images (uncomment if desired)
    # docker images --format "table {{.Repository}}" | grep -E "opennlp-test-" | xargs -r docker rmi -f
    
    log_info "Cleanup completed"
}

# Main execution
main() {
    log_header "OpenNLP GPU Examples - Docker Testing Suite"
    
    # Initialize
    check_docker
    local report_file
    report_file=$(initialize_test_report)
    
    # Setup cleanup on exit
    trap cleanup EXIT
    
    # Run tests
    log_info "Starting comprehensive Docker testing for all examples"
    log_info "Output directory: ${TEST_OUTPUT_DIR}"
    log_info "Test timeout: ${TIMEOUT_SECONDS} seconds per test"
    
    local linux_failed=0
    local windows_failed=0
    
    # Test Linux images
    if ! test_linux_images; then
        linux_failed=1
    fi
    
    # Test Windows images
    if ! test_windows_images; then
        windows_failed=1
    fi
    
    # Generate final report
    generate_final_report
    
    # Summary
    log_header "Test Summary"
    log_info "Total Tests: ${TOTAL_TESTS}"
    log_success "Passed: ${PASSED_TESTS}"
    if [[ ${FAILED_TESTS} -gt 0 ]]; then
        log_error "Failed: ${FAILED_TESTS}"
    else
        log_success "Failed: ${FAILED_TESTS}"
    fi
    
    local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    if [[ ${success_rate} -ge 90 ]]; then
        log_success "Success Rate: ${success_rate}% - Excellent!"
    elif [[ ${success_rate} -ge 70 ]]; then
        log_warning "Success Rate: ${success_rate}% - Good"
    else
        log_error "Success Rate: ${success_rate}% - Needs improvement"
    fi
    
    log_info "Detailed report: ${report_file}"
    log_info "Test logs available in: ${TEST_OUTPUT_DIR}"
    
    # Exit with appropriate code
    if [[ ${FAILED_TESTS} -eq 0 ]]; then
        log_success "🎉 All tests passed!"
        exit 0
    else
        log_error "❌ Some tests failed. Check logs for details."
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
