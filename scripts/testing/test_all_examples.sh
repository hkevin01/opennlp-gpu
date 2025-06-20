#!/bin/bash

# test_all_examples.sh
# Test all OpenNLP GPU examples to ensure they run correctly
# This script runs each example in test mode for faster execution

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_OUTPUT_DIR="${PROJECT_ROOT}/test-output/examples"
TIMEOUT_SECONDS=120  # 2 minutes per example

# Examples configuration
declare -a EXAMPLES=(
    "sentiment_analysis:GpuSentimentAnalysis:GPU-accelerated sentiment analysis"
    "ner:GpuNamedEntityRecognition:Named Entity Recognition with GPU acceleration"
    "classification:GpuDocumentClassification:Document classification with GPU support"
    "language_detection:GpuLanguageDetection:Multi-language detection with GPU"
    "question_answering:GpuQuestionAnswering:Question answering with GPU acceleration"
)

# Test results tracking
declare -A TEST_RESULTS
TOTAL_EXAMPLES=0
PASSED_EXAMPLES=0
FAILED_EXAMPLES=0

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
initialize_report() {
    local report_file="${TEST_OUTPUT_DIR}/examples_test_report.md"
    cat > "${report_file}" << EOF
# OpenNLP GPU Examples Test Report

**Generated**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
**Project**: OpenNLP GPU Acceleration
**Test Suite**: All Examples

## Test Overview

This report shows the results of testing all OpenNLP GPU examples to ensure they run correctly.

### Examples Tested

| Example | Class | Description | Status |
|---------|-------|-------------|--------|
EOF
    echo "${report_file}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Java
    if ! command -v java &> /dev/null; then
        log_error "Java is not installed or not in PATH"
        exit 1
    fi
    
    # Check Maven
    if ! command -v mvn &> /dev/null; then
        log_error "Maven is not installed or not in PATH"
        exit 1
    fi
    
    # Check project structure
    if [[ ! -f "${PROJECT_ROOT}/pom.xml" ]]; then
        log_error "pom.xml not found in project root: ${PROJECT_ROOT}"
        exit 1
    fi
    
    if [[ ! -d "${PROJECT_ROOT}/examples" ]]; then
        log_error "examples directory not found: ${PROJECT_ROOT}/examples"
        exit 1
    fi
    
    log_success "All prerequisites check passed"
}

# Build project
build_project() {
    log_info "Building project..."
    
    cd "${PROJECT_ROOT}"
    
    # Set Java environment
    export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
    export PATH="/usr/lib/jvm/java-21-openjdk-amd64/bin:${PATH}"
    
    if mvn clean compile > "${TEST_OUTPUT_DIR}/build.log" 2>&1; then
        log_success "Project built successfully"
        return 0
    else
        log_error "Project build failed"
        log_error "Check build log: ${TEST_OUTPUT_DIR}/build.log"
        return 1
    fi
}

# Test individual example
test_example() {
    local example_dir="$1"
    local example_class="$2"
    local description="$3"
    
    log_info "Testing ${example_class} (${description})"
    
    TOTAL_EXAMPLES=$((TOTAL_EXAMPLES + 1))
    
    local log_file="${TEST_OUTPUT_DIR}/${example_class}.log"
    local example_path="${PROJECT_ROOT}/examples/${example_dir}"
    
    # Check if example directory exists
    if [[ ! -d "${example_path}" ]]; then
        log_error "Example directory not found: ${example_path}"
        TEST_RESULTS["${example_class}"]="FAILED"
        FAILED_EXAMPLES=$((FAILED_EXAMPLES + 1))
        return 1
    fi
    
    # Check if Java file exists
    local java_file="${example_path}/${example_class}.java"
    if [[ ! -f "${java_file}" ]]; then
        log_error "Java file not found: ${java_file}"
        TEST_RESULTS["${example_class}"]="FAILED"
        FAILED_EXAMPLES=$((FAILED_EXAMPLES + 1))
        return 1
    fi
    
    cd "${PROJECT_ROOT}"
    
    # Set environment variables
    export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
    export PATH="/usr/lib/jvm/java-21-openjdk-amd64/bin:${PATH}"
    
    # Run the example with test mode arguments
    local test_args="--test-mode --batch-size=5 --quick-test"
    
    log_info "Running: mvn exec:java -Dexec.mainClass=\"${example_class}\" -Dexec.args=\"${test_args}\""
    
    if timeout "${TIMEOUT_SECONDS}" mvn exec:java \
        -Dexec.mainClass="${example_class}" \
        -Dexec.args="${test_args}" \
        > "${log_file}" 2>&1; then
        
        # Check if the test actually succeeded by looking for success indicators
        if grep -qE "(SUCCESS|PASSED|✅|Test completed successfully|Example completed)" "${log_file}" && \
           ! grep -qE "(ERROR|FAILED|Exception|❌)" "${log_file}"; then
            log_success "✅ ${example_class} PASSED"
            TEST_RESULTS["${example_class}"]="PASSED"
            PASSED_EXAMPLES=$((PASSED_EXAMPLES + 1))
            return 0
        else
            # If no clear success/failure indicators, check exit code and basic output
            if grep -qE "(Processing|Analyzing|Detecting|Classifying)" "${log_file}"; then
                log_success "✅ ${example_class} PASSED (executed successfully)"
                TEST_RESULTS["${example_class}"]="PASSED"
                PASSED_EXAMPLES=$((PASSED_EXAMPLES + 1))
                return 0
            else
                log_error "❌ ${example_class} FAILED (execution completed but no expected output)"
                TEST_RESULTS["${example_class}"]="FAILED"
                FAILED_EXAMPLES=$((FAILED_EXAMPLES + 1))
                return 1
            fi
        fi
    else
        local exit_code=$?
        if [[ ${exit_code} -eq 124 ]]; then
            log_error "❌ ${example_class} TIMEOUT (exceeded ${TIMEOUT_SECONDS}s)"
            TEST_RESULTS["${example_class}"]="TIMEOUT"
        else
            log_error "❌ ${example_class} FAILED (exit code: ${exit_code})"
            TEST_RESULTS["${example_class}"]="FAILED"
        fi
        FAILED_EXAMPLES=$((FAILED_EXAMPLES + 1))
        return 1
    fi
}

# Test all examples
test_all_examples() {
    log_header "Testing All Examples"
    
    for example in "${EXAMPLES[@]}"; do
        IFS=':' read -r example_dir example_class description <<< "${example}"
        test_example "${example_dir}" "${example_class}" "${description}"
    done
}

# Check example READMEs
check_example_readmes() {
    log_header "Checking Example READMEs"
    
    local readme_issues=0
    
    for example in "${EXAMPLES[@]}"; do
        IFS=':' read -r example_dir example_class description <<< "${example}"
        local readme_file="${PROJECT_ROOT}/examples/${example_dir}/README.md"
        
        if [[ -f "${readme_file}" ]]; then
            log_success "README exists for ${example_dir}"
            
            # Check if README has essential sections
            local missing_sections=""
            if ! grep -q "## Overview\|# Overview" "${readme_file}"; then
                missing_sections="${missing_sections} Overview"
            fi
            if ! grep -q "## Usage\|# Usage" "${readme_file}"; then
                missing_sections="${missing_sections} Usage"
            fi
            if ! grep -q "## Running\|# Running\|## Run" "${readme_file}"; then
                missing_sections="${missing_sections} Running"
            fi
            
            if [[ -n "${missing_sections}" ]]; then
                log_warning "README for ${example_dir} missing sections:${missing_sections}"
                readme_issues=$((readme_issues + 1))
            fi
        else
            log_error "README missing for ${example_dir}"
            readme_issues=$((readme_issues + 1))
        fi
    done
    
    if [[ ${readme_issues} -eq 0 ]]; then
        log_success "All example READMEs are present and complete"
    else
        log_warning "${readme_issues} README issues found"
    fi
}

# Run GPU diagnostics
run_gpu_diagnostics() {
    log_header "Running GPU Diagnostics"
    
    cd "${PROJECT_ROOT}"
    
    local diag_log="${TEST_OUTPUT_DIR}/gpu_diagnostics.log"
    
    if timeout 60 mvn exec:java \
        -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics" \
        > "${diag_log}" 2>&1; then
        
        log_success "GPU diagnostics completed successfully"
        
        # Extract key information from diagnostics
        if grep -q "GPU acceleration is ready" "${diag_log}"; then
            log_success "✅ GPU acceleration is available"
        elif grep -q "CPU fallback mode" "${diag_log}"; then
            log_warning "⚠️ Using CPU fallback mode (no GPU available)"
        else
            log_info "GPU status unclear from diagnostics"
        fi
        
        return 0
    else
        log_warning "GPU diagnostics failed or timed out"
        log_warning "This might be expected if GPU tools are not available"
        return 1
    fi
}

# Generate final report
generate_final_report() {
    local report_file="${TEST_OUTPUT_DIR}/examples_test_report.md"
    
    # Add example results to report
    for example in "${EXAMPLES[@]}"; do
        IFS=':' read -r example_dir example_class description <<< "${example}"
        local status="${TEST_RESULTS[${example_class}]:-UNKNOWN}"
        local status_emoji=""
        
        case "${status}" in
            "PASSED") status_emoji="✅" ;;
            "FAILED") status_emoji="❌" ;;
            "TIMEOUT") status_emoji="⏰" ;;
            *) status_emoji="❓" ;;
        esac
        
        echo "| ${example_dir} | ${example_class} | ${description} | ${status_emoji} ${status} |" >> "${report_file}"
    done
    
    cat >> "${report_file}" << EOF

## Test Results Summary

- **Total Examples**: ${TOTAL_EXAMPLES}
- **Passed**: ${PASSED_EXAMPLES}
- **Failed**: ${FAILED_EXAMPLES}
- **Success Rate**: $(( PASSED_EXAMPLES * 100 / TOTAL_EXAMPLES ))%

## Detailed Test Logs

| Example | Log File | Status |
|---------|----------|--------|
EOF

    for example in "${EXAMPLES[@]}"; do
        IFS=':' read -r example_dir example_class description <<< "${example}"
        local status="${TEST_RESULTS[${example_class}]:-UNKNOWN}"
        local status_emoji=""
        
        case "${status}" in
            "PASSED") status_emoji="✅" ;;
            "FAILED") status_emoji="❌" ;;
            "TIMEOUT") status_emoji="⏰" ;;
            *) status_emoji="❓" ;;
        esac
        
        echo "| ${example_class} | ${example_class}.log | ${status_emoji} ${status} |" >> "${report_file}"
    done
    
    cat >> "${report_file}" << EOF

## Test Environment

- **Test Date**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
- **Java Version**: $(java -version 2>&1 | head -n1)
- **Maven Version**: $(mvn --version 2>&1 | head -n1)
- **Host OS**: $(uname -a)
- **Project Root**: ${PROJECT_ROOT}
- **Test Timeout**: ${TIMEOUT_SECONDS} seconds per example

## Notes

- All examples run in test mode with reduced datasets for faster execution
- Examples automatically fall back to CPU if GPU is not available
- Test arguments used: \`--test-mode --batch-size=5 --quick-test\`

EOF

    if [[ ${FAILED_EXAMPLES} -gt 0 ]]; then
        cat >> "${report_file}" << EOF
## Failed Examples

The following examples failed and need attention:

EOF
        for example in "${EXAMPLES[@]}"; do
            IFS=':' read -r example_dir example_class description <<< "${example}"
            local status="${TEST_RESULTS[${example_class}]:-UNKNOWN}"
            
            if [[ "${status}" == "FAILED" || "${status}" == "TIMEOUT" ]]; then
                echo "- **${example_class}**: ${description} (${status})" >> "${report_file}"
                echo "  - Log: \`test-output/examples/${example_class}.log\`" >> "${report_file}"
                echo "  - Location: \`examples/${example_dir}/\`" >> "${report_file}"
                echo >> "${report_file}"
            fi
        done
    fi
    
    log_success "Test report generated: ${report_file}"
}

# Main execution
main() {
    log_header "OpenNLP GPU Examples Test Suite"
    
    # Initialize
    local report_file
    report_file=$(initialize_report)
    
    log_info "Testing all OpenNLP GPU examples"
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Output directory: ${TEST_OUTPUT_DIR}"
    log_info "Test timeout: ${TIMEOUT_SECONDS} seconds per example"
    
    # Run tests
    check_prerequisites
    
    if ! build_project; then
        log_error "Cannot proceed with tests due to build failure"
        exit 1
    fi
    
    run_gpu_diagnostics || true  # Don't fail if diagnostics fail
    
    test_all_examples
    
    check_example_readmes
    
    generate_final_report
    
    # Summary
    log_header "Test Summary"
    log_info "Total Examples: ${TOTAL_EXAMPLES}"
    log_success "Passed: ${PASSED_EXAMPLES}"
    
    if [[ ${FAILED_EXAMPLES} -gt 0 ]]; then
        log_error "Failed: ${FAILED_EXAMPLES}"
    else
        log_success "Failed: ${FAILED_EXAMPLES}"
    fi
    
    local success_rate=$((PASSED_EXAMPLES * 100 / TOTAL_EXAMPLES))
    if [[ ${success_rate} -eq 100 ]]; then
        log_success "Success Rate: ${success_rate}% - Perfect!"
    elif [[ ${success_rate} -ge 80 ]]; then
        log_success "Success Rate: ${success_rate}% - Excellent!"
    elif [[ ${success_rate} -ge 60 ]]; then
        log_warning "Success Rate: ${success_rate}% - Good"
    else
        log_error "Success Rate: ${success_rate}% - Needs improvement"
    fi
    
    log_info "Detailed report: ${report_file}"
    log_info "Test logs available in: ${TEST_OUTPUT_DIR}"
    
    # Exit with appropriate code
    if [[ ${FAILED_EXAMPLES} -eq 0 ]]; then
        log_success "🎉 All examples passed!"
        exit 0
    else
        log_error "❌ Some examples failed. Check logs for details."
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
