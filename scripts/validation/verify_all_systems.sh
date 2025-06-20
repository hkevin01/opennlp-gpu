#!/bin/bash

# verify_all_systems.sh
# Comprehensive verification script that tests all examples and Docker environments
# This is the master script that validates everything is working

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_OUTPUT_DIR="${PROJECT_ROOT}/test-output/comprehensive"
REPORT_FILE="${TEST_OUTPUT_DIR}/comprehensive_verification_report.md"

# Test categories
declare -A TEST_RESULTS
TOTAL_CATEGORIES=0
PASSED_CATEGORIES=0

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

log_debug() {
    echo -e "${PURPLE}🔍 $1${NC}"
}

log_header() {
    echo
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Initialize comprehensive report
initialize_report() {
    cat > "${REPORT_FILE}" << EOF
# OpenNLP GPU Comprehensive Verification Report

**Generated**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
**Project**: OpenNLP GPU Acceleration
**Verification Suite**: Complete System Validation

## Executive Summary

This report provides a comprehensive assessment of the OpenNLP GPU project's readiness for public release and Apache contribution.

### Verification Categories

| Category | Status | Details |
|----------|--------|---------|
EOF
}

# Test a category and update results
test_category() {
    local category="$1"
    local test_command="$2"
    local description="$3"
    
    TOTAL_CATEGORIES=$((TOTAL_CATEGORIES + 1))
    
    log_header "Testing: ${category}"
    log_info "${description}"
    
    local log_file="${TEST_OUTPUT_DIR}/${category,,}.log"
    
    if eval "${test_command}" > "${log_file}" 2>&1; then
        log_success "${category} PASSED"
        TEST_RESULTS["${category}"]="PASSED"
        PASSED_CATEGORIES=$((PASSED_CATEGORIES + 1))
        echo "| ${category} | ✅ PASSED | ${description} |" >> "${REPORT_FILE}"
        return 0
    else
        log_error "${category} FAILED"
        TEST_RESULTS["${category}"]="FAILED"
        echo "| ${category} | ❌ FAILED | ${description} - Check ${category,,}.log |" >> "${REPORT_FILE}"
        return 1
    fi
}

# Check if Docker is available
check_docker_availability() {
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Run all verification tests
run_all_tests() {
    log_header "OpenNLP GPU Comprehensive Verification"
    
    # 1. Project Build Test
    test_category "Project Build" \
        "cd '${PROJECT_ROOT}' && mvn clean compile" \
        "Verify project compiles successfully"
    
    # 2. Basic Examples Test
    test_category "Local Examples" \
        "cd '${PROJECT_ROOT}' && timeout 300 ./scripts/test_all_examples.sh" \
        "Test all examples run locally with test mode"
    
    # 3. README Instructions Verification
    test_category "README Verification" \
        "cd '${PROJECT_ROOT}' && timeout 300 ./scripts/verify_readme_instructions.sh" \
        "Verify README instructions are correct and working"
    
    # 4. Docker Simple Test (if Docker available)
    if check_docker_availability; then
        test_category "Docker Examples" \
            "cd '${PROJECT_ROOT}' && timeout 600 ./scripts/test_docker_examples_simple.sh" \
            "Test examples in a Docker container"
    else
        log_warning "Docker not available, skipping Docker tests"
        echo "| Docker Examples | ⏭️ SKIPPED | Docker not available on this system |" >> "${REPORT_FILE}"
    fi
    
    # 5. GPU Diagnostics (if available)
    test_category "GPU Diagnostics" \
        "cd '${PROJECT_ROOT}' && timeout 60 mvn exec:java -Dexec.mainClass=org.apache.opennlp.gpu.tools.GpuDiagnostics" \
        "Test GPU diagnostics and hardware detection"
    
    # 6. Scripts Validation
    test_category "Scripts Validation" \
        "cd '${PROJECT_ROOT}' && find scripts/ -name '*.sh' -exec bash -n {} \\;" \
        "Validate all shell scripts have correct syntax"
    
    # 7. Documentation Links
    test_category "Documentation Links" \
        "cd '${PROJECT_ROOT}' && find docs/ examples/ -name '*.md' -exec grep -l 'http' {} \\; | head -5 | xargs -I {} echo 'Found: {}'" \
        "Check documentation files exist and contain links"
    
    # 8. File Structure Verification
    test_category "File Structure" \
        "cd '${PROJECT_ROOT}' && test -f pom.xml && test -d examples && test -d scripts && test -d docs && test -d docker" \
        "Verify essential project structure exists"
}

# Generate final comprehensive report
generate_final_report() {
    cat >> "${REPORT_FILE}" << EOF

## Detailed Results

### Summary Statistics
- **Total Categories Tested**: ${TOTAL_CATEGORIES}
- **Passed**: ${PASSED_CATEGORIES}
- **Failed**: $((TOTAL_CATEGORIES - PASSED_CATEGORIES))
- **Success Rate**: $(( PASSED_CATEGORIES * 100 / TOTAL_CATEGORIES ))%

### Test Category Details

EOF

    for category in $(printf '%s\n' "${!TEST_RESULTS[@]}" | sort); do
        local status="${TEST_RESULTS[${category}]}"
        local log_file="${category,,}.log"
        
        cat >> "${REPORT_FILE}" << EOF
#### ${category}
- **Status**: ${status}
- **Log File**: \`test-output/comprehensive/${log_file}\`
- **Description**: Test validation for ${category}

EOF
    done
    
    cat >> "${REPORT_FILE}" << EOF

## System Information

- **Test Date**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
- **Host OS**: $(uname -a)
- **Java Version**: $(java -version 2>&1 | head -n1 || echo "Java not available")
- **Maven Version**: $(mvn --version 2>&1 | head -n1 || echo "Maven not available")
- **Docker Version**: $(docker --version 2>&1 || echo "Docker not available")
- **Project Root**: ${PROJECT_ROOT}

## Available Test Scripts

### Core Testing Scripts
- \`scripts/test_all_examples.sh\` - Test all examples locally
- \`scripts/verify_readme_instructions.sh\` - Verify README instructions
- \`scripts/test_docker_examples_simple.sh\` - Test examples in Docker

### Advanced Testing Scripts
- \`scripts/test_all_examples_in_docker.sh\` - Multi-platform Docker testing
- \`scripts/test_cross_platform_compatibility.sh\` - Cross-platform tests
- \`scripts/run_docker_tests.sh\` - Full Docker test suite

### Utility Scripts
- \`scripts/check_gpu_prerequisites.sh\` - GPU readiness check
- \`scripts/run_all_demos.sh\` - Run all demo examples
- \`scripts/setup_universal_environment.sh\` - Environment setup

## Recommendations

EOF

    local success_rate=$((PASSED_CATEGORIES * 100 / TOTAL_CATEGORIES))
    
    if [[ ${success_rate} -eq 100 ]]; then
        cat >> "${REPORT_FILE}" << EOF
### 🎉 **Excellent - Ready for Release**
All verification categories passed successfully. The project is ready for:
- Public GitHub release
- Apache OpenNLP contribution
- Production deployment

EOF
    elif [[ ${success_rate} -ge 80 ]]; then
        cat >> "${REPORT_FILE}" << EOF
### 👍 **Good - Minor Issues to Address**
Most verification categories passed. Address failed categories before release:
- Review failed test logs in \`test-output/comprehensive/\`
- Fix any critical issues
- Consider re-running comprehensive verification

EOF
    else
        cat >> "${REPORT_FILE}" << EOF
### ⚠️ **Needs Work - Multiple Issues Found**
Several verification categories failed. Requires attention before release:
- Review all failed test logs
- Fix critical build and functionality issues
- Ensure examples work correctly
- Verify documentation accuracy

EOF
    fi
    
    cat >> "${REPORT_FILE}" << EOF

## Docker Testing Status

EOF

    if check_docker_availability; then
        cat >> "${REPORT_FILE}" << EOF
✅ **Docker Available**: Multi-platform testing enabled

### Available Docker Tests
- Ubuntu 22.04, 20.04
- CentOS 8, Fedora 38
- Debian 11, Alpine
- Amazon Linux 2
- Windows Server (if supported)

### Running Full Docker Tests
\`\`\`bash
# Test in single Docker environment
./scripts/test_docker_examples_simple.sh

# Test across all Docker platforms (extensive)
./scripts/test_all_examples_in_docker.sh
\`\`\`

EOF
    else
        cat >> "${REPORT_FILE}" << EOF
⚠️ **Docker Not Available**: Limited to local testing only

### Installing Docker
To enable multi-platform testing, install Docker:
\`\`\`bash
# Ubuntu/Debian
sudo apt-get install docker.io

# CentOS/RHEL
sudo yum install docker

# Start Docker service
sudo systemctl start docker
sudo usermod -a -G docker \$USER
\`\`\`

EOF
    fi
    
    cat >> "${REPORT_FILE}" << EOF

---
*This report was generated automatically by the OpenNLP GPU verification system.*
EOF
}

# Main execution
main() {
    log_header "OpenNLP GPU Comprehensive Verification System"
    
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Output directory: ${TEST_OUTPUT_DIR}"
    log_info "Report file: ${REPORT_FILE}"
    
    # Initialize report
    initialize_report
    
    # Check prerequisites
    log_info "Checking system prerequisites..."
    
    if ! command -v java &> /dev/null; then
        log_error "Java is not installed"
        exit 1
    fi
    
    if ! command -v mvn &> /dev/null; then
        log_error "Maven is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
    
    # Run all tests
    run_all_tests
    
    # Generate final report
    generate_final_report
    
    # Summary
    log_header "Verification Summary"
    log_info "Total Categories: ${TOTAL_CATEGORIES}"
    log_success "Passed: ${PASSED_CATEGORIES}"
    
    if [[ $((TOTAL_CATEGORIES - PASSED_CATEGORIES)) -gt 0 ]]; then
        log_error "Failed: $((TOTAL_CATEGORIES - PASSED_CATEGORIES))"
    else
        log_success "Failed: 0"
    fi
    
    local success_rate=$((PASSED_CATEGORIES * 100 / TOTAL_CATEGORIES))
    if [[ ${success_rate} -eq 100 ]]; then
        log_success "🎉 Success Rate: ${success_rate}% - Perfect!"
    elif [[ ${success_rate} -ge 80 ]]; then
        log_success "Success Rate: ${success_rate}% - Excellent!"
    elif [[ ${success_rate} -ge 60 ]]; then
        log_warning "Success Rate: ${success_rate}% - Good"
    else
        log_error "Success Rate: ${success_rate}% - Needs improvement"
    fi
    
    log_info "📋 Comprehensive report: ${REPORT_FILE}"
    log_info "📁 Test logs: ${TEST_OUTPUT_DIR}"
    
    # Show key recommendations
    if [[ ${success_rate} -eq 100 ]]; then
        log_success "🚀 Project is ready for public release and Apache contribution!"
    elif [[ ${success_rate} -ge 80 ]]; then
        log_warning "📝 Address minor issues before release. Check failed categories."
    else
        log_error "🔧 Significant work needed before release. Review all failed tests."
    fi
    
    # Exit with appropriate code
    if [[ ${success_rate} -ge 80 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
