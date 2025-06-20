#!/bin/bash

# Universal Multi-Platform Testing Script for OpenNLP GPU
# Orchestrates testing across Linux, macOS, Windows, and Docker containers

set -e

# Source cross-platform compatibility library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cross_platform_lib.sh"

# Configuration
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_OUTPUT_DIR="$PROJECT_ROOT/test-output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$TEST_OUTPUT_DIR/universal_test_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TEST_SUITES=0
PASSED_TEST_SUITES=0
FAILED_TEST_SUITES=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$MAIN_LOG"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$MAIN_LOG"
    ((PASSED_TEST_SUITES++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$MAIN_LOG"
    ((FAILED_TEST_SUITES++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$MAIN_LOG"
}

log_header() {
    echo -e "${PURPLE}[TEST]${NC} $1" | tee -a "$MAIN_LOG"
}

# Function to run cross-platform compatibility tests
run_cross_platform_tests() {
    log_header "Running cross-platform compatibility tests..."
    ((TOTAL_TEST_SUITES++))
    
    if [[ -x "$SCRIPT_DIR/test_cross_platform_compatibility.sh" ]]; then
        if "$SCRIPT_DIR/test_cross_platform_compatibility.sh" 2>&1 | tee -a "$MAIN_LOG"; then
            log_success "Cross-platform compatibility tests passed"
            return 0
        else
            log_error "Cross-platform compatibility tests failed"
            return 1
        fi
    else
        log_warning "Cross-platform compatibility test script not found or not executable"
        return 1
    fi
}

# Function to run Linux Docker container tests
run_linux_docker_tests() {
    log_header "Running Linux Docker container tests..."
    ((TOTAL_TEST_SUITES++))
    
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        if [[ -x "$SCRIPT_DIR/run_docker_tests.sh" ]]; then
            if "$SCRIPT_DIR/run_docker_tests.sh" --linux-only 2>&1 | tee -a "$MAIN_LOG"; then
                log_success "Linux Docker tests passed"
                return 0
            else
                log_error "Linux Docker tests failed"
                return 1
            fi
        else
            log_warning "Docker test script not found or not executable"
            return 1
        fi
    else
        log_warning "Docker not available - skipping Linux container tests"
        return 1
    fi
}

# Function to run Windows Docker container tests
run_windows_docker_tests() {
    log_header "Running Windows Docker container tests..."
    ((TOTAL_TEST_SUITES++))
    
    local os=$(detect_os)
    
    if [[ "$os" == "windows" ]] && command -v docker &> /dev/null; then
        # Check if Docker is in Windows container mode
        local docker_os=$(docker version --format '{{.Server.Os}}' 2>/dev/null || echo "unknown")
        
        if [[ "$docker_os" == "windows" ]]; then
            if [[ -x "$SCRIPT_DIR/test_windows_docker.sh" ]]; then
                if "$SCRIPT_DIR/test_windows_docker.sh" 2>&1 | tee -a "$MAIN_LOG"; then
                    log_success "Windows Docker tests passed"
                    return 0
                else
                    log_error "Windows Docker tests failed"
                    return 1
                fi
            else
                log_warning "Windows Docker test script not found or not executable"
                return 1
            fi
        else
            log_warning "Docker is not in Windows container mode - skipping Windows container tests"
            log_info "To run Windows container tests:"
            log_info "  1. Switch Docker Desktop to Windows containers"
            log_info "  2. Re-run this script"
            return 1
        fi
    else
        log_warning "Windows Docker tests require Windows host with Docker - skipping"
        return 1
    fi
}

# Function to run Java/Maven build tests
run_build_tests() {
    log_header "Running Java/Maven build tests..."
    ((TOTAL_TEST_SUITES++))
    
    cd "$PROJECT_ROOT"
    
    if command -v mvn &> /dev/null && command -v java &> /dev/null; then
        log_info "Running Maven clean compile..."
        if mvn clean compile -q 2>&1 | tee -a "$MAIN_LOG"; then
            log_info "Maven compile successful"
            
            log_info "Running Maven test compilation..."
            if mvn test-compile -q 2>&1 | tee -a "$MAIN_LOG"; then
                log_success "Build tests passed"
                return 0
            else
                log_error "Maven test compilation failed"
                return 1
            fi
        else
            log_error "Maven compile failed"
            return 1
        fi
    else
        log_warning "Java or Maven not available - skipping build tests"
        return 1
    fi
}

# Function to run GPU diagnostics
run_gpu_diagnostics() {
    log_header "Running GPU diagnostics..."
    ((TOTAL_TEST_SUITES++))
    
    cd "$PROJECT_ROOT"
    
    if command -v mvn &> /dev/null && [[ -f "pom.xml" ]]; then
        log_info "Running GPU diagnostics tool..."
        if mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics" -q 2>&1 | tee -a "$MAIN_LOG"; then
            log_success "GPU diagnostics completed"
            return 0
        else
            log_warning "GPU diagnostics failed (may be expected without GPU hardware)"
            return 0  # Don't fail the entire test suite for GPU unavailability
        fi
    else
        log_warning "Maven or pom.xml not available - skipping GPU diagnostics"
        return 1
    fi
}

# Function to run example demonstrations
run_demo_tests() {
    log_header "Running example demonstrations..."
    ((TOTAL_TEST_SUITES++))
    
    if [[ -x "$SCRIPT_DIR/run_all_demos.sh" ]]; then
        if "$SCRIPT_DIR/run_all_demos.sh" 2>&1 | tee -a "$MAIN_LOG"; then
            log_success "Demo tests passed"
            return 0
        else
            log_error "Demo tests failed"
            return 1
        fi
    else
        log_warning "Demo test script not found or not executable"
        return 1
    fi
}

# Function to verify all scripts exist and are executable
verify_test_scripts() {
    log_header "Verifying test scripts..."
    
    local scripts=(
        "test_cross_platform_compatibility.sh"
        "run_docker_tests.sh"
        "test_windows_docker.sh"
        "run_all_demos.sh"
        "check_gpu_prerequisites.sh"
        "setup_universal_environment.sh"
    )
    
    local missing_scripts=0
    
    for script in "${scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        if [[ -f "$script_path" ]]; then
            if [[ -x "$script_path" ]]; then
                log_info "✅ $script (executable)"
            else
                log_warning "⚠️ $script (not executable)"
                chmod +x "$script_path" 2>/dev/null || log_warning "Could not make $script executable"
            fi
        else
            log_error "❌ $script (missing)"
            ((missing_scripts++))
        fi
    done
    
    if [[ $missing_scripts -eq 0 ]]; then
        log_success "All test scripts verified"
        return 0
    else
        log_error "$missing_scripts scripts are missing"
        return 1
    fi
}

# Function to generate comprehensive test report
generate_comprehensive_report() {
    local report_file="$TEST_OUTPUT_DIR/universal_test_report_$TIMESTAMP.md"
    local os=$(detect_os)
    local arch=$(detect_arch)
    local distro=$(detect_distro)
    
    cat > "$report_file" << EOF
# OpenNLP GPU Universal Test Report

**Test Date:** $(date)
**Test Environment:** $os ($arch) - $distro
**Java Version:** $(java -version 2>&1 | head -n 1 | cut -d'"' -f2 2>/dev/null || echo "Not Available")
**Maven Version:** $(mvn -version 2>/dev/null | head -n 1 | cut -d' ' -f3 2>/dev/null || echo "Not Available")
**Docker Version:** $(docker --version 2>/dev/null || echo "Not Available")

## Summary

- **Total Test Suites:** $TOTAL_TEST_SUITES
- **Passed:** $PASSED_TEST_SUITES
- **Failed:** $FAILED_TEST_SUITES
- **Success Rate:** $(( PASSED_TEST_SUITES * 100 / TOTAL_TEST_SUITES ))%

## Test Suites Results

### Cross-Platform Compatibility
$(if grep -q "Cross-platform compatibility tests passed" "$MAIN_LOG" 2>/dev/null; then echo "✅ PASSED"; else echo "❌ FAILED"; fi)

### Java/Maven Build
$(if grep -q "Build tests passed" "$MAIN_LOG" 2>/dev/null; then echo "✅ PASSED"; else echo "❌ FAILED"; fi)

### GPU Diagnostics
$(if grep -q "GPU diagnostics completed" "$MAIN_LOG" 2>/dev/null; then echo "✅ PASSED"; else echo "⚠️ SKIPPED/FAILED"; fi)

### Linux Docker Containers
$(if grep -q "Linux Docker tests passed" "$MAIN_LOG" 2>/dev/null; then echo "✅ PASSED"; else echo "❌ FAILED/SKIPPED"; fi)

### Windows Docker Containers
$(if grep -q "Windows Docker tests passed" "$MAIN_LOG" 2>/dev/null; then echo "✅ PASSED"; else echo "❌ FAILED/SKIPPED"; fi)

### Example Demonstrations
$(if grep -q "Demo tests passed" "$MAIN_LOG" 2>/dev/null; then echo "✅ PASSED"; else echo "❌ FAILED"; fi)

## Platform Support

- **Linux:** ✅ Supported (tested on $distro)
- **macOS:** $(if [[ "$os" == "macos" ]]; then echo "✅ Supported"; else echo "🔄 Requires testing on macOS"; fi)
- **Windows:** $(if [[ "$os" == "windows" ]]; then echo "✅ Supported"; else echo "🔄 Requires testing on Windows"; fi)

## Container Support

- **Linux Containers:** $(if command -v docker &>/dev/null; then echo "✅ Available"; else echo "❌ Docker not available"; fi)
- **Windows Containers:** $(if [[ "$os" == "windows" ]] && command -v docker &>/dev/null; then echo "✅ Available"; else echo "⚠️ Requires Windows host with Docker"; fi)

## Log Files

- **Main Test Log:** \`$MAIN_LOG\`
- **Individual Test Logs:** \`$TEST_OUTPUT_DIR/\`

## Recommendations

$(if [[ $FAILED_TEST_SUITES -eq 0 ]]; then
    echo "- ✅ All tests passed! The project is ready for production deployment"
    echo "- Consider setting up CI/CD pipelines for automated testing"
    echo "- Document the tested platforms in your deployment guide"
else
    echo "- ⚠️ Review failed tests and address compatibility issues"
    echo "- Check system requirements for failed platforms"
    echo "- Update documentation with supported platform limitations"
fi)

## Next Steps

1. Review individual test logs for detailed results
2. Address any failed test cases
3. Update platform support documentation
4. Set up automated testing pipelines
5. Deploy to target environments

EOF

    log_info "Comprehensive test report generated: $report_file"
}

# Function to display system information
display_system_info() {
    echo "🖥️ System Information"
    echo "===================="
    
    local os=$(detect_os)
    local arch=$(detect_arch)
    local distro=$(detect_distro)
    local cpu_count=$(xp_get_cpu_count)
    local memory_gb=$(xp_get_memory_gb)
    
    echo "Operating System: $os"
    echo "Architecture: $arch"
    echo "Distribution: $distro"
    echo "CPU Cores: $cpu_count"
    echo "Memory: ${memory_gb}GB"
    echo "Java: $(java -version 2>&1 | head -n 1 | cut -d'"' -f2 2>/dev/null || echo "Not Available")"
    echo "Maven: $(mvn -version 2>/dev/null | head -n 1 | cut -d' ' -f3 2>/dev/null || echo "Not Available")"
    echo "Docker: $(docker --version 2>/dev/null || echo "Not Available")"
    echo ""
}

# Main execution function
main() {
    echo "🚀 OpenNLP GPU Universal Multi-Platform Testing"
    echo "==============================================="
    echo ""
    
    # Create output directory
    mkdir -p "$TEST_OUTPUT_DIR"
    
    # Initialize log
    log_info "Starting universal testing at $(date)"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Test output: $TEST_OUTPUT_DIR"
    echo ""
    
    # Display system information
    display_system_info
    
    # Verify test scripts
    if ! verify_test_scripts; then
        log_error "Test script verification failed"
        exit 1
    fi
    echo ""
    
    # Run test suites
    log_info "Starting test execution..."
    echo ""
    
    # Always run these tests
    run_cross_platform_tests
    echo ""
    
    run_build_tests
    echo ""
    
    run_gpu_diagnostics
    echo ""
    
    run_demo_tests
    echo ""
    
    # Conditional tests based on environment
    run_linux_docker_tests
    echo ""
    
    run_windows_docker_tests
    echo ""
    
    # Generate comprehensive report
    generate_comprehensive_report
    
    # Display final results
    echo "🏁 Universal Testing Complete"
    echo "============================="
    log_info "Total test suites: $TOTAL_TEST_SUITES"
    log_success "Passed test suites: $PASSED_TEST_SUITES"
    
    if [[ $FAILED_TEST_SUITES -gt 0 ]]; then
        log_error "Failed test suites: $FAILED_TEST_SUITES"
        echo ""
        echo "❌ Some test suites failed. Check the logs for details:"
        echo "   Main log: $MAIN_LOG"
        echo "   Report: $TEST_OUTPUT_DIR/universal_test_report_$TIMESTAMP.md"
        exit 1
    else
        echo ""
        echo "✅ All test suites passed! OpenNLP GPU is ready for cross-platform deployment."
        echo "📋 Comprehensive report: $TEST_OUTPUT_DIR/universal_test_report_$TIMESTAMP.md"
    fi
}

# Handle script arguments
case "${1:-}" in
    --cross-platform-only)
        main() {
            echo "🔄 Running cross-platform tests only..."
            mkdir -p "$TEST_OUTPUT_DIR"
            log_info "Starting cross-platform-only testing at $(date)"
            verify_test_scripts
            run_cross_platform_tests
            generate_comprehensive_report
            if [[ $FAILED_TEST_SUITES -gt 0 ]]; then exit 1; fi
        }
        ;;
    --docker-only)
        main() {
            echo "🐳 Running Docker tests only..."
            mkdir -p "$TEST_OUTPUT_DIR"
            log_info "Starting Docker-only testing at $(date)"
            verify_test_scripts
            run_linux_docker_tests
            run_windows_docker_tests
            generate_comprehensive_report
            if [[ $FAILED_TEST_SUITES -gt 0 ]]; then exit 1; fi
        }
        ;;
    --build-only)
        main() {
            echo "🔨 Running build tests only..."
            mkdir -p "$TEST_OUTPUT_DIR"
            log_info "Starting build-only testing at $(date)"
            run_build_tests
            run_gpu_diagnostics
            generate_comprehensive_report
            if [[ $FAILED_TEST_SUITES -gt 0 ]]; then exit 1; fi
        }
        ;;
    --help|-h)
        echo "OpenNLP GPU Universal Testing Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --cross-platform-only  Run only cross-platform compatibility tests"
        echo "  --docker-only          Run only Docker container tests"
        echo "  --build-only           Run only build and GPU tests"
        echo "  --help, -h             Show this help message"
        echo ""
        echo "Default: Run all available tests"
        exit 0
        ;;
esac

# Execute main function
main "$@"
