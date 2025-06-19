#!/bin/bash

# Windows Docker Container Testing Script for OpenNLP GPU
# Specifically tests Windows container support

set -e

# Source cross-platform compatibility library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cross_platform_lib.sh"

# Configuration
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"
TEST_OUTPUT_DIR="$PROJECT_ROOT/test-output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="$TEST_OUTPUT_DIR/windows_docker_test_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$TEST_LOG"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$TEST_LOG"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$TEST_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$TEST_LOG"
}

# Function to check if Docker is available and configured for Windows containers
check_windows_docker() {
    log_info "Checking Docker Windows container support..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker Desktop for Windows."
        return 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker Desktop."
        return 1
    fi
    
    # Check Docker version and OS
    local docker_os=$(docker version --format '{{.Server.Os}}' 2>/dev/null || echo "unknown")
    local docker_arch=$(docker version --format '{{.Server.Arch}}' 2>/dev/null || echo "unknown")
    
    log_info "Docker server OS: $docker_os"
    log_info "Docker server architecture: $docker_arch"
    
    # Check if Windows containers are available
    if [[ "$docker_os" == "windows" ]]; then
        log_success "Windows containers available"
        return 0
    else
        log_warning "Docker is configured for $docker_os containers"
        log_info "To run Windows containers:"
        log_info "  1. Right-click Docker Desktop system tray icon"
        log_info "  2. Select 'Switch to Windows containers...'"
        log_info "  3. Wait for Docker to restart"
        log_info "  4. Re-run this script"
        return 1
    fi
}

# Function to check Windows host requirements
check_windows_host() {
    log_info "Checking Windows host requirements..."
    
    local os=$(detect_os)
    if [[ "$os" != "windows" ]]; then
        log_warning "Not running on Windows host"
        log_info "Windows containers require a Windows host with:"
        log_info "  • Windows 10/11 Pro/Enterprise/Education"
        log_info "  • Windows Server 2016/2019/2022"
        log_info "  • Hyper-V enabled"
        log_info "  • Docker Desktop with Windows containers"
        return 1
    fi
    
    log_success "Running on Windows host"
    return 0
}

# Function to build Windows container
build_windows_container() {
    local dockerfile="$1"
    local image_name="$2"
    local description="$3"
    
    log_info "Building Windows container: $description"
    log_info "Dockerfile: $dockerfile"
    log_info "Image name: $image_name"
    
    cd "$PROJECT_ROOT"
    
    if docker build -f "$dockerfile" -t "$image_name" . 2>&1 | tee -a "$TEST_LOG"; then
        log_success "Successfully built $image_name"
        return 0
    else
        log_error "Failed to build $image_name"
        return 1
    fi
}

# Function to run Windows container test
run_windows_container() {
    local image_name="$1"
    local container_name="$2"
    local description="$3"
    
    log_info "Running Windows container test: $description"
    log_info "Image: $image_name"
    log_info "Container: $container_name"
    
    # Create Windows-compatible volume path
    local host_output_path=""
    if [[ -d "$TEST_OUTPUT_DIR" ]]; then
        # Convert Unix path to Windows path if needed
        host_output_path=$(cygpath -w "$TEST_OUTPUT_DIR" 2>/dev/null || echo "$TEST_OUTPUT_DIR")
    else
        host_output_path="$TEST_OUTPUT_DIR"
    fi
    
    log_info "Mounting host path: $host_output_path"
    
    if docker run --rm \
        --name "$container_name" \
        -v "${host_output_path}:C:\\opennlp-gpu\\test-output" \
        -e "OPENNLP_GPU_TEST_MODE=1" \
        -e "CI=true" \
        "$image_name" 2>&1 | tee -a "$TEST_LOG"; then
        log_success "Container $container_name completed successfully"
        return 0
    else
        log_error "Container $container_name failed"
        return 1
    fi
}

# Function to run Docker Compose Windows tests
run_compose_windows_tests() {
    log_info "Running Docker Compose Windows profile tests..."
    
    cd "$PROJECT_ROOT"
    
    if docker-compose -f "$DOCKER_DIR/docker-compose.yml" \
        --profile windows \
        up --build --abort-on-container-exit 2>&1 | tee -a "$TEST_LOG"; then
        log_success "Docker Compose Windows tests completed"
        return 0
    else
        log_error "Docker Compose Windows tests failed"
        return 1
    fi
    
    # Cleanup
    docker-compose -f "$DOCKER_DIR/docker-compose.yml" \
        --profile windows \
        down --remove-orphans 2>/dev/null || true
}

# Function to cleanup Windows containers
cleanup_windows_containers() {
    log_info "Cleaning up Windows containers..."
    
    # Stop any running test containers
    local containers=$(docker ps -a --filter "name=opennlp-gpu-test-windows" --format "{{.Names}}" 2>/dev/null || echo "")
    if [[ -n "$containers" ]]; then
        echo "$containers" | xargs docker rm -f 2>/dev/null || true
    fi
    
    # Remove test images (optional)
    local images=$(docker images --filter "reference=opennlp-gpu-test-windows*" --format "{{.Repository}}:{{.Tag}}" 2>/dev/null || echo "")
    if [[ -n "$images" ]] && [[ "${CLEANUP_IMAGES:-false}" == "true" ]]; then
        echo "$images" | xargs docker rmi -f 2>/dev/null || true
    fi
    
    log_info "Cleanup completed"
}

# Function to generate Windows test report
generate_windows_report() {
    local report_file="$TEST_OUTPUT_DIR/windows_docker_test_report_$TIMESTAMP.md"
    local total_tests=2  # ServerCore + Nano
    local passed_tests=0
    local failed_tests=0
    
    # Count results from log
    passed_tests=$(grep -c "\[PASS\]" "$TEST_LOG" 2>/dev/null || echo "0")
    failed_tests=$(grep -c "\[FAIL\]" "$TEST_LOG" 2>/dev/null || echo "0")
    
    cat > "$report_file" << EOF
# OpenNLP GPU Windows Docker Test Report

**Test Date:** $(date)
**Test Environment:** Windows Docker Containers
**Docker Version:** $(docker --version 2>/dev/null || echo "Unknown")

## Summary

- **Total Tests:** $total_tests
- **Passed:** $passed_tests
- **Failed:** $failed_tests
- **Windows Container Support:** $(if check_windows_docker &>/dev/null; then echo "✅ Available"; else echo "❌ Not Available"; fi)

## Test Details

### Windows Server Core Container
$(if grep -q "Successfully built.*windows.*servercore" "$TEST_LOG" 2>/dev/null; then echo "✅ Built and tested successfully"; else echo "❌ Build or test failed"; fi)

### Windows Nano Server Container  
$(if grep -q "Successfully built.*windows.*nano" "$TEST_LOG" 2>/dev/null; then echo "✅ Built and tested successfully"; else echo "❌ Build or test failed"; fi)

## Requirements for Windows Containers

- Windows 10/11 Pro, Enterprise, or Education
- Windows Server 2016/2019/2022
- Hyper-V enabled
- Docker Desktop with Windows containers mode
- Sufficient disk space for Windows base images (~4-8 GB)

## Log File
Full test output available at: \`$TEST_LOG\`

## Troubleshooting

If tests failed:
1. Ensure Docker Desktop is set to Windows containers mode
2. Check that Hyper-V is enabled in Windows Features
3. Verify sufficient disk space for Windows base images
4. Check firewall settings for Docker communication
5. Review the log file for specific error messages

EOF

    log_info "Windows test report generated: $report_file"
}

# Main execution function
main() {
    echo "🪟 OpenNLP GPU Windows Docker Container Testing"
    echo "==============================================="
    echo ""
    
    # Create output directory
    mkdir -p "$TEST_OUTPUT_DIR"
    
    # Initialize log
    log_info "Starting Windows Docker tests at $(date)"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Docker directory: $DOCKER_DIR"
    log_info "Test output: $TEST_OUTPUT_DIR"
    echo ""
    
    # Check requirements
    if ! check_windows_host; then
        log_warning "Windows host checks failed - continuing with available tests"
    fi
    
    if ! check_windows_docker; then
        log_error "Windows Docker support not available"
        log_info "This test requires Docker Desktop in Windows containers mode"
        exit 1
    fi
    
    # Cleanup any existing containers
    cleanup_windows_containers
    
    # Test Windows Server Core container
    log_info "Testing Windows Server Core container..."
    if build_windows_container "$DOCKER_DIR/test-windows.Dockerfile" "opennlp-gpu-test-windows-servercore" "Windows Server Core"; then
        run_windows_container "opennlp-gpu-test-windows-servercore" "test-windows-servercore" "Windows Server Core test"
    fi
    echo ""
    
    # Test Windows Nano Server container
    log_info "Testing Windows Nano Server container..."
    if build_windows_container "$DOCKER_DIR/test-windows-nano.Dockerfile" "opennlp-gpu-test-windows-nano" "Windows Nano Server"; then
        run_windows_container "opennlp-gpu-test-windows-nano" "test-windows-nano" "Windows Nano Server test"
    fi
    echo ""
    
    # Test with Docker Compose
    log_info "Testing with Docker Compose..."
    run_compose_windows_tests
    echo ""
    
    # Generate report
    generate_windows_report
    
    # Final cleanup
    cleanup_windows_containers
    
    # Summary
    echo "🏁 Windows Docker Testing Complete"
    echo "=================================="
    log_info "Test log: $TEST_LOG"
    log_info "Report: $TEST_OUTPUT_DIR/windows_docker_test_report_$TIMESTAMP.md"
    echo ""
    echo "✅ Windows container testing completed!"
    echo "📋 Check the report for detailed results and troubleshooting information."
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "OpenNLP GPU Windows Docker Testing Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h      Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  CLEANUP_IMAGES  Set to 'true' to remove test images after testing"
        echo ""
        echo "Requirements:"
        echo "  • Windows 10/11 Pro/Enterprise/Education or Windows Server"
        echo "  • Docker Desktop with Windows containers enabled"
        echo "  • Hyper-V enabled"
        echo "  • Sufficient disk space for Windows base images"
        exit 0
        ;;
esac

# Execute main function
main "$@"
