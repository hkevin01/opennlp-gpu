#!/bin/bash

# Multi-Platform Docker Testing Script for OpenNLP GPU
# Runs compatibility tests across multiple Linux distributions

set -e

echo "🐳 OpenNLP GPU Multi-Platform Docker Testing"
echo "============================================"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_RESULTS_DIR="$PROJECT_ROOT/test-output/docker-tests"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="$TEST_RESULTS_DIR/docker_test_summary_$TIMESTAMP.log"

# Create test results directory
mkdir -p "$TEST_RESULTS_DIR"

# Initialize summary log
echo "OpenNLP GPU Docker Multi-Platform Test Summary - $(date)" > "$SUMMARY_LOG"
echo "=======================================================" >> "$SUMMARY_LOG"
echo >> "$SUMMARY_LOG"

# Test environments
ENVIRONMENTS=(
    "ubuntu22:Ubuntu 22.04 LTS"
    "ubuntu20:Ubuntu 20.04 LTS"
    "centos8:CentOS 8 Stream"
    "fedora38:Fedora 38"
    "alpine:Alpine Linux 3.18"
    "amazonlinux2:Amazon Linux 2"
    "debian11:Debian 11"
)

# Results tracking
TOTAL_ENVIRONMENTS=${#ENVIRONMENTS[@]}
PASSED_ENVIRONMENTS=0
FAILED_ENVIRONMENTS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$SUMMARY_LOG"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$SUMMARY_LOG"
    ((PASSED_ENVIRONMENTS++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$SUMMARY_LOG"
    ((FAILED_ENVIRONMENTS++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$SUMMARY_LOG"
}

# Function to check Docker availability
check_docker() {
    log_info "Checking Docker availability..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        echo "Installation instructions: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running. Please start Docker first."
        exit 1
    fi
    
    log_success "Docker is available and running"
}

# Function to check Docker Compose availability
check_docker_compose() {
    log_info "Checking Docker Compose availability..."
    
    if command -v docker-compose &> /dev/null; then
        log_success "Docker Compose available (standalone)"
    elif docker compose version &> /dev/null; then
        log_success "Docker Compose available (plugin)"
        # Use the plugin version
        DOCKER_COMPOSE_CMD="docker compose"
    else
        log_error "Docker Compose not found. Please install Docker Compose."
        echo "Installation instructions: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Default to standalone if available
    DOCKER_COMPOSE_CMD=${DOCKER_COMPOSE_CMD:-"docker-compose"}
}

# Function to run test in a specific environment
run_environment_test() {
    local env_name="$1"
    local env_description="$2"
    local service_name="test-$env_name"
    
    log_info "Testing environment: $env_description"
    
    # Build and run the container
    cd "$SCRIPT_DIR"
    
    if $DOCKER_COMPOSE_CMD build "$service_name" && \
       $DOCKER_COMPOSE_CMD run --rm "$service_name"; then
        log_success "$env_description - All tests passed"
        return 0
    else
        log_error "$env_description - Tests failed"
        return 1
    fi
}

# Function to cleanup Docker resources
cleanup_docker() {
    log_info "Cleaning up Docker resources..."
    
    cd "$SCRIPT_DIR"
    
    # Stop and remove containers
    $DOCKER_COMPOSE_CMD down --remove-orphans 2>/dev/null || true
    
    # Remove dangling images
    docker image prune -f 2>/dev/null || true
    
    log_info "Docker cleanup completed"
}

# Function to run all environment tests
run_all_tests() {
    log_info "Starting multi-platform testing across $TOTAL_ENVIRONMENTS environments"
    echo ""
    
    for env_info in "${ENVIRONMENTS[@]}"; do
        IFS=':' read -r env_name env_description <<< "$env_info"
        
        echo ""
        echo "🧪 Testing: $env_description"
        echo "----------------------------------------"
        
        if run_environment_test "$env_name" "$env_description"; then
            echo "✅ $env_description completed successfully"
        else
            echo "❌ $env_description failed"
            log_warning "Continuing with remaining environments..."
        fi
        
        echo ""
    done
}

# Function to generate detailed summary
generate_summary() {
    echo "" >> "$SUMMARY_LOG"
    echo "Test Summary" >> "$SUMMARY_LOG"
    echo "============" >> "$SUMMARY_LOG"
    echo "Total environments tested: $TOTAL_ENVIRONMENTS" >> "$SUMMARY_LOG"
    echo "Environments passed: $PASSED_ENVIRONMENTS" >> "$SUMMARY_LOG"
    echo "Environments failed: $FAILED_ENVIRONMENTS" >> "$SUMMARY_LOG"
    echo "Success rate: $(( PASSED_ENVIRONMENTS * 100 / TOTAL_ENVIRONMENTS ))%" >> "$SUMMARY_LOG"
    echo "" >> "$SUMMARY_LOG"
    
    if [[ $FAILED_ENVIRONMENTS -eq 0 ]]; then
        echo "✅ All environments passed!" >> "$SUMMARY_LOG"
    else
        echo "⚠️ Some environments failed. Check individual logs for details." >> "$SUMMARY_LOG"
    fi
    
    echo "Test completed at: $(date)" >> "$SUMMARY_LOG"
}

# Function to display final results
display_results() {
    echo ""
    echo "🏁 Multi-Platform Testing Complete"
    echo "=================================="
    log_info "Total environments tested: $TOTAL_ENVIRONMENTS"
    log_success "Environments passed: $PASSED_ENVIRONMENTS"
    
    if [[ $FAILED_ENVIRONMENTS -gt 0 ]]; then
        log_error "Environments failed: $FAILED_ENVIRONMENTS"
        echo ""
        echo "❌ Some environments failed. Success rate: $(( PASSED_ENVIRONMENTS * 100 / TOTAL_ENVIRONMENTS ))%"
    else
        echo ""
        echo "🎉 All environments passed! Your project is compatible across all tested platforms."
    fi
    
    echo ""
    echo "📋 Detailed test results available at: $SUMMARY_LOG"
    echo "📁 Individual test logs in: $TEST_RESULTS_DIR"
    echo ""
    echo "💡 Next Steps:"
    echo "   • Review any failed environment logs for specific issues"
    echo "   • Update scripts based on compatibility findings"
    echo "   • Run tests again after making improvements"
}

# Main execution
main() {
    echo ""
    log_info "Starting Docker multi-platform testing at $(date)"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Test results directory: $TEST_RESULTS_DIR"
    echo ""
    
    # Pre-flight checks
    check_docker
    check_docker_compose
    echo ""
    
    # Run tests
    run_all_tests
    
    # Generate summary
    generate_summary
    
    # Display results
    display_results
    
    # Cleanup
    cleanup_docker
    
    # Exit with appropriate code
    if [[ $FAILED_ENVIRONMENTS -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# Handle script interruption
trap cleanup_docker EXIT

# Execute main function
main "$@"
