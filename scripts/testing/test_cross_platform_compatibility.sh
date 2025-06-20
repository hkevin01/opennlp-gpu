#!/bin/bash

# Cross-Platform Compatibility Test Suite for OpenNLP GPU Scripts
# Tests all major scripts across different environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_LOG="$SCRIPT_DIR/../test-output/cross_platform_test_$(date +%Y%m%d_%H%M%S).log"
FAILED_TESTS=0
PASSED_TESTS=0
SKIPPED_TESTS=0
TOTAL_TESTS=0

# Ensure test output directory exists
mkdir -p "$(dirname "$TEST_LOG")"

# Source cross-platform library
source "$SCRIPT_DIR/cross_platform_lib.sh"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$TEST_LOG"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$TEST_LOG"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$TEST_LOG"
    ((FAILED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$TEST_LOG"
}

# Test helper functions
assert_command_exists() {
    local cmd="$1"
    local description="$2"
    
    if command -v "$cmd" &> /dev/null; then
        log_success "Command '$cmd' exists: $description"
        return 0
    else
        log_error "Command '$cmd' not found: $description"
        return 1
    fi
}

assert_file_exists() {
    local file="$1"
    local description="$2"
    
    if [[ -f "$file" ]]; then
        log_success "File exists: $file ($description)"
        return 0
    else
        log_error "File missing: $file ($description)"
        return 1
    fi
}

assert_script_executable() {
    local script="$1"
    local description="$2"
    
    if [[ -x "$script" ]]; then
        log_success "Script executable: $script ($description)"
        return 0
    else
        log_error "Script not executable: $script ($description)"
        return 1
    fi
}

test_script_syntax() {
    local script="$1"
    local description="$2"
    
    if bash -n "$script" 2>/dev/null; then
        log_success "Script syntax valid: $script ($description)"
        return 0
    else
        log_error "Script syntax error: $script ($description)"
        return 1
    fi
}

# Platform detection tests
test_platform_detection() {
    log_info "Testing platform detection functions..."
    
    local os=$(detect_os)
    local arch=$(detect_arch)
    local distro=$(detect_distro)
    local pm=$(detect_package_manager)
    
    if [[ "$os" != "unknown" ]]; then
        log_success "OS detection: $os"
    else
        log_error "OS detection failed"
    fi
    
    if [[ "$arch" != "unknown" ]]; then
        log_success "Architecture detection: $arch"
    else
        log_error "Architecture detection failed"
    fi
    
    if [[ "$distro" != "unknown" ]]; then
        log_success "Distribution detection: $distro"
    else
        log_warning "Distribution detection returned unknown (may be expected on some systems)"
    fi
    
    if [[ "$pm" != "unknown" ]]; then
        log_success "Package manager detection: $pm"
    else
        log_warning "Package manager detection returned unknown (may be expected on some systems)"
    fi
}

# Cross-platform utility tests
test_cross_platform_utilities() {
    log_info "Testing cross-platform utility functions..."
    
    # Test CPU count detection
    local cpu_count=$(xp_get_cpu_count)
    if [[ "$cpu_count" =~ ^[0-9]+$ ]] && [[ "$cpu_count" -gt 0 ]]; then
        log_success "CPU count detection: $cpu_count cores"
    else
        log_error "CPU count detection failed: $cpu_count"
    fi
    
    # Test memory detection
    local memory_gb=$(xp_get_memory_gb)
    if [[ "$memory_gb" =~ ^[0-9]+$ ]] && [[ "$memory_gb" -gt 0 ]]; then
        log_success "Memory detection: ${memory_gb}GB"
    elif [[ "$memory_gb" == "unknown" ]]; then
        log_warning "Memory detection returned unknown (may be expected on some systems)"
    else
        log_error "Memory detection failed: $memory_gb"
    fi
    
    # Test path separator
    local sep=$(xp_path_separator)
    if [[ "$sep" == ":" ]] || [[ "$sep" == ";" ]]; then
        log_success "Path separator detection: '$sep'"
    else
        log_error "Path separator detection failed: '$sep'"
    fi
    
    # Test temp directory
    local temp_dir=$(xp_get_temp_dir)
    if [[ -d "$temp_dir" ]]; then
        log_success "Temp directory detection: $temp_dir"
    else
        log_error "Temp directory detection failed: $temp_dir"
    fi
    
    # Test home directory
    local home_dir=$(xp_get_home_dir)
    if [[ -d "$home_dir" ]]; then
        log_success "Home directory detection: $home_dir"
    else
        log_error "Home directory detection failed: $home_dir"
    fi
}

# Script existence and syntax tests
test_script_files() {
    log_info "Testing script files existence and syntax..."
    
    local scripts=(
        "check_gpu_prerequisites.sh"
        "setup_universal_environment.sh"
        "run_all_demos.sh"
        "setup_aws_gpu_environment.sh"
        "cross_platform_lib.sh"
        "validate_java_runtime.sh"
        "check_ide_setup.sh"
        "fix_java_environment.sh"
    )
    
    for script in "${scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        
        assert_file_exists "$script_path" "Core script"
        assert_script_executable "$script_path" "Core script"
        test_script_syntax "$script_path" "Core script"
    done
}

# Java environment tests
test_java_environment() {
    log_info "Testing Java environment detection..."
    
    if command -v java &> /dev/null; then
        local java_version=$(java -version 2>&1 | head -n 1)
        log_success "Java found: $java_version"
        
        # Test Java version parsing
        local version_num=$(java -version 2>&1 | grep version | cut -d'"' -f2 | cut -d'.' -f1)
        if [[ "$version_num" =~ ^[0-9]+$ ]]; then
            log_success "Java version parsing: $version_num"
        else
            log_error "Java version parsing failed: $version_num"
        fi
    else
        log_warning "Java not found (may be expected in test environment)"
    fi
    
    if [[ -n "$JAVA_HOME" ]]; then
        if [[ -d "$JAVA_HOME" ]]; then
            log_success "JAVA_HOME set and valid: $JAVA_HOME"
        else
            log_error "JAVA_HOME set but invalid: $JAVA_HOME"
        fi
    else
        log_warning "JAVA_HOME not set (may be expected)"
    fi
}

# Maven environment tests
test_maven_environment() {
    log_info "Testing Maven environment..."
    
    if command -v mvn &> /dev/null; then
        local maven_version=$(mvn -version 2>/dev/null | head -n 1)
        log_success "Maven found: $maven_version"
    else
        log_warning "Maven not found (may be expected in test environment)"
    fi
}

# GPU detection tests (non-invasive)
test_gpu_detection() {
    log_info "Testing GPU detection capabilities..."
    
    # Test NVIDIA detection commands
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA tools available"
    else
        log_info "NVIDIA tools not available (expected on non-NVIDIA systems)"
    fi
    
    # Test AMD detection commands
    if command -v rocm-smi &> /dev/null; then
        log_success "AMD ROCm tools available"
    else
        log_info "AMD ROCm tools not available (expected on non-AMD systems)"
    fi
    
    # Test Intel detection commands
    if command -v intel_gpu_top &> /dev/null; then
        log_success "Intel GPU tools available"
    else
        log_info "Intel GPU tools not available (expected on non-Intel GPU systems)"
    fi
    
    # Test OpenCL detection
    if command -v clinfo &> /dev/null; then
        log_success "OpenCL tools available"
    else
        log_info "OpenCL tools not available (may be expected)"
    fi
    
    # Test lspci for hardware detection
    if command -v lspci &> /dev/null; then
        log_success "lspci available for hardware detection"
    else
        log_warning "lspci not available (may limit GPU detection)"
    fi
}

# Network connectivity tests
test_network_connectivity() {
    log_info "Testing network connectivity for package downloads..."
    
    # Test common package repository connectivity
    local test_urls=(
        "https://repo1.maven.org"
        "https://github.com"
    )
    
    for url in "${test_urls[@]}"; do
        if command -v curl &> /dev/null; then
            if curl -s --head --connect-timeout 5 "$url" > /dev/null; then
                log_success "Network connectivity: $url"
            else
                log_warning "Network connectivity failed: $url (may be expected in restricted environments)"
            fi
        elif command -v wget &> /dev/null; then
            if wget --spider --timeout=5 "$url" &> /dev/null; then
                log_success "Network connectivity: $url"
            else
                log_warning "Network connectivity failed: $url (may be expected in restricted environments)"
            fi
        else
            log_warning "No network testing tools available (curl/wget)"
            break
        fi
    done
}

# File system permissions tests
test_file_permissions() {
    log_info "Testing file system permissions..."
    
    local temp_dir=$(xp_get_temp_dir)
    local test_file="$temp_dir/opennlp_gpu_test_$$"
    
    # Test write permissions
    if echo "test" > "$test_file" 2>/dev/null; then
        log_success "Temp directory write permissions"
        rm -f "$test_file"
    else
        log_error "Temp directory write permissions failed"
    fi
    
    # Test script directory permissions
    if [[ -r "$SCRIPT_DIR" ]]; then
        log_success "Script directory read permissions"
    else
        log_error "Script directory read permissions failed"
    fi
}

# Simulate different platform scenarios
test_platform_scenarios() {
    log_info "Testing platform-specific scenarios..."
    
    local os=$(detect_os)
    
    case $os in
        linux)
            log_info "Testing Linux-specific features..."
            
            # Test systemd availability
            if command -v systemctl &> /dev/null; then
                log_success "systemctl available"
            else
                log_info "systemctl not available (older init system)"
            fi
            
            # Test package managers
            if command -v apt-get &> /dev/null; then
                log_success "apt package manager available"
            elif command -v yum &> /dev/null; then
                log_success "yum package manager available"
            elif command -v dnf &> /dev/null; then
                log_success "dnf package manager available"
            else
                log_warning "No recognized package manager found"
            fi
            ;;
        macos)
            log_info "Testing macOS-specific features..."
            
            # Test Homebrew
            if command -v brew &> /dev/null; then
                log_success "Homebrew available"
            else
                log_warning "Homebrew not available"
            fi
            
            # Test system_profiler
            if command -v system_profiler &> /dev/null; then
                log_success "system_profiler available"
            else
                log_error "system_profiler not available (unexpected on macOS)"
            fi
            ;;
        windows)
            log_info "Testing Windows-specific features..."
            
            # Test Windows package managers
            if command -v choco &> /dev/null; then
                log_success "Chocolatey available"
            elif command -v winget &> /dev/null; then
                log_success "winget available"
            else
                log_warning "No Windows package manager found"
            fi
            ;;
        *)
            log_warning "Unknown operating system: $os"
            ;;
    esac
}

# Main test execution
main() {
    echo "🧪 OpenNLP GPU Cross-Platform Compatibility Test Suite"
    echo "====================================================="
    echo ""
    
    log_info "Starting test execution at $(date)"
    log_info "Platform: $(detect_os) $(detect_arch) - $(detect_distro)"
    log_info "Test log: $TEST_LOG"
    echo ""
    
    # Run all test suites
    test_platform_detection
    echo ""
    
    test_cross_platform_utilities
    echo ""
    
    test_script_files
    echo ""
    
    test_java_environment
    echo ""
    
    test_maven_environment
    echo ""
    
    test_gpu_detection
    echo ""
    
    test_network_connectivity
    echo ""
    
    test_file_permissions
    echo ""
    
    test_platform_scenarios
    echo ""
    
    # Summary
    echo "🏁 Test Summary"
    echo "==============="
    log_info "Tests completed at $(date)"
    log_success "Passed tests: $PASSED_TESTS"
    
    if [[ $FAILED_TESTS -gt 0 ]]; then
        log_error "Failed tests: $FAILED_TESTS"
        echo ""
        echo "❌ Some tests failed. Check the log for details: $TEST_LOG"
        exit 1
    else
        echo ""
        echo "✅ All tests passed! Scripts are ready for cross-platform use."
        echo "📋 Full test log available at: $TEST_LOG"
    fi
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Cross-platform compatibility testing for OpenNLP GPU scripts"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verbose  Enable verbose output"
    echo "  -l, --log-only Write results only to log file"
    echo
    echo "Examples:"
    echo "  $0                    # Run all tests with normal output"
    echo "  $0 --verbose          # Run all tests with detailed output"
    echo "  $0 --log-only         # Run tests quietly, results in log only"
}

# Command line argument handling
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        -l|--log-only)
            exec > "$TEST_LOG" 2>&1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"
