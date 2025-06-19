#!/bin/bash

# Cross-Platform Script Testing Framework for OpenNLP GPU
# Tests all scripts for compatibility across Linux, macOS, and Windows (via WSL/GitBash)

set -e

echo "🧪 OpenNLP GPU - Cross-Platform Script Compatibility Tests"
echo "=========================================================="
echo

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_RESULTS_DIR="${SCRIPT_DIR}/../test-output/script-tests"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="${TEST_RESULTS_DIR}/compatibility_test_${TIMESTAMP}.log"

# Create test results directory
mkdir -p "$TEST_RESULTS_DIR"

# Initialize test log
echo "Cross-Platform Compatibility Test - $(date)" > "$TEST_LOG"
echo "=============================================" >> "$TEST_LOG"
echo >> "$TEST_LOG"

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Function to detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to detect architecture
detect_arch() {
    local arch=$(uname -m)
    case $arch in
        x86_64|amd64)
            echo "x86_64"
            ;;
        arm64|aarch64)
            echo "arm64"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Function to log test result
log_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    
    echo "[$status] $test_name: $details" | tee -a "$TEST_LOG"
    
    case $status in
        PASS)
            ((PASSED_TESTS++))
            ;;
        FAIL)
            ((FAILED_TESTS++))
            ;;
        SKIP)
            ((SKIPPED_TESTS++))
            ;;
    esac
    ((TOTAL_TESTS++))
}

# Function to test script syntax
test_script_syntax() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    echo "🔍 Testing syntax: $script_name"
    
    if [[ ! -f "$script_path" ]]; then
        log_result "Syntax-$script_name" "FAIL" "Script file not found"
        return 1
    fi
    
    if bash -n "$script_path" 2>/dev/null; then
        log_result "Syntax-$script_name" "PASS" "Valid bash syntax"
        return 0
    else
        log_result "Syntax-$script_name" "FAIL" "Invalid bash syntax"
        return 1
    fi
}

# Function to test script executability
test_script_executable() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    echo "🔍 Testing executability: $script_name"
    
    if [[ -x "$script_path" ]]; then
        log_result "Executable-$script_name" "PASS" "Script is executable"
        return 0
    else
        log_result "Executable-$script_name" "FAIL" "Script is not executable"
        return 1
    fi
}

# Function to test platform-specific commands
test_platform_commands() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    local os=$(detect_os)
    
    echo "🔍 Testing platform commands: $script_name on $os"
    
    # Check for Linux-specific commands that might not work on other platforms
    local linux_only_commands=("apt-get" "yum" "dnf" "systemctl" "lspci" "nvidia-smi")
    local macos_only_commands=("brew" "system_profiler")
    local windows_only_commands=("wmic" "reg")
    
    local issues_found=0
    
    # Read script content
    local script_content=$(cat "$script_path")
    
    case $os in
        linux)
            # Check for macOS/Windows commands on Linux
            for cmd in "${macos_only_commands[@]}" "${windows_only_commands[@]}"; do
                if echo "$script_content" | grep -q "\b$cmd\b" && ! echo "$script_content" | grep -q "command -v $cmd"; then
                    log_result "PlatformCmd-$script_name" "FAIL" "Uses $cmd without availability check on Linux"
                    ((issues_found++))
                fi
            done
            ;;
        macos)
            # Check for Linux-specific commands on macOS
            for cmd in "${linux_only_commands[@]}"; do
                if echo "$script_content" | grep -q "\b$cmd\b" && ! echo "$script_content" | grep -q "command -v $cmd"; then
                    log_result "PlatformCmd-$script_name" "FAIL" "Uses $cmd without availability check on macOS"
                    ((issues_found++))
                fi
            done
            ;;
        windows)
            # Check for Linux/macOS commands on Windows
            for cmd in "${linux_only_commands[@]}" "${macos_only_commands[@]}"; do
                if echo "$script_content" | grep -q "\b$cmd\b" && ! echo "$script_content" | grep -q "command -v $cmd"; then
                    log_result "PlatformCmd-$script_name" "FAIL" "Uses $cmd without availability check on Windows"
                    ((issues_found++))
                fi
            done
            ;;
    esac
    
    if [[ $issues_found -eq 0 ]]; then
        log_result "PlatformCmd-$script_name" "PASS" "No platform-specific command issues found"
        return 0
    else
        return 1
    fi
}

# Function to test environment variable handling
test_environment_variables() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    echo "🔍 Testing environment variables: $script_name"
    
    local script_content=$(cat "$script_path")
    local issues_found=0
    
    # Check for common environment variable issues
    if echo "$script_content" | grep -q '\$HOME' && ! echo "$script_content" | grep -q '${HOME}'; then
        # This is actually fine, just checking patterns
        :
    fi
    
    # Check for PATH modifications without proper quoting
    if echo "$script_content" | grep -q 'PATH=' && ! echo "$script_content" | grep -q '".*PATH.*"'; then
        log_result "EnvVar-$script_name" "FAIL" "PATH modification without proper quoting"
        ((issues_found++))
    fi
    
    if [[ $issues_found -eq 0 ]]; then
        log_result "EnvVar-$script_name" "PASS" "Environment variable handling looks good"
        return 0
    else
        return 1
    fi
}

# Function to test help/usage functionality
test_help_functionality() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    echo "🔍 Testing help functionality: $script_name"
    
    # Test if script responds to --help or -h
    local help_exit_code=0
    timeout 10s bash "$script_path" --help &>/dev/null || help_exit_code=$?
    
    if [[ $help_exit_code -eq 124 ]]; then
        log_result "Help-$script_name" "FAIL" "Script hangs on --help (timeout)"
        return 1
    elif [[ $help_exit_code -eq 0 ]]; then
        log_result "Help-$script_name" "PASS" "Script responds to --help"
        return 0
    else
        # Try -h
        timeout 10s bash "$script_path" -h &>/dev/null || help_exit_code=$?
        if [[ $help_exit_code -eq 0 ]]; then
            log_result "Help-$script_name" "PASS" "Script responds to -h"
            return 0
        else
            log_result "Help-$script_name" "SKIP" "No help functionality detected"
            return 0
        fi
    fi
}

# Function to run dry-run tests
test_dry_run() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    echo "🔍 Testing dry run: $script_name"
    
    # Some scripts might support dry-run mode
    local script_content=$(cat "$script_path")
    
    if echo "$script_content" | grep -q "dry.run\|DRY_RUN\|--dry-run"; then
        log_result "DryRun-$script_name" "PASS" "Script supports dry-run mode"
        return 0
    else
        log_result "DryRun-$script_name" "SKIP" "No dry-run support detected"
        return 0
    fi
}

# Main test execution
main() {
    local os=$(detect_os)
    local arch=$(detect_arch)
    
    echo "🖥️ Testing on: $os ($arch)"
    echo "📝 Test log: $TEST_LOG"
    echo
    
    # List of scripts to test (relative to script directory)
    local scripts_to_test=(
        "check_gpu_prerequisites.sh"
        "setup_universal_environment.sh"
        "run_all_demos.sh"
        "setup_aws_gpu_environment.sh"
        "validate_java_runtime.sh"
        "fix_java_environment.sh"
        "check_ide_setup.sh"
    )
    
    echo "🧪 Starting comprehensive script tests..."
    echo
    
    for script in "${scripts_to_test[@]}"; do
        local script_path="${SCRIPT_DIR}/$script"
        
        echo "========================================"
        echo "Testing: $script"
        echo "========================================"
        
        # Run all test functions
        test_script_syntax "$script_path"
        test_script_executable "$script_path"
        test_platform_commands "$script_path"
        test_environment_variables "$script_path"
        test_help_functionality "$script_path"
        test_dry_run "$script_path"
        
        echo
    done
    
    # Generate summary
    echo "📊 Test Summary"
    echo "==============="
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Skipped: $SKIPPED_TESTS"
    echo
    
    # Write summary to log
    {
        echo
        echo "Test Summary:"
        echo "============="
        echo "Total Tests: $TOTAL_TESTS"
        echo "Passed: $PASSED_TESTS"
        echo "Failed: $FAILED_TESTS"
        echo "Skipped: $SKIPPED_TESTS"
        echo "Success Rate: $(echo "scale=2; $PASSED_TESTS * 100 / ($TOTAL_TESTS - $SKIPPED_TESTS)" | bc -l)%"
    } >> "$TEST_LOG"
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        echo "🎉 All tests passed! Scripts are cross-platform compatible."
        exit 0
    else
        echo "❌ Some tests failed. Check the log for details: $TEST_LOG"
        exit 1
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

# Run main function
main "$@"
