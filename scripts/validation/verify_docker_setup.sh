#!/bin/bash

# Quick Docker Setup Verification for OpenNLP GPU
# Tests that Docker testing infrastructure is properly configured

set -e

echo "🐳 Docker Testing Infrastructure Verification"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_ROOT/docker"

CHECKS_PASSED=0
CHECKS_FAILED=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((CHECKS_PASSED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((CHECKS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check Docker installation
check_docker_installation() {
    log_info "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        log_success "Docker command available"
        
        # Check Docker version
        local docker_version=$(docker --version 2>/dev/null || echo "unknown")
        log_info "Docker version: $docker_version"
        
        # Check if Docker daemon is running (simplified check)
        if docker ps &> /dev/null; then
            log_success "Docker daemon is running"
        else
            log_warning "Docker daemon may not be running or permission denied"
            log_info "Try: sudo systemctl start docker"
            log_info "Or add user to docker group: sudo usermod -aG docker \$USER"
        fi
    else
        log_error "Docker not found"
        log_info "Install Docker: https://docs.docker.com/get-docker/"
    fi
}

# Check Docker Compose
check_docker_compose() {
    log_info "Checking Docker Compose..."
    
    if command -v docker-compose &> /dev/null; then
        log_success "Docker Compose (standalone) available"
        local compose_version=$(docker-compose --version 2>/dev/null || echo "unknown")
        log_info "Version: $compose_version"
    elif docker compose version &> /dev/null 2>&1; then
        log_success "Docker Compose (plugin) available"
        local compose_version=$(docker compose version 2>/dev/null || echo "unknown")
        log_info "Version: $compose_version"
    else
        log_error "Docker Compose not found"
        log_info "Install Docker Compose: https://docs.docker.com/compose/install/"
    fi
}

# Check Docker files exist
check_docker_files() {
    log_info "Checking Docker configuration files..."
    
    local dockerfiles=(
        "test-ubuntu22.04.Dockerfile"
        "test-ubuntu20.04.Dockerfile"
        "test-centos8.Dockerfile"
        "test-fedora38.Dockerfile"
        "test-alpine.Dockerfile"
        "test-amazonlinux2.Dockerfile"
        "test-debian11.Dockerfile"
    )
    
    for dockerfile in "${dockerfiles[@]}"; do
        local file_path="$DOCKER_DIR/$dockerfile"
        if [[ -f "$file_path" ]]; then
            log_success "Dockerfile exists: $dockerfile"
        else
            log_error "Dockerfile missing: $dockerfile"
        fi
    done
    
    # Check docker-compose.yml
    if [[ -f "$DOCKER_DIR/docker-compose.yml" ]]; then
        log_success "Docker Compose configuration exists"
    else
        log_error "Docker Compose configuration missing"
    fi
    
    # Check README
    if [[ -f "$DOCKER_DIR/README.md" ]]; then
        log_success "Docker documentation exists"
    else
        log_error "Docker documentation missing"
    fi
}

# Check test scripts
check_test_scripts() {
    log_info "Checking test execution scripts..."
    
    local test_scripts=(
        "run_docker_tests.sh"
        "test_cross_platform_compatibility.sh"
    )
    
    for script in "${test_scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        if [[ -f "$script_path" ]]; then
            log_success "Test script exists: $script"
            
            if [[ -x "$script_path" ]]; then
                log_success "Test script executable: $script"
            else
                log_error "Test script not executable: $script"
                log_info "Fix with: chmod +x $script_path"
            fi
        else
            log_error "Test script missing: $script"
        fi
    done
}

# Check project structure
check_project_structure() {
    log_info "Checking project structure for Docker testing..."
    
    # Check essential project files
    local essential_files=(
        "pom.xml"
        "README.md"
        "scripts/cross_platform_lib.sh"
        "scripts/check_gpu_prerequisites.sh"
        "scripts/setup_universal_environment.sh"
    )
    
    for file in "${essential_files[@]}"; do
        local file_path="$PROJECT_ROOT/$file"
        if [[ -f "$file_path" ]]; then
            log_success "Essential file exists: $file"
        else
            log_error "Essential file missing: $file"
        fi
    done
    
    # Check test output directory
    local test_output_dir="$PROJECT_ROOT/test-output"
    if [[ -d "$test_output_dir" ]]; then
        log_success "Test output directory exists"
    else
        log_info "Creating test output directory..."
        mkdir -p "$test_output_dir"
        log_success "Test output directory created"
    fi
}

# Test Docker Compose syntax
test_docker_compose_syntax() {
    log_info "Testing Docker Compose configuration syntax..."
    
    cd "$DOCKER_DIR"
    
    if docker-compose config &> /dev/null || docker compose config &> /dev/null; then
        log_success "Docker Compose configuration syntax is valid"
    else
        log_error "Docker Compose configuration has syntax errors"
        log_info "Check docker-compose.yml for errors"
    fi
}

# Test basic Docker functionality
test_docker_basic() {
    log_info "Testing basic Docker functionality..."
    
    # Test simple container run with timeout
    if timeout 10 docker run --rm hello-world &> /dev/null; then
        log_success "Basic Docker container execution works"
    else
        log_warning "Basic Docker test failed (may need Docker daemon restart)"
    fi
}

# Estimate disk space requirements
check_disk_space() {
    log_info "Checking available disk space..."
    
    local available_space_kb=$(df "$PROJECT_ROOT" | tail -n 1 | awk '{print $4}')
    local available_space_gb=$((available_space_kb / 1024 / 1024))
    
    log_info "Available disk space: ${available_space_gb}GB"
    
    if [[ $available_space_gb -ge 4 ]]; then
        log_success "Sufficient disk space for Docker testing (${available_space_gb}GB >= 4GB)"
    else
        log_warning "Limited disk space (${available_space_gb}GB < 4GB recommended)"
        log_info "Consider cleaning up with: docker system prune -a"
    fi
}

# Generate verification report
generate_report() {
    echo ""
    echo "🏁 Docker Setup Verification Summary"
    echo "===================================="
    echo ""
    
    log_info "Checks passed: $CHECKS_PASSED"
    if [[ $CHECKS_FAILED -gt 0 ]]; then
        log_error "Checks failed: $CHECKS_FAILED"
    fi
    
    echo ""
    
    if [[ $CHECKS_FAILED -eq 0 ]]; then
        echo "✅ Docker testing infrastructure is ready!"
        echo ""
        echo "🚀 Next steps:"
        echo "   • Run a single environment test:"
        echo "     cd docker/ && docker-compose run --rm test-ubuntu22"
        echo ""
        echo "   • Run all environment tests:"
        echo "     ./scripts/run_docker_tests.sh"
        echo ""
        echo "   • View Docker documentation:"
        echo "     cat docker/README.md"
    else
        echo "❌ Docker testing infrastructure needs attention"
        echo ""
        echo "🔧 Fix the failed checks above, then run this script again"
        echo ""
        echo "💡 Common fixes:"
        echo "   • Install Docker: https://docs.docker.com/get-docker/"
        echo "   • Start Docker daemon: sudo systemctl start docker"
        echo "   • Add user to docker group: sudo usermod -aG docker \$USER"
        echo "   • Install Docker Compose: https://docs.docker.com/compose/install/"
    fi
    
    echo ""
    echo "📁 Docker files location: $DOCKER_DIR"
    echo "📁 Test scripts location: $SCRIPT_DIR"
}

# Main execution
main() {
    echo ""
    log_info "Verifying Docker testing setup for OpenNLP GPU project"
    log_info "Project root: $PROJECT_ROOT"
    echo ""
    
    check_docker_installation
    echo ""
    
    check_docker_compose
    echo ""
    
    check_docker_files
    echo ""
    
    check_test_scripts
    echo ""
    
    check_project_structure
    echo ""
    
    test_docker_compose_syntax
    echo ""
    
    test_docker_basic
    echo ""
    
    check_disk_space
    echo ""
    
    generate_report
    
    # Exit with appropriate code
    if [[ $CHECKS_FAILED -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# Execute main function
main "$@"
