#!/bin/bash

# OpenNLP GPU - Universal Environment Setup Script
# Automatically detects and configures the environment for GPU acceleration

set -e

echo "🌐 OpenNLP GPU - Universal Environment Setup"
echo "============================================"

# Source cross-platform compatibility library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cross_platform_lib.sh"

# Function to check Java installation
check_java() {
    echo "☕ Checking Java installation..."
    
    if command -v java &> /dev/null; then
        local java_version=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
        if [[ "$java_version" =~ ^1\. ]]; then
            java_version=$(echo $java_version | cut -d'.' -f2)
        fi
        
        if [[ $java_version -ge 11 ]]; then
            echo "✅ Java $java_version found - Compatible"
            return 0
        else
            echo "❌ Java $java_version found - Need Java 11+"
            return 1
        fi
    else
        echo "❌ Java not found"
        return 1
    fi
}

# Function to install Java if needed
install_java() {
    local os=$(detect_os)
    local pm=$(detect_package_manager)
    echo "📦 Installing Java 17..."
    
    case $pm in
        apt)
            sudo apt-get update
            sudo apt-get install -y openjdk-17-jdk
            ;;
        dnf)
            sudo dnf install -y java-17-openjdk-devel
            ;;
        yum)
            sudo yum install -y java-17-openjdk-devel
            ;;
        zypper)
            sudo zypper install -y java-17-openjdk-devel
            ;;
        pacman)
            sudo pacman -S --noconfirm jdk17-openjdk
            ;;
        brew)
            brew install openjdk@17
            ;;
        choco)
            choco install -y openjdk17
            ;;
        winget)
            winget install Microsoft.OpenJDK.17
            ;;
        *)
            echo "❌ Unsupported package manager: $pm"
            echo "Please install Java 17 manually from: https://adoptium.net/"
            return 1
            ;;
    esac
}

# Function to setup GPU environment
setup_gpu_environment() {
    echo "🔧 Setting up GPU environment..."
    
    # Run GPU prerequisites check
    if [[ -f "./scripts/check_gpu_prerequisites.sh" ]]; then
        echo "🔍 Running GPU prerequisites check..."
        ./scripts/check_gpu_prerequisites.sh
    else
        echo "⚠️ GPU prerequisites check script not found"
    fi
}

# Function to setup Maven environment
setup_maven() {
    echo "📦 Checking Maven installation..."
    
    if command -v mvn &> /dev/null; then
        echo "✅ Maven found"
        return 0
    else
        echo "❌ Maven not found - installing..."
        local pm=$(detect_package_manager)
        
        case $pm in
            apt)
                sudo apt-get install -y maven
                ;;
            dnf)
                sudo dnf install -y maven
                ;;
            yum)
                sudo yum install -y maven
                ;;
            zypper)
                sudo zypper install -y maven
                ;;
            pacman)
                sudo pacman -S --noconfirm maven
                ;;
            brew)
                brew install maven
                ;;
            choco)
                choco install -y maven
                ;;
            winget)
                winget install Apache.Maven
                ;;
            *)
                echo "❌ Unsupported package manager: $pm"
                echo "Please install Maven manually from: https://maven.apache.org/"
                return 1
                ;;
        esac
    fi
}

# Function to build the project
build_project() {
    echo "🔨 Building OpenNLP GPU project..."
    
    if [[ -f "pom.xml" ]]; then
        mvn clean compile
        echo "✅ Project built successfully"
    else
        echo "❌ pom.xml not found. Are you in the project root directory?"
        return 1
    fi
}

# Function to run diagnostics
run_diagnostics() {
    echo "🔍 Running system diagnostics..."
    
    if mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics" -q; then
        echo "✅ System diagnostics completed"
    else
        echo "⚠️ Diagnostics completed with warnings (this is normal if no GPU is available)"
    fi
}

# Main execution
main() {
    local os=$(detect_os)
    local arch=$(detect_arch)
    local distro=$(detect_distro)
    local pm=$(detect_package_manager)
    local cpu_count=$(xp_get_cpu_count)
    local memory_gb=$(xp_get_memory_gb)
    
    echo "🖥️ System Information:"
    echo "   OS: $os ($arch)"
    echo "   Distribution: $distro"
    echo "   Package Manager: $pm"
    echo "   CPU Cores: $cpu_count"
    echo "   Memory: ${memory_gb}GB"
    echo ""
    
    # Check and install Java if needed
    if ! check_java; then
        install_java || exit 1
    fi
    
    # Setup Maven
    setup_maven || exit 1
    
    # Setup GPU environment
    setup_gpu_environment
    
    # Build project
    build_project || exit 1
    
    # Run diagnostics
    run_diagnostics
    
    echo ""
    echo "🎉 Universal environment setup completed!"
    echo "✅ Your system is ready for OpenNLP GPU acceleration"
    echo ""
    echo "Next steps:"
    echo "  1. Run examples: ./scripts/run_all_demos.sh"
    echo "  2. Integrate with your project using the README instructions"
    echo "  3. Check GPU performance with included benchmarks"
}

# Execute main function
main "$@"
