#!/bin/bash

# OpenNLP GPU - Universal Environment Setup Script
# Automatically detects and configures the environment for GPU acceleration

set -e

echo "🌐 OpenNLP GPU - Universal Environment Setup"
echo "============================================"

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
    echo "📦 Installing Java 17..."
    
    case $os in
        linux)
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y openjdk-17-jdk
            elif command -v yum &> /dev/null; then
                sudo yum install -y java-17-openjdk-devel
            elif command -v dnf &> /dev/null; then
                sudo dnf install -y java-17-openjdk-devel
            else
                echo "❌ Unsupported Linux distribution for automatic Java installation"
                return 1
            fi
            ;;
        macos)
            if command -v brew &> /dev/null; then
                brew install openjdk@17
            else
                echo "❌ Homebrew not found. Please install Java 17 manually."
                return 1
            fi
            ;;
        windows)
            echo "❌ Automatic Java installation not supported on Windows"
            echo "Please download and install Java 17 from: https://adoptium.net/"
            return 1
            ;;
        *)
            echo "❌ Unsupported operating system for automatic Java installation"
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
        local os=$(detect_os)
        
        case $os in
            linux)
                if command -v apt-get &> /dev/null; then
                    sudo apt-get install -y maven
                elif command -v yum &> /dev/null; then
                    sudo yum install -y maven
                elif command -v dnf &> /dev/null; then
                    sudo dnf install -y maven
                else
                    echo "❌ Unsupported Linux distribution for automatic Maven installation"
                    return 1
                fi
                ;;
            macos)
                if command -v brew &> /dev/null; then
                    brew install maven
                else
                    echo "❌ Homebrew not found. Please install Maven manually."
                    return 1
                fi
                ;;
            *)
                echo "❌ Automatic Maven installation not supported on this OS"
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
    
    echo "🖥️ Detected OS: $os"
    echo "🏗️ Detected Architecture: $arch"
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
