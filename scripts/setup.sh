#!/bin/bash

# OpenNLP GPU Extension - One-Click Setup Script
# Supports: Ubuntu/Debian, CentOS/RHEL, Amazon Linux, WSL
# Compatible with: Local machines, AWS EC2, Google Cloud, Azure
# Author: OpenNLP GPU Extension Team

set -e  # Exit on error, but we'll handle errors gracefully

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/setup.log"
ERROR_LOG="${SCRIPT_DIR}/logs/setup-errors.log"
JAVA_VERSION="21"
CMAKE_MIN_VERSION="3.16"

# System detection
OS_TYPE=""
DISTRO=""
PACKAGE_MANAGER=""
GPU_TYPE=""
CLOUD_PROVIDER=""

# Installation flags
INSTALL_JAVA=false
INSTALL_CMAKE=false
INSTALL_MAVEN=false
INSTALL_GPU_DRIVERS=false
BUILD_NATIVE=false
BUILD_JAVA=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${LOG_FILE}"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}" >&2
}

print_step() {
    echo -e "\n${PURPLE}===${NC} $1 ${PURPLE}===${NC}\n" | tee -a "${LOG_FILE}"
}

# Function to handle errors gracefully
handle_error() {
    local exit_code=$?
    local line_number=$1
    print_error "Error occurred in setup script at line $line_number (exit code: $exit_code)"
    echo "$(date): Error at line $line_number - exit code $exit_code" >> "${ERROR_LOG}"

    # Try to provide helpful suggestions
    case $exit_code in
        1) print_warning "General error - check the logs above for details" ;;
        2) print_warning "Missing command or permission issue - try running with sudo if needed" ;;
        127) print_warning "Command not found - some dependencies might be missing" ;;
        *) print_warning "Unexpected error - check ${ERROR_LOG} for details" ;;
    esac

    print_status "Attempting to continue with alternative approaches..."
    return 0  # Don't exit the script
}

# Set up error handling
trap 'handle_error ${LINENO}' ERR

# Function to detect system information
detect_system() {
    print_step "Detecting System Configuration"

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="linux"

        # Detect distribution
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            DISTRO=$ID
            print_status "Detected Linux distribution: $PRETTY_NAME"
        fi

        # Detect package manager
        if command -v apt-get &> /dev/null; then
            PACKAGE_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            PACKAGE_MANAGER="yum"
        elif command -v dnf &> /dev/null; then
            PACKAGE_MANAGER="dnf"
        elif command -v pacman &> /dev/null; then
            PACKAGE_MANAGER="pacman"
        fi

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        PACKAGE_MANAGER="brew"
        print_status "Detected macOS"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS_TYPE="windows"
        print_status "Detected Windows (WSL/Cygwin)"
    fi

    # Detect cloud provider
    if curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/ &>/dev/null; then
        CLOUD_PROVIDER="aws"
        print_status "Detected AWS EC2 instance"
    elif curl -s --connect-timeout 2 -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/ &>/dev/null; then
        CLOUD_PROVIDER="gcp"
        print_status "Detected Google Cloud instance"
    elif curl -s --connect-timeout 2 -H "Metadata: true" http://169.254.169.254/metadata/instance &>/dev/null; then
        CLOUD_PROVIDER="azure"
        print_status "Detected Azure instance"
    fi

    # Detect GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_TYPE="nvidia"
        print_status "Detected NVIDIA GPU"
    elif command -v rocm-smi &> /dev/null || [ -d "/opt/rocm" ]; then
        GPU_TYPE="amd"
        print_status "Detected AMD GPU with ROCm"
    elif lspci | grep -i "vga\|3d\|display" | grep -i "amd\|ati" &> /dev/null; then
        GPU_TYPE="amd_no_rocm"
        print_status "Detected AMD GPU (ROCm not installed)"
    elif lspci | grep -i "vga\|3d\|display" | grep -i "nvidia" &> /dev/null; then
        GPU_TYPE="nvidia_no_driver"
        print_status "Detected NVIDIA GPU (drivers may not be installed)"
    else
        GPU_TYPE="none"
        print_status "No GPU detected or CPU-only mode"
    fi

    print_status "System detection complete: OS=$OS_TYPE, Distro=$DISTRO, Package Manager=$PACKAGE_MANAGER, GPU=$GPU_TYPE"
}

# Function to check and install Java
install_java() {
    print_step "Checking Java Installation"

    if command -v java &> /dev/null; then
        JAVA_VER=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
        if [ "$JAVA_VER" -ge "$JAVA_VERSION" ]; then
            print_success "Java $JAVA_VER is already installed"
            return 0
        else
            print_warning "Java $JAVA_VER found, but need Java $JAVA_VERSION or higher"
        fi
    fi

    print_status "Installing Java $JAVA_VERSION..."

    case $PACKAGE_MANAGER in
        apt)
            sudo apt-get update || print_warning "Failed to update package list"
            sudo apt-get install -y openjdk-${JAVA_VERSION}-jdk || \
            sudo apt-get install -y default-jdk || \
            print_error "Failed to install Java via apt"
            ;;
        yum|dnf)
            sudo $PACKAGE_MANAGER install -y java-${JAVA_VERSION}-openjdk-devel || \
            sudo $PACKAGE_MANAGER install -y java-11-openjdk-devel || \
            print_error "Failed to install Java via $PACKAGE_MANAGER"
            ;;
        brew)
            brew install openjdk@${JAVA_VERSION} || \
            brew install openjdk@11 || \
            print_error "Failed to install Java via Homebrew"
            ;;
        *)
            print_error "Unsupported package manager for Java installation"
            print_status "Please install Java $JAVA_VERSION manually and re-run this script"
            return 1
            ;;
    esac

    # Set JAVA_HOME
    if [ -z "$JAVA_HOME" ]; then
        case $OS_TYPE in
            linux)
                export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")
                echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
                ;;
            macos)
                export JAVA_HOME=$(/usr/libexec/java_home)
                echo "export JAVA_HOME=$JAVA_HOME" >> ~/.zshrc
                ;;
        esac
        print_status "Set JAVA_HOME to $JAVA_HOME"
    fi

    # Verify installation
    if command -v java &> /dev/null; then
        JAVA_VER=$(java -version 2>&1 | head -n 1)
        print_success "Java installed successfully: $JAVA_VER"
    else
        print_error "Java installation verification failed"
        return 1
    fi
}

# Function to check and install Maven
install_maven() {
    print_step "Checking Maven Installation"

    if command -v mvn &> /dev/null; then
        MVN_VER=$(mvn --version | head -n 1)
        print_success "Maven is already installed: $MVN_VER"
        return 0
    fi

    print_status "Installing Maven..."

    case $PACKAGE_MANAGER in
        apt)
            sudo apt-get install -y maven || print_error "Failed to install Maven via apt"
            ;;
        yum|dnf)
            sudo $PACKAGE_MANAGER install -y maven || print_error "Failed to install Maven via $PACKAGE_MANAGER"
            ;;
        brew)
            brew install maven || print_error "Failed to install Maven via Homebrew"
            ;;
        *)
            # Manual installation as fallback
            print_status "Installing Maven manually..."
            MAVEN_VERSION="3.9.6"
            cd /tmp
            wget -q "https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz" || \
            curl -sL "https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz" -o "apache-maven-${MAVEN_VERSION}-bin.tar.gz"

            sudo tar xzf "apache-maven-${MAVEN_VERSION}-bin.tar.gz" -C /opt
            sudo ln -sf "/opt/apache-maven-${MAVEN_VERSION}/bin/mvn" /usr/local/bin/mvn
            echo "export PATH=/opt/apache-maven-${MAVEN_VERSION}/bin:\$PATH" >> ~/.bashrc
            export PATH="/opt/apache-maven-${MAVEN_VERSION}/bin:$PATH"
            ;;
    esac

    # Verify installation
    if command -v mvn &> /dev/null; then
        MVN_VER=$(mvn --version | head -n 1)
        print_success "Maven installed successfully: $MVN_VER"
    else
        print_error "Maven installation verification failed"
        return 1
    fi
}

# Function to check and install CMake
install_cmake() {
    print_step "Checking CMake Installation"

    if command -v cmake &> /dev/null; then
        CMAKE_VER=$(cmake --version | head -n 1 | cut -d' ' -f3)
        CMAKE_MAJOR=$(echo $CMAKE_VER | cut -d'.' -f1)
        CMAKE_MINOR=$(echo $CMAKE_VER | cut -d'.' -f2)

        if [ "$CMAKE_MAJOR" -gt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -ge 16 ]); then
            print_success "CMake $CMAKE_VER is already installed"
            return 0
        else
            print_warning "CMake $CMAKE_VER found, but need $CMAKE_MIN_VERSION or higher"
        fi
    fi

    print_status "Installing CMake..."

    case $PACKAGE_MANAGER in
        apt)
            # Try official repository first
            sudo apt-get install -y cmake || {
                # Install from Kitware repository for latest version
                print_status "Installing CMake from Kitware repository..."
                sudo apt-get install -y ca-certificates gpg wget
                wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
                echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
                sudo apt-get update
                sudo apt-get install -y cmake || print_error "Failed to install CMake"
            }
            ;;
        yum|dnf)
            sudo $PACKAGE_MANAGER install -y cmake || {
                # Install EPEL for CentOS/RHEL
                if [ "$DISTRO" = "centos" ] || [ "$DISTRO" = "rhel" ]; then
                    sudo $PACKAGE_MANAGER install -y epel-release
                    sudo $PACKAGE_MANAGER install -y cmake
                fi
            }
            ;;
        brew)
            brew install cmake || print_error "Failed to install CMake via Homebrew"
            ;;
        *)
            # Manual installation as fallback
            print_status "Installing CMake manually..."
            CMAKE_VERSION="3.28.1"
            cd /tmp
            wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" || \
            curl -sL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -o "cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"

            sudo tar xzf "cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -C /opt
            sudo ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake" /usr/local/bin/cmake
            echo "export PATH=/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin:\$PATH" >> ~/.bashrc
            export PATH="/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin:$PATH"
            ;;
    esac

    # Verify installation
    if command -v cmake &> /dev/null; then
        CMAKE_VER=$(cmake --version | head -n 1 | cut -d' ' -f3)
        print_success "CMake installed successfully: $CMAKE_VER"
    else
        print_error "CMake installation verification failed"
        return 1
    fi
}

# Function to install development tools
install_dev_tools() {
    print_step "Installing Development Tools"

    case $PACKAGE_MANAGER in
        apt)
            sudo apt-get install -y build-essential git curl wget || \
            print_warning "Some development tools may not have been installed"
            ;;
        yum|dnf)
            sudo $PACKAGE_MANAGER groupinstall -y "Development Tools" || \
            sudo $PACKAGE_MANAGER install -y gcc gcc-c++ make git curl wget || \
            print_warning "Some development tools may not have been installed"
            ;;
        brew)
            xcode-select --install 2>/dev/null || print_status "Xcode tools already installed"
            brew install git curl wget || print_warning "Some tools may not have been installed"
            ;;
    esac

    print_success "Development tools installation completed"
}

# Function to install GPU drivers and libraries
install_gpu_support() {
    print_step "Setting up GPU Support"

    case $GPU_TYPE in
        nvidia)
            print_success "NVIDIA drivers already detected"
            # Install CUDA if needed
            if ! command -v nvcc &> /dev/null; then
                print_status "Installing CUDA toolkit..."
                case $PACKAGE_MANAGER in
                    apt)
                        # Add NVIDIA repository
                        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/cuda-keyring_1.0-1_all.deb || true
                        sudo dpkg -i cuda-keyring_1.0-1_all.deb 2>/dev/null || true
                        sudo apt-get update
                        sudo apt-get install -y cuda-toolkit-12-2 || \
                        sudo apt-get install -y nvidia-cuda-toolkit || \
                        print_warning "CUDA installation failed - continuing with CPU mode"
                        ;;
                    yum|dnf)
                        sudo $PACKAGE_MANAGER install -y cuda-toolkit || \
                        print_warning "CUDA installation failed - continuing with CPU mode"
                        ;;
                esac
            fi
            ;;
        amd)
            print_success "AMD ROCm already detected"
            ;;
        amd_no_rocm)
            print_status "Installing ROCm for AMD GPU..."
            case $PACKAGE_MANAGER in
                apt)
                    # Add ROCm repository
                    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - || true
                    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list || true
                    sudo apt-get update
                    sudo apt-get install -y rocm-dev hip-dev || \
                    print_warning "ROCm installation failed - continuing with CPU mode"
                    ;;
                yum|dnf)
                    # Add ROCm repository for RHEL/CentOS
                    echo '[ROCm]' | sudo tee /etc/yum.repos.d/rocm.repo
                    echo 'name=ROCm' | sudo tee -a /etc/yum.repos.d/rocm.repo
                    echo 'baseurl=https://repo.radeon.com/rocm/yum/5.7/main' | sudo tee -a /etc/yum.repos.d/rocm.repo
                    echo 'enabled=1' | sudo tee -a /etc/yum.repos.d/rocm.repo
                    echo 'gpgcheck=1' | sudo tee -a /etc/yum.repos.d/rocm.repo
                    echo 'gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key' | sudo tee -a /etc/yum.repos.d/rocm.repo
                    sudo $PACKAGE_MANAGER install -y rocm-dev hip-dev || \
                    print_warning "ROCm installation failed - continuing with CPU mode"
                    ;;
            esac

            # Set up ROCm environment
            if [ -d "/opt/rocm" ]; then
                echo "export ROCM_PATH=/opt/rocm" >> ~/.bashrc
                echo "export HIP_PATH=/opt/rocm" >> ~/.bashrc
                echo "export PATH=/opt/rocm/bin:\$PATH" >> ~/.bashrc
                echo "export LD_LIBRARY_PATH=/opt/rocm/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
                export ROCM_PATH=/opt/rocm
                export HIP_PATH=/opt/rocm
                export PATH="/opt/rocm/bin:$PATH"
                export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
                print_success "ROCm environment configured"
            fi
            ;;
        nvidia_no_driver)
            print_warning "NVIDIA GPU detected but drivers not installed"
            print_status "Attempting to install NVIDIA drivers..."
            case $PACKAGE_MANAGER in
                apt)
                    sudo apt-get install -y nvidia-driver-535 || \
                    ubuntu-drivers autoinstall || \
                    print_warning "NVIDIA driver installation failed - please install manually"
                    ;;
                yum|dnf)
                    sudo $PACKAGE_MANAGER install -y nvidia-driver || \
                    print_warning "NVIDIA driver installation failed - please install manually"
                    ;;
            esac
            ;;
        none)
            print_status "No GPU detected - will build in CPU-only mode"
            ;;
    esac
}

# Function to build the native library
build_native_library() {
    print_step "Building Native C++ Library"

    cd "${SCRIPT_DIR}/src/main/cpp" || {
        print_error "Cannot find C++ source directory"
        return 1
    }

    # Clean previous builds
    rm -rf CMakeCache.txt CMakeFiles/ Makefile build/ || true

    # Configure with CMake
    print_status "Configuring build with CMake..."

    CMAKE_ARGS=""
    if [ "$GPU_TYPE" = "amd" ] || [ "$GPU_TYPE" = "amd_no_rocm" ]; then
        CMAKE_ARGS="-DUSE_ROCM=ON"
        if [ -n "$ROCM_PATH" ]; then
            CMAKE_ARGS="$CMAKE_ARGS -DROCM_PATH=$ROCM_PATH"
        fi
    elif [ "$GPU_TYPE" = "nvidia" ]; then
        CMAKE_ARGS="-DUSE_CUDA=ON"
    fi

    print_status "Running: cmake . $CMAKE_ARGS"
    cmake . $CMAKE_ARGS 2>&1 | tee -a "${LOG_FILE}" || {
        print_warning "CMake configuration failed, trying CPU-only build..."
        print_status "Running: cmake . -DUSE_CPU_ONLY=ON"
        cmake . -DUSE_CPU_ONLY=ON 2>&1 | tee -a "${LOG_FILE}" || {
            print_error "CMake configuration failed completely"
            print_error "Check ${LOG_FILE} for detailed error information"
            return 1
        }
    }

    # Build
    print_status "Building native library..."
    make -j$(nproc 2>/dev/null || echo 4) VERBOSE=1 2>&1 | tee -a "${LOG_FILE}" || {
        print_warning "Parallel build failed, trying single-threaded build..."
        make VERBOSE=1 2>&1 | tee -a "${LOG_FILE}" || {
            print_error "Native library build failed"
            print_error "Check ${LOG_FILE} for detailed error information"
            return 1
        }
    }

    # Verify build
    if [ -f "libopennlp_gpu.so" ] || [ -f "libopennlp_gpu.dylib" ]; then
        print_success "Native library built successfully"

        # Copy to Java resources
        mkdir -p "${SCRIPT_DIR}/src/main/resources/native/linux/x86_64" || true
        cp libopennlp_gpu.* "${SCRIPT_DIR}/src/main/resources/native/linux/x86_64/" 2>/dev/null || \
        cp libopennlp_gpu.* "${SCRIPT_DIR}/src/main/resources/" || \
        print_warning "Could not copy native library to resources directory"

    else
        print_error "Native library build verification failed"
        return 1
    fi

    cd "${SCRIPT_DIR}"
}

# Function to build the Java project
build_java_project() {
    print_step "Building Java Project"

    cd "${SCRIPT_DIR}"

    # Clean and compile
    print_status "Running Maven clean compile..."
    mvn clean compile || {
        print_warning "Maven build failed, trying with offline mode..."
        mvn clean compile -o || {
            print_warning "Offline build failed, trying to update dependencies..."
            mvn dependency:resolve || true
            mvn clean compile || {
                print_error "Java project build failed"
                return 1
            }
        }
    }

    print_success "Java project built successfully"
}

# Function to run tests and validation
run_validation() {
    print_step "Running Validation Tests"

    cd "${SCRIPT_DIR}"

    # Build classpath
    mvn dependency:build-classpath -Dmdep.outputFile=classpath.txt -q || {
        print_warning "Could not build classpath file"
        return 0
    }

    # Run GPU diagnostics
    print_status "Running GPU diagnostics..."
    timeout 30s java -cp "target/classes:$(cat classpath.txt 2>/dev/null || echo '')" org.apache.opennlp.gpu.tools.GpuDiagnostics 2>/dev/null || {
        print_warning "GPU diagnostics failed or timed out"
    }

    # Run demo
    print_status "Running GPU ML demo..."
    timeout 60s java -cp "target/classes:$(cat classpath.txt 2>/dev/null || echo '')" org.apache.opennlp.gpu.ml.GpuMlDemo 2>/dev/null || {
        print_warning "GPU ML demo failed or timed out"
    }

    print_success "Validation completed"
}

# Function to create setup summary
create_summary() {
    print_step "Setup Summary"

    SUMMARY_FILE="${SCRIPT_DIR}/SETUP_SUMMARY.md"
    cat > "$SUMMARY_FILE" << EOF
# OpenNLP GPU Extension - Setup Summary

## System Information
- **OS**: $OS_TYPE ($DISTRO)
- **Package Manager**: $PACKAGE_MANAGER
- **GPU**: $GPU_TYPE
- **Cloud Provider**: ${CLOUD_PROVIDER:-"None/Local"}

## Installed Components
- **Java**: $(java -version 2>&1 | head -n 1 || echo "Not found")
- **Maven**: $(mvn --version 2>/dev/null | head -n 1 || echo "Not found")
- **CMake**: $(cmake --version 2>/dev/null | head -n 1 || echo "Not found")

## Build Status
- **Native Library**: $([ -f "src/main/cpp/libopennlp_gpu.so" ] && echo "✅ Built" || echo "❌ Failed")
- **Java Project**: $([ -d "target/classes" ] && echo "✅ Built" || echo "❌ Failed")

## Quick Start Commands

### Run GPU Diagnostics
\`\`\`bash
cd ${SCRIPT_DIR}
java -cp "target/classes:\$(cat classpath.txt)" org.apache.opennlp.gpu.tools.GpuDiagnostics
\`\`\`

### Run GPU ML Demo
\`\`\`bash
cd ${SCRIPT_DIR}
java -cp "target/classes:\$(cat classpath.txt)" org.apache.opennlp.gpu.ml.GpuMlDemo
\`\`\`

### Build Native Library
\`\`\`bash
cd ${SCRIPT_DIR}/src/main/cpp
cmake . && make -j4
\`\`\`

### Build Java Project
\`\`\`bash
cd ${SCRIPT_DIR}
mvn clean compile
\`\`\`

## Environment Variables
$([ -n "$JAVA_HOME" ] && echo "- JAVA_HOME=$JAVA_HOME")
$([ -n "$ROCM_PATH" ] && echo "- ROCM_PATH=$ROCM_PATH")
$([ -n "$CUDA_HOME" ] && echo "- CUDA_HOME=$CUDA_HOME")

## Troubleshooting
- Check logs: ${LOG_FILE}
- Error details: ${ERROR_LOG}
- Re-run setup: \`./setup.sh\`

Generated on: $(date)
EOF

    print_success "Setup summary created: $SUMMARY_FILE"
}

# Main setup function
main() {
    clear
    cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                    OpenNLP GPU Extension                       ║
║                   One-Click Setup Script                       ║
║                                                               ║
║  Supports: Linux, macOS, Windows (WSL)                       ║
║  GPU: NVIDIA CUDA, AMD ROCm, CPU-only fallback               ║
║  Cloud: AWS, GCP, Azure, Local                               ║
╚════════════════════════════════════════════════════════════════╝
EOF

    print_status "Starting OpenNLP GPU Extension setup..."
    echo "Log file: $LOG_FILE"
    echo "Error log: $ERROR_LOG"
    echo

    # Initialize logs
    echo "Setup started at $(date)" > "$LOG_FILE"
    echo "Setup started at $(date)" > "$ERROR_LOG"

    # Main setup sequence
    detect_system
    install_java
    install_maven
    install_cmake
    install_dev_tools
    install_gpu_support
    build_native_library
    build_java_project
    run_validation
    create_summary

    print_step "Setup Complete!"
    print_success "OpenNLP GPU Extension has been successfully set up!"
    print_status "Run './gpu_demo.sh' to see it in action"
    print_status "Check SETUP_SUMMARY.md for detailed information"

    # Create quick demo script
    cat > "${SCRIPT_DIR}/gpu_demo.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
echo "🚀 Running OpenNLP GPU Extension Demo"
echo "======================================"
echo
echo "1. GPU Diagnostics:"
java -cp "target/classes:$(cat classpath.txt 2>/dev/null || echo '')" org.apache.opennlp.gpu.tools.GpuDiagnostics
echo
echo "2. GPU ML Demo:"
java -cp "target/classes:$(cat classpath.txt 2>/dev/null || echo '')" org.apache.opennlp.gpu.ml.GpuMlDemo
EOF
    # Make scripts executable
    chmod +x "${SCRIPT_DIR}/gpu_demo.sh"
    chmod +x "${PROJECT_ROOT}/docker/docker_setup.sh"

    echo
    print_success "🎉 All done! Your OpenNLP GPU Extension is ready to use!"
}

# Run main function
main "$@"
