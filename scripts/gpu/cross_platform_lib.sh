#!/bin/bash

# Cross-Platform Compatibility Library for OpenNLP GPU Scripts
# Source this file in other scripts for consistent cross-platform behavior

# Platform detection functions
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

detect_arch() {
    local arch=$(uname -m 2>/dev/null || echo "unknown")
    case $arch in
        x86_64|amd64)
            echo "x86_64"
            ;;
        arm64|aarch64)
            echo "arm64"
            ;;
        i386|i686)
            echo "i386"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

detect_distro() {
    local os=$(detect_os)
    
    if [[ "$os" == "linux" ]]; then
        if command -v lsb_release &> /dev/null; then
            lsb_release -si 2>/dev/null | tr '[:upper:]' '[:lower:]'
        elif [[ -f /etc/os-release ]]; then
            . /etc/os-release
            echo "$ID"
        elif [[ -f /etc/redhat-release ]]; then
            echo "redhat"
        elif [[ -f /etc/debian_version ]]; then
            echo "debian"
        else
            echo "unknown"
        fi
    elif [[ "$os" == "macos" ]]; then
        echo "macos"
    elif [[ "$os" == "windows" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Package manager detection
detect_package_manager() {
    local distro=$(detect_distro)
    
    case $distro in
        ubuntu|debian)
            echo "apt"
            ;;
        fedora|centos|rhel|redhat)
            if command -v dnf &> /dev/null; then
                echo "dnf"
            elif command -v yum &> /dev/null; then
                echo "yum"
            else
                echo "unknown"
            fi
            ;;
        opensuse*)
            echo "zypper"
            ;;
        arch)
            echo "pacman"
            ;;
        macos)
            if command -v brew &> /dev/null; then
                echo "brew"
            else
                echo "none"
            fi
            ;;
        windows)
            if command -v choco &> /dev/null; then
                echo "choco"
            elif command -v winget &> /dev/null; then
                echo "winget"
            else
                echo "none"
            fi
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Cross-platform command wrappers
xp_which() {
    local cmd="$1"
    command -v "$cmd" 2>/dev/null
}

xp_is_command_available() {
    local cmd="$1"
    command -v "$cmd" &> /dev/null
}

# Cross-platform file operations
xp_realpath() {
    local path="$1"
    
    if command -v realpath &> /dev/null; then
        realpath "$path"
    elif command -v readlink &> /dev/null; then
        readlink -f "$path" 2>/dev/null || echo "$path"
    else
        # Fallback for systems without realpath or readlink
        echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
    fi
}

xp_get_cpu_count() {
    local os=$(detect_os)
    
    case $os in
        linux|macos)
            nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1"
            ;;
        windows)
            echo "${NUMBER_OF_PROCESSORS:-1}"
            ;;
        *)
            echo "1"
            ;;
    esac
}

xp_get_memory_gb() {
    local os=$(detect_os)
    
    case $os in
        linux)
            if [[ -f /proc/meminfo ]]; then
                local mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
                echo $((mem_kb / 1024 / 1024))
            else
                echo "unknown"
            fi
            ;;
        macos)
            local mem_bytes=$(sysctl -n hw.memsize 2>/dev/null)
            if [[ -n "$mem_bytes" ]]; then
                echo $((mem_bytes / 1024 / 1024 / 1024))
            else
                echo "unknown"
            fi
            ;;
        windows)
            # This works in Git Bash/WSL
            if command -v wmic &> /dev/null; then
                local mem_mb=$(wmic computersystem get TotalPhysicalMemory /value | grep = | cut -d= -f2 | tr -d '\r')
                if [[ -n "$mem_mb" ]]; then
                    echo $((mem_mb / 1024 / 1024 / 1024))
                else
                    echo "unknown"
                fi
            else
                echo "unknown"
            fi
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Cross-platform package installation
xp_install_package() {
    local package="$1"
    local pm=$(detect_package_manager)
    
    case $pm in
        apt)
            sudo apt-get update && sudo apt-get install -y "$package"
            ;;
        dnf)
            sudo dnf install -y "$package"
            ;;
        yum)
            sudo yum install -y "$package"
            ;;
        zypper)
            sudo zypper install -y "$package"
            ;;
        pacman)
            sudo pacman -S --noconfirm "$package"
            ;;
        brew)
            brew install "$package"
            ;;
        choco)
            choco install -y "$package"
            ;;
        winget)
            winget install "$package"
            ;;
        *)
            echo "❌ Unknown package manager. Cannot install $package"
            return 1
            ;;
    esac
}

# Cross-platform service management
xp_is_service_running() {
    local service="$1"
    local os=$(detect_os)
    
    case $os in
        linux)
            if command -v systemctl &> /dev/null; then
                systemctl is-active "$service" &> /dev/null
            elif command -v service &> /dev/null; then
                service "$service" status &> /dev/null
            else
                return 1
            fi
            ;;
        macos)
            if command -v launchctl &> /dev/null; then
                launchctl list | grep -q "$service"
            else
                return 1
            fi
            ;;
        windows)
            if command -v sc &> /dev/null; then
                sc query "$service" | grep -q "RUNNING"
            else
                return 1
            fi
            ;;
        *)
            return 1
            ;;
    esac
}

# Cross-platform path handling
xp_path_separator() {
    local os=$(detect_os)
    
    case $os in
        windows)
            echo ";"
            ;;
        *)
            echo ":"
            ;;
    esac
}

xp_add_to_path() {
    local new_path="$1"
    local separator=$(xp_path_separator)
    
    if [[ ":$PATH:" != *":$new_path:"* ]]; then
        export PATH="$new_path$separator$PATH"
    fi
}

# Cross-platform temporary directory
xp_get_temp_dir() {
    local os=$(detect_os)
    
    case $os in
        windows)
            echo "${TEMP:-/tmp}"
            ;;
        *)
            echo "${TMPDIR:-/tmp}"
            ;;
    esac
}

# Cross-platform user home directory
xp_get_home_dir() {
    echo "${HOME:-$USERPROFILE}"
}

# Cross-platform clipboard operations
xp_copy_to_clipboard() {
    local text="$1"
    local os=$(detect_os)
    
    case $os in
        linux)
            if xp_is_command_available xclip; then
                echo "$text" | xclip -selection clipboard
            elif xp_is_command_available xsel; then
                echo "$text" | xsel --clipboard --input
            else
                echo "❌ No clipboard utility found (install xclip or xsel)"
                return 1
            fi
            ;;
        macos)
            echo "$text" | pbcopy
            ;;
        windows)
            echo "$text" | clip
            ;;
        *)
            echo "❌ Clipboard not supported on this platform"
            return 1
            ;;
    esac
}

# Cross-platform network connectivity check
xp_check_internet() {
    local test_hosts=("8.8.8.8" "1.1.1.1" "google.com")
    
    for host in "${test_hosts[@]}"; do
        if ping -c 1 -W 3 "$host" &> /dev/null; then
            return 0
        fi
    done
    
    return 1
}

# Cross-platform Java detection
xp_find_java() {
    local java_cmd=""
    
    # Check JAVA_HOME first
    if [[ -n "$JAVA_HOME" ]]; then
        if [[ -x "$JAVA_HOME/bin/java" ]]; then
            java_cmd="$JAVA_HOME/bin/java"
        fi
    fi
    
    # Check PATH
    if [[ -z "$java_cmd" ]] && xp_is_command_available java; then
        java_cmd="java"
    fi
    
    # Platform-specific locations
    if [[ -z "$java_cmd" ]]; then
        local os=$(detect_os)
        local common_locations=()
        
        case $os in
            linux)
                common_locations=(
                    "/usr/lib/jvm/default-java/bin/java"
                    "/usr/lib/jvm/java-17-openjdk*/bin/java"
                    "/usr/lib/jvm/java-11-openjdk*/bin/java"
                    "/opt/java/*/bin/java"
                )
                ;;
            macos)
                common_locations=(
                    "/Library/Java/JavaVirtualMachines/*/Contents/Home/bin/java"
                    "/System/Library/Java/JavaVirtualMachines/*/Contents/Home/bin/java"
                    "/usr/libexec/java_home -v 17"
                    "/usr/libexec/java_home -v 11"
                )
                ;;
            windows)
                common_locations=(
                    "/c/Program Files/Java/*/bin/java.exe"
                    "/c/Program Files (x86)/Java/*/bin/java.exe"
                )
                ;;
        esac
        
        for location in "${common_locations[@]}"; do
            if [[ -x "$location" ]]; then
                java_cmd="$location"
                break
            fi
        done
    fi
    
    echo "$java_cmd"
}

xp_get_java_version() {
    local java_cmd=$(xp_find_java)
    
    if [[ -n "$java_cmd" ]]; then
        local version_output=$("$java_cmd" -version 2>&1 | head -n 1)
        local version=$(echo "$version_output" | sed -n 's/.*version "\([0-9][0-9]*\).*/\1/p')
        
        if [[ -z "$version" ]]; then
            # Try alternative format for older Java versions
            version=$(echo "$version_output" | sed -n 's/.*version "1\.\([0-9][0-9]*\).*/\1/p')
        fi
        
        echo "$version"
    else
        echo ""
    fi
}

# Cross-platform Maven detection
xp_find_maven() {
    local maven_cmd=""
    
    # Check M2_HOME first
    if [[ -n "$M2_HOME" ]]; then
        if [[ -x "$M2_HOME/bin/mvn" ]]; then
            maven_cmd="$M2_HOME/bin/mvn"
        fi
    fi
    
    # Check PATH
    if [[ -z "$maven_cmd" ]] && xp_is_command_available mvn; then
        maven_cmd="mvn"
    fi
    
    echo "$maven_cmd"
}

# Cross-platform GPU detection
xp_detect_gpu() {
    local gpus_found=()
    
    # NVIDIA
    if xp_is_command_available nvidia-smi; then
        local nvidia_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null)
        if [[ -n "$nvidia_gpus" ]]; then
            while IFS= read -r gpu; do
                gpus_found+=("NVIDIA:$gpu")
            done <<< "$nvidia_gpus"
        fi
    fi
    
    # AMD
    if xp_is_command_available rocm-smi; then
        local amd_gpus=$(rocm-smi --showproductname 2>/dev/null | grep -v "^$")
        if [[ -n "$amd_gpus" ]]; then
            while IFS= read -r gpu; do
                gpus_found+=("AMD:$gpu")
            done <<< "$amd_gpus"
        fi
    fi
    
    # Intel (Linux)
    if [[ $(detect_os) == "linux" ]] && xp_is_command_available intel_gpu_top; then
        gpus_found+=("Intel:Intel GPU")
    fi
    
    # Apple Silicon (macOS)
    if [[ $(detect_os) == "macos" ]] && system_profiler SPHardwareDataType 2>/dev/null | grep -q "Apple"; then
        gpus_found+=("Apple:Apple Silicon GPU")
    fi
    
    # Generic detection via lspci (Linux)
    if [[ $(detect_os) == "linux" ]] && xp_is_command_available lspci; then
        local pci_gpus=$(lspci 2>/dev/null | grep -i vga)
        if [[ -n "$pci_gpus" ]]; then
            while IFS= read -r gpu; do
                if [[ "$gpu" == *"NVIDIA"* ]] && [[ "${gpus_found[*]}" != *"NVIDIA"* ]]; then
                    gpus_found+=("NVIDIA:$(echo "$gpu" | sed 's/.*NVIDIA/NVIDIA/')")
                elif [[ "$gpu" == *"AMD"* ]] && [[ "${gpus_found[*]}" != *"AMD"* ]]; then
                    gpus_found+=("AMD:$(echo "$gpu" | sed 's/.*AMD/AMD/')")
                elif [[ "$gpu" == *"Intel"* ]] && [[ "${gpus_found[*]}" != *"Intel"* ]]; then
                    gpus_found+=("Intel:$(echo "$gpu" | sed 's/.*Intel/Intel/')")
                fi
            done <<< "$pci_gpus"
        fi
    fi
    
    printf '%s\n' "${gpus_found[@]}"
}

# Cross-platform error handling
xp_error() {
    local message="$1"
    local exit_code="${2:-1}"
    
    echo "❌ Error: $message" >&2
    exit "$exit_code"
}

xp_warning() {
    local message="$1"
    echo "⚠️ Warning: $message" >&2
}

xp_info() {
    local message="$1"
    echo "ℹ️ Info: $message"
}

xp_success() {
    local message="$1"
    echo "✅ Success: $message"
}

# Cross-platform progress indicators
xp_spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    
    while ps -p "$pid" > /dev/null 2>&1; do
        local temp=${spinstr#?}
        printf ' [%c]  ' "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf '\b\b\b\b\b\b'
    done
    
    printf '    \b\b\b\b'
}

# Platform information summary
xp_system_info() {
    echo "System Information:"
    echo "=================="
    echo "OS: $(detect_os)"
    echo "Architecture: $(detect_arch)"
    echo "Distribution: $(detect_distro)"
    echo "Package Manager: $(detect_package_manager)"
    echo "CPU Cores: $(xp_get_cpu_count)"
    echo "Memory: $(xp_get_memory_gb)GB"
    echo "Java: $(xp_get_java_version)"
    echo "Maven: $(xp_find_maven)"
    echo "Home Directory: $(xp_get_home_dir)"
    echo "Temp Directory: $(xp_get_temp_dir)"
    
    local gpus=($(xp_detect_gpu))
    if [[ ${#gpus[@]} -gt 0 ]]; then
        echo "GPUs:"
        for gpu in "${gpus[@]}"; do
            echo "  - $gpu"
        done
    else
        echo "GPUs: None detected"
    fi
}

# Initialization message
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being run directly, not sourced
    echo "🌐 OpenNLP GPU Cross-Platform Compatibility Library"
    echo "This library provides cross-platform functions for OpenNLP GPU scripts."
    echo
    echo "Usage: source this file in your scripts"
    echo "Example: source \"\$(dirname \"\$0\")/cross_platform_lib.sh\""
    echo
    
    if [[ "$1" == "--test" ]]; then
        echo "Running system information test..."
        echo
        xp_system_info
    elif [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        echo "Available functions:"
        echo "==================="
        declare -F | grep "xp_" | sed 's/declare -f xp_/  xp_/' | sort
    fi
fi
