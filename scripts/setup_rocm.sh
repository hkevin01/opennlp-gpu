#!/bin/bash

# ROCm Detection and Setup Script for OpenNLP GPU
# This script detects ROCm installation and sets up the environment

echo "🔍 ROCm Detection and Setup"
echo "=========================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Check for ROCm installation
echo "🔎 Checking for ROCm installation..."

ROCM_PATHS=(
    "/opt/rocm"
    "/usr/local/rocm"
    "$HOME/rocm"
)

ROCM_FOUND=false
ROCM_PATH=""

for path in "${ROCM_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/bin/hipcc" ]; then
        echo "✅ ROCm found at: $path"
        ROCM_PATH="$path"
        ROCM_FOUND=true
        break
    fi
done

if [ "$ROCM_FOUND" = false ]; then
    echo "❌ ROCm not found in standard locations"
    echo "   Checking system packages..."
    
    # Check for system package installation
    if command -v hipcc >/dev/null 2>&1; then
        ROCM_PATH=$(dirname $(dirname $(which hipcc)))
        echo "✅ ROCm found via system packages at: $ROCM_PATH"
        ROCM_FOUND=true
    elif dpkg -l | grep -q rocm; then
        echo "✅ ROCm packages detected via dpkg"
        ROCM_PATH="/usr"
        ROCM_FOUND=true
    elif rpm -qa | grep -q rocm 2>/dev/null; then
        echo "✅ ROCm packages detected via rpm"
        ROCM_PATH="/usr"
        ROCM_FOUND=true
    fi
fi

if [ "$ROCM_FOUND" = false ]; then
    echo ""
    echo "❌ ROCm installation not detected"
    echo ""
    echo "📥 To install ROCm, please visit:"
    echo "   https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
    echo ""
    echo "📦 Quick install commands for Ubuntu/Debian:"
    echo "   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.4.50403-1_all.deb"
    echo "   sudo dpkg -i amdgpu-install_5.4.50403-1_all.deb"
    echo "   sudo amdgpu-install --usecase=rocm"
    exit 1
fi

# Display ROCm information
echo ""
echo "📋 ROCm Installation Details:"
echo "   Path: $ROCM_PATH"

if [ -f "$ROCM_PATH/bin/rocm-smi" ]; then
    echo "   Version: $($ROCM_PATH/bin/rocm-smi --showproductname 2>/dev/null | head -1 || echo 'Unknown')"
fi

# Check for GPU devices
echo ""
echo "🎮 Checking for AMD GPU devices..."
if [ -f "$ROCM_PATH/bin/rocm-smi" ]; then
    GPU_COUNT=$($ROCM_PATH/bin/rocm-smi -i | grep -c "GPU\[" 2>/dev/null || echo "0")
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo "✅ Found $GPU_COUNT AMD GPU(s):"
        $ROCM_PATH/bin/rocm-smi --showproductname 2>/dev/null || echo "   (Unable to get GPU details)"
    else
        echo "❌ No AMD GPUs detected"
        echo "   Note: You may still build ROCm support for future use"
    fi
else
    echo "⚠️  rocm-smi not found, cannot detect GPUs"
fi

# Check compiler
echo ""
echo "🔧 Checking ROCm compiler..."
if [ -f "$ROCM_PATH/bin/hipcc" ]; then
    echo "✅ HIP compiler found: $ROCM_PATH/bin/hipcc"
    HIP_VERSION=$($ROCM_PATH/bin/hipcc --version 2>/dev/null | grep -o "HIP version: [0-9.]*" || echo "Unknown")
    echo "   Version: $HIP_VERSION"
else
    echo "❌ HIP compiler not found"
    exit 1
fi

# Set up environment
echo ""
echo "🌍 Setting up ROCm environment..."
export ROCM_PATH="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

# Update VS Code settings for ROCm
echo ""
echo "⚙️  Updating VS Code settings for ROCm..."
VSCODE_SETTINGS=".vscode/settings.json"

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Add ROCm environment to VS Code settings
if [ -f "$VSCODE_SETTINGS" ]; then
    # Check if ROCm settings already exist
    if ! grep -q "ROCM_PATH" "$VSCODE_SETTINGS"; then
        # Add ROCm environment variables using jq
        if command -v jq >/dev/null 2>&1; then
            # Read existing settings or create empty object
            if [ -s "$VSCODE_SETTINGS" ]; then
                CURRENT_SETTINGS=$(cat "$VSCODE_SETTINGS")
            else
                CURRENT_SETTINGS="{}"
            fi
            
            # Merge new settings
            echo "$CURRENT_SETTINGS" | jq ". += {
                \"terminal.integrated.env.linux\": (.\"terminal.integrated.env.linux\" // {} | . += {
                    \"ROCM_PATH\": \"$ROCM_PATH\",
                    \"HIP_PATH\": \"$ROCM_PATH\",
                    \"PATH\": \"$ROCM_PATH/bin:\${env:PATH}\",
                    \"LD_LIBRARY_PATH\": \"$ROCM_PATH/lib:\${env:LD_LIBRARY_PATH}\"
                }),
                \"cmake.configureEnvironment\": (.\"cmake.configureEnvironment\" // {} | . += {
                    \"ROCM_PATH\": \"$ROCM_PATH\",
                    \"HIP_PATH\": \"$ROCM_PATH\"
                }),
                \"cmake.configureArgs\": ((.\"cmake.configureArgs\" // []) + [\"-DROCM_PATH=$ROCM_PATH\", \"-DUSE_ROCM=ON\"] | unique)
            }" > "$VSCODE_SETTINGS.tmp" && mv "$VSCODE_SETTINGS.tmp" "$VSCODE_SETTINGS"
            
            echo "✅ VS Code settings updated with ROCm configuration"
        else
            # Fallback if jq is not available
            echo "⚠️  jq not available, skipping VS Code ROCm configuration"
        fi
    else
        echo "✅ ROCm settings already present in VS Code configuration"
    fi
else
    echo "⚠️  VS Code settings file not found, creating basic configuration..."
    cat > "$VSCODE_SETTINGS" << EOF
{
  "terminal.integrated.env.linux": {
    "ROCM_PATH": "$ROCM_PATH",
    "HIP_PATH": "$ROCM_PATH",
    "PATH": "$ROCM_PATH/bin:\${env:PATH}",
    "LD_LIBRARY_PATH": "$ROCM_PATH/lib:\${env:LD_LIBRARY_PATH}"
  },
  "cmake.configureEnvironment": {
    "ROCM_PATH": "$ROCM_PATH",
    "HIP_PATH": "$ROCM_PATH"
  },
  "cmake.configureArgs": [
    "-DROCM_PATH=$ROCM_PATH",
    "-DUSE_ROCM=ON"
  ]
}
EOF
fi

# Test CMake configuration
echo ""
echo "🧪 Testing CMake configuration..."
cd "$PROJECT_DIR"

# Clean previous build
if [ -d "build" ]; then
    echo "   Cleaning previous build..."
    rm -rf build
fi

mkdir -p build
cd build

echo "   Running CMake configure..."
cmake -DROCM_PATH="$ROCM_PATH" -DUSE_ROCM=ON ../src/main/cpp

if [ $? -eq 0 ]; then
    echo "✅ CMake configuration successful!"
    echo ""
    echo "🚀 To build the project:"
    echo "   cd build && make -j\$(nproc)"
else
    echo "❌ CMake configuration failed"
    echo "   Check the error messages above"
    exit 1
fi

echo ""
echo "✨ ROCm setup complete!"
echo ""
echo "📝 Environment variables set:"
echo "   ROCM_PATH=$ROCM_PATH"
echo "   HIP_PATH=$ROCM_PATH"
echo "   PATH includes: $ROCM_PATH/bin"
echo "   LD_LIBRARY_PATH includes: $ROCM_PATH/lib"
echo ""
echo "🔄 To reload VS Code with new settings:"
echo "   Press Ctrl+Shift+P and run 'Developer: Reload Window'"
