#!/bin/bash
# filepath: scripts/ide/fix_cmake_configure.sh

set -e

echo "🔧 Fixing CMake configuration issues in VS Code..."

# Check if cmake is installed
if ! command -v cmake >/dev/null 2>&1; then
    echo "⬇️ Installing CMake..."
    if [ "$(uname)" = "Darwin" ]; then
        if command -v brew >/dev/null 2>&1; then
            brew install cmake
        else
            echo "❌ Homebrew not found. Please install CMake manually."
            exit 1
        fi
    elif [ -f /etc/debian_version ]; then
        sudo apt-get update
        sudo apt-get install -y cmake
    elif [ -f /etc/redhat-release ]; then
        sudo yum install -y cmake
    else
        echo "❌ Unsupported OS. Please install CMake manually."
        exit 1
    fi
fi

CMAKE_PATH="$(command -v cmake)"
echo "✅ CMake found at: $CMAKE_PATH"

VSCODE_SETTINGS=".vscode/settings.json"
mkdir -p .vscode

# Update VS Code settings to configure CMake and disable auto-configure for Java projects
if [ -f "$VSCODE_SETTINGS" ]; then
    # Backup existing settings
    cp "$VSCODE_SETTINGS" "$VSCODE_SETTINGS.backup"
    
    # Use Python to merge settings (more reliable than sed for JSON)
    python3 -c "
import json
import sys

try:
    with open('$VSCODE_SETTINGS', 'r') as f:
        settings = json.load(f)
except:
    settings = {}

# CMake configuration
settings['cmake.cmakePath'] = '$CMAKE_PATH'
settings['cmake.configureOnOpen'] = False
settings['cmake.configureOnEdit'] = False
settings['cmake.autoSelectActiveFolder'] = False
settings['cmake.showConfigureWithDebuggerNotification'] = False

# Disable CMake for Java projects
settings['cmake.sourceDirectory'] = None
settings['files.associations'] = settings.get('files.associations', {})
settings['files.associations']['CMakeLists.txt'] = 'cmake'

# Java project settings (ensure CMake doesn't interfere)
settings['java.configuration.detectJdksAtStart'] = True
settings['java.import.gradle.enabled'] = False
settings['java.import.maven.enabled'] = True

with open('$VSCODE_SETTINGS', 'w') as f:
    json.dump(settings, f, indent=4)

print('Settings updated successfully')
" 2>/dev/null || {
    # Fallback if Python is not available - create minimal settings
    cat > "$VSCODE_SETTINGS" <<EOF
{
    "cmake.cmakePath": "$CMAKE_PATH",
    "cmake.configureOnOpen": false,
    "cmake.configureOnEdit": false,
    "cmake.autoSelectActiveFolder": false,
    "cmake.showConfigureWithDebuggerNotification": false,
    "cmake.sourceDirectory": null,
    "java.configuration.detectJdksAtStart": true,
    "java.import.gradle.enabled": false,
    "java.import.maven.enabled": true,
    "files.associations": {
        "CMakeLists.txt": "cmake"
    }
}
EOF
}
else
    # Create new settings file
    cat > "$VSCODE_SETTINGS" <<EOF
{
    "cmake.cmakePath": "$CMAKE_PATH",
    "cmake.configureOnOpen": false,
    "cmake.configureOnEdit": false,
    "cmake.autoSelectActiveFolder": false,
    "cmake.showConfigureWithDebuggerNotification": false,
    "cmake.sourceDirectory": null,
    "java.configuration.detectJdksAtStart": true,
    "java.import.gradle.enabled": false,
    "java.import.maven.enabled": true,
    "files.associations": {
        "CMakeLists.txt": "cmake"
    }
}
EOF
fi

# Remove any CMakeCache.txt or CMakeFiles that might be causing issues
if [ -f "CMakeCache.txt" ]; then
    echo "🗑️ Removing problematic CMakeCache.txt"
    rm -f CMakeCache.txt
fi

if [ -d "CMakeFiles" ]; then
    echo "🗑️ Removing problematic CMakeFiles directory"
    rm -rf CMakeFiles
fi

# Create a .cmake-disable file to prevent auto-detection
touch .cmake-disable

echo "✅ CMake configuration fixed:"
echo "   - CMake path set to: $CMAKE_PATH"
echo "   - Auto-configuration disabled"
echo "   - CMake debugger notifications disabled"
echo "   - Java project settings prioritized"
echo "   - Problematic cache files removed"
echo ""
echo "🔄 Please reload VS Code for changes to take effect:"
echo "   Ctrl+Shift+P -> 'Developer: Reload Window'"
