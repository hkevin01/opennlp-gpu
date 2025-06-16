#!/bin/bash
set -e

echo "🔍 Checking for cmake..."

if command -v cmake >/dev/null 2>&1; then
    CMAKE_PATH="$(command -v cmake)"
    echo "✅ cmake found at $CMAKE_PATH"
else
    echo "❌ cmake not found. Installing..."
    if [ "$(uname)" = "Darwin" ]; then
        if command -v brew >/dev/null 2>&1; then
            brew install cmake
        else
            echo "❌ Homebrew not found. Please install Homebrew or cmake manually."
            exit 1
        fi
    elif [ -f /etc/debian_version ]; then
        sudo apt-get update
        sudo apt-get install -y cmake
    elif [ -f /etc/redhat-release ]; then
        sudo yum install -y cmake
    else
        echo "❌ Unsupported OS. Please install cmake manually."
        exit 1
    fi
    CMAKE_PATH="$(command -v cmake)"
    echo "✅ cmake installed at $CMAKE_PATH"
fi

VSCODE_SETTINGS=".vscode/settings.json"
mkdir -p .vscode

# Update or add cmake.cmakePath in settings.json
if [ -f "$VSCODE_SETTINGS" ]; then
    # Insert or update the cmake.cmakePath setting
    if grep -q '"cmake.cmakePath"' "$VSCODE_SETTINGS"; then
        sed -i "s#\"cmake.cmakePath\": \".*\"#\"cmake.cmakePath\": \"$CMAKE_PATH\"#g" "$VSCODE_SETTINGS"
    else
        # Insert before last }
        sed -i '$s/}/,\n    "cmake.cmakePath": "'"$CMAKE_PATH"'"\n}/' "$VSCODE_SETTINGS"
    fi
else
    cat > "$VSCODE_SETTINGS" <<EOF
{
    "cmake.cmakePath": "$CMAKE_PATH"
}
EOF
fi

echo "✅ VS Code settings updated: cmake.cmakePath = $CMAKE_PATH"
