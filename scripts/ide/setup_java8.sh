#!/bin/bash
# filepath: scripts/ide/setup_java8.sh

set -e

JDK_VERSION="8u392-b08"
JDK_BUILD="OpenJDK8U-jdk_x64_linux_hotspot_8u392b08.tar.gz"
JDK_URL="https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u392-b08/$JDK_BUILD"
JDK_DIR=".jdk"
JDK_HOME="$(pwd)/$JDK_DIR/jdk8u392-b08"

echo "🔍 Checking for Java 8..."

if [ -x "$JDK_HOME/bin/java" ]; then
    echo "✅ Java 8 already installed at $JDK_HOME"
else
    echo "⬇️  Downloading Java 8 from Adoptium..."
    mkdir -p "$JDK_DIR"
    cd "$JDK_DIR"
    curl -L -o "$JDK_BUILD" "$JDK_URL"
    tar -xzf "$JDK_BUILD"
    rm "$JDK_BUILD"
    cd ..
    echo "✅ Java 8 installed at $JDK_HOME"
fi

echo "🔧 Setting JAVA_HOME and updating VS Code settings..."

export JAVA_HOME="$JDK_HOME"
export PATH="$JAVA_HOME/bin:$PATH"

VSCODE_SETTINGS=".vscode/settings.json"
mkdir -p .vscode

# Update VS Code settings.json
cat > "$VSCODE_SETTINGS" <<EOF
{
    "java.home": "$JAVA_HOME",
    "java.configuration.runtimes": [
        {
            "name": "JavaSE-1.8",
            "path": "$JAVA_HOME",
            "default": true
        }
    ],
    "github.copilot.enable": true,
    "github.copilot.inlineSuggest.enable": true,
    "github.copilot.editor.enable": true,
    "github.copilot.editor.showEditorCompletions": true,
    "github.copilot.advanced": {
        "inlineSuggestEnable": true,
        "inlineSuggestCount": 3
    },
    "github.copilot.experimental.panel.enabled": false,
    "github.copilot.experimental.ask": false
}
EOF

echo "✅ VS Code settings updated to use Java 8 at $JAVA_HOME"

# Make Java environment persistent in shell profiles
PROFILE_LINES="
# >>> Java 8 (OpenJDK) setup by setup_java8.sh >>>
export JAVA_HOME=\"$JAVA_HOME\"
export PATH=\"\$JAVA_HOME/bin:\$PATH\"
# <<< Java 8 (OpenJDK) setup by setup_java8.sh <<<
"

append_if_not_exists() {
    local file="$1"
    if [ -f "$file" ]; then
        if ! grep -q 'setup_java8.sh' "$file"; then
            echo "🔧 Adding Java 8 environment to $file"
            printf "%s\n" "$PROFILE_LINES" >> "$file"
        else
            echo "ℹ️  Java 8 environment already present in $file"
        fi
    fi
}

append_if_not_exists "$HOME/.bashrc"
append_if_not_exists "$HOME/.zshrc"

echo ""
echo "🎉 Java 8 is ready for VS Code and your environment. No more runtime warnings!"
echo "✅ Java 8 environment variables have been added to your ~/.bashrc and ~/.zshrc (if present)."
echo "   Restart your terminal or run 'source ~/.bashrc' or 'source ~/.zshrc' to activate."