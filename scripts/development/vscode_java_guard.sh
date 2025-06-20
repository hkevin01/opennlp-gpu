#!/bin/bash
# VSCode Java Environment Guard - runs on VSCode startup

JAVA_21_PATH="/usr/lib/jvm/java-21-openjdk-amd64"

# Check if Java environment is correct
if [ "$JAVA_HOME" != "$JAVA_21_PATH" ] || ! command -v java &> /dev/null; then
    echo "🔧 Fixing Java environment..."
    export JAVA_HOME="$JAVA_21_PATH"
    export PATH="$JAVA_HOME/bin:$PATH"
    
    # Update shell environment
    if ! grep -q "JAVA_HOME.*java-21-openjdk-amd64" ~/.bashrc; then
        echo "export JAVA_HOME=$JAVA_21_PATH" >> ~/.bashrc
        echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc
    fi
    
    echo "✅ Java environment fixed"
fi

# Verify VSCode settings
if [ ! -f ".vscode/settings.json" ] || ! grep -q "java-21-openjdk-amd64" .vscode/settings.json; then
    echo "🔧 Updating VSCode settings..."
    ./scripts/setup_vscode_java.sh
fi

echo "🛡️ Java environment guard complete"
