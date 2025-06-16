#!/bin/bash
# Java Environment Setup for OpenNLP GPU

# Detect Java installation
if command -v java >/dev/null 2>&1; then
    JAVA_PATH=$(which java)
    JAVA_HOME_DETECTED=$(readlink -f "$JAVA_PATH" | sed 's/\/bin\/java$//')
    
    export JAVA_HOME="$JAVA_HOME_DETECTED"
    export PATH="$JAVA_HOME/bin:$PATH"
    
    # Get Java version
    JAVA_VERSION=$(java -version 2>&1 | head -n1 | cut -d'"' -f2)
    JAVA_MAJOR=$(echo "$JAVA_VERSION" | cut -d'.' -f1)
    if [[ "$JAVA_MAJOR" == "1" ]]; then
        JAVA_MAJOR=$(echo "$JAVA_VERSION" | cut -d'.' -f2)
    fi
    
    echo "Java Environment:"
    echo "  JAVA_HOME: $JAVA_HOME"
    echo "  Java Version: $JAVA_VERSION"
    echo "  Java Major: $JAVA_MAJOR"
    echo "  PATH: $PATH"
    
    # Warn if Java 17+ (Oracle extension conflict)
    if [[ "$JAVA_MAJOR" -ge 17 ]]; then
        echo "⚠️ Java $JAVA_MAJOR detected - Oracle Java extension may interfere"
        echo "💡 Use Red Hat Java extension for VS Code"
    fi
    
else
    echo "❌ Java not found. Please install Java 8 or higher."
    exit 1
fi

# Maven configuration
export MAVEN_OPTS="-Xmx1024m -XX:MaxPermSize=256m"
export M2_HOME="${M2_HOME:-/usr/share/maven}"

echo "Maven Environment:"
echo "  MAVEN_OPTS: $MAVEN_OPTS"
echo "  M2_HOME: $M2_HOME"

# VS Code Java configuration
export VSCODE_JAVA_HOME="$JAVA_HOME"
export VSCODE_MAVEN_HOME="$M2_HOME"

echo "VS Code Environment:"
echo "  VSCODE_JAVA_HOME: $VSCODE_JAVA_HOME"
echo "  VSCODE_MAVEN_HOME: $VSCODE_MAVEN_HOME"
