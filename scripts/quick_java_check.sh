#!/bin/bash

# Quick Java Environment Check
# Run this script to verify Java setup is working

echo "🔍 Quick Java Environment Check"
echo "==============================="

# Check Java version
echo "Java version:"
java -version

echo ""
echo "Java compiler version:"
javac -version

echo ""
echo "JAVA_HOME: ${JAVA_HOME:-'Not set'}"

echo ""
echo "Maven Java version:"
mvn -version | grep "Java version"

echo ""
echo "VSCode Java settings:"
if [ -f ".vscode/settings.json" ]; then
    grep "java.home" .vscode/settings.json || echo "java.home not found in VSCode settings"
else
    echo "No VSCode settings file found"
fi

echo ""
echo "Testing Maven compilation:"
if mvn compile -q; then
    echo "✅ Maven compilation successful"
else
    echo "❌ Maven compilation failed"
fi

echo ""
echo "✅ Java environment check complete!"
