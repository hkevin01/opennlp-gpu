#!/bin/bash
# VS Code reload script for OpenNLP GPU project

echo "🔄 Reloading VS Code Java configuration..."

# Set Java environment
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"

# Kill VS Code Java processes
echo "Stopping VS Code Java processes..."
pkill -f "java.*jdt.ls" 2>/dev/null || true
sleep 2

# Clean project
echo "Cleaning Maven project..."
mvn clean -q

# Restart with proper Java
echo "VS Code will now use Java "
echo "1. Restart VS Code"
echo "2. Press Ctrl+Shift+P"
echo "3. Run: 'Java: Reload Projects'"
echo "4. Run: 'Java: Rebuild Workspace'"
