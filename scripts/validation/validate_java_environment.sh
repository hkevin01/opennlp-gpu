#!/bin/bash

# Java Environment Validation Script
# Ensures proper Java configuration for OpenNLP GPU project

echo "🔍 Validating Java Environment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    elif [ "$status" = "ERROR" ]; then
        echo -e "${RED}❌ $message${NC}"
    else
        echo -e "${BLUE}ℹ️  $message${NC}"
    fi
}

# Check available Java versions
print_status "INFO" "Checking available Java versions..."
echo "Available Java installations:"
if command -v update-java-alternatives &> /dev/null; then
    update-java-alternatives --list
else
    ls /usr/lib/jvm/ 2>/dev/null || echo "No JVM directory found"
fi

# Check current JAVA_HOME
print_status "INFO" "Current JAVA_HOME: ${JAVA_HOME:-'Not set'}"

# Check current java command
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n1)
    print_status "INFO" "Current java command: $JAVA_VERSION"
    
    # Extract version number
    if [[ $JAVA_VERSION =~ \"([0-9]+)\.([0-9]+)\.([0-9]+) ]] || [[ $JAVA_VERSION =~ \"([0-9]+)\.([0-9]+) ]] || [[ $JAVA_VERSION =~ \"([0-9]+) ]]; then
        MAJOR_VERSION=${BASH_REMATCH[1]}
        if [ "$MAJOR_VERSION" -ge 17 ]; then
            print_status "OK" "Java version $MAJOR_VERSION is compatible (>=17)"
        elif [ "$MAJOR_VERSION" -ge 11 ]; then
            print_status "WARN" "Java version $MAJOR_VERSION works but Java 17+ recommended"
        else
            print_status "ERROR" "Java version $MAJOR_VERSION is too old (need 11+)"
        fi
    else
        print_status "ERROR" "Could not parse Java version"
    fi
else
    print_status "ERROR" "Java command not found"
fi

# Check javac
if command -v javac &> /dev/null; then
    JAVAC_VERSION=$(javac -version 2>&1)
    print_status "OK" "Java compiler available: $JAVAC_VERSION"
else
    print_status "ERROR" "Java compiler (javac) not found"
fi

# Check Maven Java version
if command -v mvn &> /dev/null; then
    MVN_JAVA=$(mvn -version 2>&1 | grep "Java version")
    print_status "INFO" "Maven using: $MVN_JAVA"
else
    print_status "WARN" "Maven not found"
fi

# Check VSCode Java settings
VSCODE_SETTINGS=".vscode/settings.json"
if [ -f "$VSCODE_SETTINGS" ]; then
    print_status "OK" "VSCode settings file exists"
    
    # Check Java home setting
    if grep -q "java.home" "$VSCODE_SETTINGS"; then
        VSCODE_JAVA_HOME=$(grep "java.home" "$VSCODE_SETTINGS" | cut -d'"' -f4)
        print_status "INFO" "VSCode Java home: $VSCODE_JAVA_HOME"
        
        if [ -d "$VSCODE_JAVA_HOME" ]; then
            print_status "OK" "VSCode Java home directory exists"
        else
            print_status "ERROR" "VSCode Java home directory does not exist"
        fi
    else
        print_status "WARN" "VSCode Java home not configured"
    fi
else
    print_status "WARN" "VSCode settings file not found"
fi

# Check project compilation
print_status "INFO" "Testing project compilation..."
if [ -f "pom.xml" ]; then
    if mvn compile -q > /tmp/mvn_compile.log 2>&1; then
        print_status "OK" "Project compiles successfully"
    else
        print_status "ERROR" "Project compilation failed"
        echo "Compilation errors:"
        tail -n 10 /tmp/mvn_compile.log
    fi
else
    print_status "WARN" "No pom.xml found"
fi

# Recommendations
echo ""
print_status "INFO" "=== Recommendations ==="

if [ "${MAJOR_VERSION:-0}" -lt 17 ]; then
    echo "  • Install Java 17 or later"
    echo "  • Run: sudo apt install openjdk-17-jdk"
fi

if [ -z "$JAVA_HOME" ] || [ "${MAJOR_VERSION:-0}" -lt 17 ]; then
    echo "  • Set JAVA_HOME to Java 17+"
    echo "  • Run: export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64"
fi

if [ ! -f "$VSCODE_SETTINGS" ] || ! grep -q "java.home" "$VSCODE_SETTINGS"; then
    echo "  • Configure VSCode Java settings"
    echo "  • Run: ./scripts/setup_vscode_java.sh"
fi

echo ""
print_status "INFO" "Validation complete!"
