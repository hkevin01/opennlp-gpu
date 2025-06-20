#!/bin/bash

# Java Runtime Validation Script
# Ensures proper Java environment for OpenNLP GPU project

set -e

echo "🔍 Validating Java Runtime Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check Java installations
echo ""
echo "🔍 Available Java Installations:"
if command -v update-java-alternatives >/dev/null 2>&1; then
    update-java-alternatives --list
else
    echo "System Java alternatives not available, checking manually..."
fi

# Check current Java version
echo ""
echo "🔍 Current Java Version:"
if command -v java >/dev/null 2>&1; then
    java -version
    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
    JAVA_MAJOR=$(echo $JAVA_VERSION | cut -d. -f1)
    
    if [ "$JAVA_MAJOR" -ge 17 ]; then
        print_status "Java $JAVA_VERSION is compatible (>= 17)"
    elif [ "$JAVA_MAJOR" -ge 11 ]; then
        print_warning "Java $JAVA_VERSION works but Java 17+ recommended"
    else
        print_error "Java $JAVA_VERSION is too old (< 11). VSCode XML will use binary server."
    fi
else
    print_error "Java not found in PATH"
    exit 1
fi

# Check JAVA_HOME
echo ""
echo "🔍 JAVA_HOME Configuration:"
if [ -n "$JAVA_HOME" ]; then
    print_info "JAVA_HOME: $JAVA_HOME"
    if [ -f "$JAVA_HOME/bin/java" ]; then
        print_status "JAVA_HOME points to valid Java installation"
        JAVA_HOME_VERSION=$($JAVA_HOME/bin/java -version 2>&1 | awk -F '"' '/version/ {print $2}')
        print_info "JAVA_HOME Java version: $JAVA_HOME_VERSION"
    else
        print_error "JAVA_HOME does not contain valid Java installation"
    fi
else
    print_warning "JAVA_HOME not set"
fi

# Check preferred Java installations
echo ""
echo "🔍 Checking for Java 17 and 21:"

JAVA_17_PATH=""
JAVA_21_PATH=""

# Check common Java installation paths
for java_path in /usr/lib/jvm/java-17-openjdk-amd64 /usr/lib/jvm/java-21-openjdk-amd64 /usr/lib/jvm/temurin-17-jdk-amd64 /usr/lib/jvm/temurin-21-jdk-amd64; do
    if [ -d "$java_path" ] && [ -f "$java_path/bin/java" ]; then
        version=$($java_path/bin/java -version 2>&1 | awk -F '"' '/version/ {print $2}')
        major=$(echo $version | cut -d. -f1)
        
        if [ "$major" = "17" ]; then
            JAVA_17_PATH="$java_path"
            print_status "Found Java 17: $java_path"
        elif [ "$major" = "21" ]; then
            JAVA_21_PATH="$java_path"
            print_status "Found Java 21: $java_path"
        fi
    fi
done

# Determine best Java to use
RECOMMENDED_JAVA=""
if [ -n "$JAVA_21_PATH" ]; then
    RECOMMENDED_JAVA="$JAVA_21_PATH"
    print_info "Recommended: Java 21 at $JAVA_21_PATH"
elif [ -n "$JAVA_17_PATH" ]; then
    RECOMMENDED_JAVA="$JAVA_17_PATH"
    print_info "Recommended: Java 17 at $JAVA_17_PATH"
else
    print_error "No suitable Java version (17 or 21) found"
    exit 1
fi

# Check VSCode settings
echo ""
echo "🔍 VSCode Java Configuration:"
VSCODE_SETTINGS=".vscode/settings.json"

if [ -f "$VSCODE_SETTINGS" ]; then
    print_status "VSCode settings file exists"
    
    # Check if Java home is configured
    if grep -q "java.home" "$VSCODE_SETTINGS"; then
        VSCODE_JAVA_HOME=$(grep "java.home" "$VSCODE_SETTINGS" | sed 's/.*: *"\([^"]*\)".*/\1/')
        print_info "VSCode java.home: $VSCODE_JAVA_HOME"
        
        if [ "$VSCODE_JAVA_HOME" = "$RECOMMENDED_JAVA" ]; then
            print_status "VSCode is using recommended Java version"
        else
            print_warning "VSCode is using different Java version"
        fi
    else
        print_warning "VSCode java.home not configured"
    fi
    
    # Check if jdt.ls.java.home is configured
    if grep -q "java.jdt.ls.java.home" "$VSCODE_SETTINGS"; then
        VSCODE_JDT_HOME=$(grep "java.jdt.ls.java.home" "$VSCODE_SETTINGS" | sed 's/.*: *"\([^"]*\)".*/\1/')
        print_info "VSCode java.jdt.ls.java.home: $VSCODE_JDT_HOME"
    fi
else
    print_warning "VSCode settings file not found"
fi

# Check Maven configuration
echo ""
echo "🔍 Maven Java Configuration:"
if command -v mvn >/dev/null 2>&1; then
    MVN_JAVA_HOME=$(mvn help:system | grep "java.home" | head -1 | cut -d= -f2 | tr -d ' ')
    print_info "Maven java.home: $MVN_JAVA_HOME"
    
    if [[ "$MVN_JAVA_HOME" == *"java-17"* ]] || [[ "$MVN_JAVA_HOME" == *"java-21"* ]]; then
        print_status "Maven is using modern Java version"
    else
        print_warning "Maven may be using older Java version"
    fi
else
    print_warning "Maven not found"
fi

# Summary and recommendations
echo ""
echo "📋 Summary and Recommendations:"

if [ "$JAVA_MAJOR" -ge 17 ] && [ "$JAVA_HOME" = "$RECOMMENDED_JAVA" ]; then
    print_status "Java environment is properly configured!"
else
    echo ""
    print_info "To fix Java configuration, run:"
    echo "  export JAVA_HOME=$RECOMMENDED_JAVA"
    echo "  export PATH=\$JAVA_HOME/bin:\$PATH"
    echo ""
    print_info "To make it permanent, add to ~/.bashrc:"
    echo "  echo 'export JAVA_HOME=$RECOMMENDED_JAVA' >> ~/.bashrc"
    echo "  echo 'export PATH=\$JAVA_HOME/bin:\$PATH' >> ~/.bashrc"
    echo ""
    print_info "Or run the auto-fix script:"
    echo "  ./scripts/fix_java_environment.sh"
fi

echo ""
print_info "Validation complete!"
