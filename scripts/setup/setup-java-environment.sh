#!/bin/bash

# Setup Java Environment for OpenNLP GPU Project
# This script ensures proper Java 17+ runtime is configured for both terminal and VSCode

set -e

echo "🔧 Setting up Java environment for OpenNLP GPU project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check available Java versions
echo ""
print_status "Checking available Java installations..."
if ! command -v update-alternatives &> /dev/null; then
    print_error "update-alternatives not found. Please install Java manually."
    exit 1
fi

# List available Java versions
JAVA_VERSIONS=$(update-alternatives --list java 2>/dev/null || true)
if [ -z "$JAVA_VERSIONS" ]; then
    print_error "No Java installations found!"
    echo "Please install Java 17 or higher:"
    echo "  sudo apt update"
    echo "  sudo apt install openjdk-17-jdk"
    exit 1
fi

echo "Available Java versions:"
echo "$JAVA_VERSIONS"

# Find Java 17 or higher
JAVA_17_PATH=""
JAVA_21_PATH=""

while IFS= read -r java_path; do
    if [[ "$java_path" == *"java-17-openjdk"* ]]; then
        JAVA_17_PATH="$java_path"
    elif [[ "$java_path" == *"java-21-openjdk"* ]]; then
        JAVA_21_PATH="$java_path"
    fi
done <<< "$JAVA_VERSIONS"

# Choose the best Java version (prefer 17 for stability)
SELECTED_JAVA=""
SELECTED_JAVA_HOME=""

if [ -n "$JAVA_17_PATH" ]; then
    SELECTED_JAVA="$JAVA_17_PATH"
    SELECTED_JAVA_HOME="${JAVA_17_PATH%/bin/java}"
    print_success "Selected Java 17: $SELECTED_JAVA"
elif [ -n "$JAVA_21_PATH" ]; then
    SELECTED_JAVA="$JAVA_21_PATH"
    SELECTED_JAVA_HOME="${JAVA_21_PATH%/bin/java}"
    print_success "Selected Java 21: $SELECTED_JAVA"
else
    print_error "No Java 17+ found! Available versions:"
    echo "$JAVA_VERSIONS"
    echo ""
    echo "Please install Java 17:"
    echo "  sudo apt update"
    echo "  sudo apt install openjdk-17-jdk"
    exit 1
fi

# Verify the selected Java version
print_status "Verifying Java version..."
JAVA_VERSION_OUTPUT=$("$SELECTED_JAVA" -version 2>&1)
echo "$JAVA_VERSION_OUTPUT"

# Extract version number
JAVA_VERSION=$(echo "$JAVA_VERSION_OUTPUT" | head -n1 | sed 's/.*version "\([^"]*\)".*/\1/')
JAVA_MAJOR_VERSION=$(echo "$JAVA_VERSION" | cut -d'.' -f1)

if [ "$JAVA_MAJOR_VERSION" -lt 11 ]; then
    print_error "Java version $JAVA_VERSION is too old. Need Java 11+."
    exit 1
fi

print_success "Java version $JAVA_VERSION is compatible."

# Set up project-specific environment
print_status "Setting up project environment..."

# Create project-specific environment file
cat > .env << EOF
# Java Environment Configuration for OpenNLP GPU
# Generated on $(date)

# Java Runtime Configuration
export JAVA_HOME="$SELECTED_JAVA_HOME"
export PATH="$SELECTED_JAVA_HOME/bin:\$PATH"

# Maven Configuration
export MAVEN_OPTS="-Xmx2g -XX:+UseG1GC"

# Project Configuration
export OPENNLP_GPU_JAVA_HOME="$SELECTED_JAVA_HOME"
export OPENNLP_GPU_JAVA_VERSION="$JAVA_VERSION"

# VSCode Java Configuration
export VSCODE_JAVA_HOME="$SELECTED_JAVA_HOME"
EOF

# Source the environment
source .env

print_success "Created .env file with Java configuration"

# Verify the setup
print_status "Verifying setup..."
echo "JAVA_HOME: $JAVA_HOME"
echo "Java version: $(java -version 2>&1 | head -n1)"
echo "JavaC version: $(javac -version 2>&1)"

# Check Maven
if command -v mvn &> /dev/null; then
    echo "Maven version: $(mvn -version | head -n1)"
    print_success "Maven is available"
else
    print_warning "Maven not found. Please install Maven."
fi

print_success "Java environment setup completed!"
echo ""
echo "To use this environment:"
echo "  source .env"
echo ""
echo "Or run this script: source scripts/setup-java-environment.sh"
