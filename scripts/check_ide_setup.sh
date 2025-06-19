#!/bin/bash

# IDE Setup Checker
# Validates VSCode configuration for OpenNLP GPU development

set -e

echo "🔧 Checking IDE Setup for OpenNLP GPU Development..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check VSCode
echo ""
echo "🔍 Checking VSCode Installation:"
if command -v code >/dev/null 2>&1; then
    CODE_VERSION=$(code --version | head -1)
    print_status "VSCode found: $CODE_VERSION"
else
    print_warning "VSCode not found in PATH"
    print_info "Install VSCode: https://code.visualstudio.com/"
fi

# Check Java configuration
echo ""
echo "🔍 Checking Java Configuration:"
./scripts/validate_java_runtime.sh | tail -20

# Check VSCode extensions
echo ""
echo "🔍 Checking Required VSCode Extensions:"

REQUIRED_EXTENSIONS=(
    "vscjava.vscode-java-pack"
    "redhat.java" 
    "vscjava.vscode-maven"
    "redhat.vscode-xml"
    "ms-vscode.test-adapter-converter"
)

OPTIONAL_EXTENSIONS=(
    "github.copilot"
    "ms-vscode.vscode-json"
    "ms-python.python"
    "ms-toolsai.jupyter"
)

if command -v code >/dev/null 2>&1; then
    INSTALLED_EXTENSIONS=$(code --list-extensions 2>/dev/null || echo "")
    
    echo "Required Extensions:"
    for ext in "${REQUIRED_EXTENSIONS[@]}"; do
        if echo "$INSTALLED_EXTENSIONS" | grep -q "$ext"; then
            print_status "$ext"
        else
            print_error "$ext (missing)"
            echo "         Install: code --install-extension $ext"
        fi
    done
    
    echo ""
    echo "Optional Extensions:"
    for ext in "${OPTIONAL_EXTENSIONS[@]}"; do
        if echo "$INSTALLED_EXTENSIONS" | grep -q "$ext"; then
            print_status "$ext"
        else
            print_info "$ext (optional)"
        fi
    done
else
    print_warning "Cannot check extensions - VSCode not found"
fi

# Check project structure
echo ""
echo "🔍 Checking Project Structure:"

REQUIRED_FILES=(
    "pom.xml"
    "src/main/java"
    "src/test/java"
    ".vscode/settings.json"
    ".vscode/tasks.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -e "$file" ]; then
        print_status "$file"
    else
        print_error "$file (missing)"
    fi
done

# Check VSCode workspace settings
echo ""
echo "🔍 Checking VSCode Workspace Settings:"

if [ -f ".vscode/settings.json" ]; then
    print_status "VSCode settings file exists"
    
    # Check critical settings
    if grep -q "java.home" .vscode/settings.json; then
        JAVA_HOME_SETTING=$(grep "java.home" .vscode/settings.json | sed 's/.*: *"\([^"]*\)".*/\1/')
        print_info "java.home: $JAVA_HOME_SETTING"
        
        if [ -f "$JAVA_HOME_SETTING/bin/java" ]; then
            print_status "Java home points to valid installation"
        else
            print_error "Java home points to invalid installation"
        fi
    else
        print_warning "java.home not configured"
    fi
    
    if grep -q "xml.java.home" .vscode/settings.json; then
        print_status "XML Java home configured"
    else
        print_warning "XML Java home not configured"
    fi
    
    if grep -q "xml.server.preferBinary.*false" .vscode/settings.json; then
        print_status "XML server configured to use Java (not binary)"
    else
        print_warning "XML server may use binary server"
    fi
else
    print_error "VSCode settings file missing"
    print_info "Run: ./scripts/fix_java_environment.sh"
fi

# Check Maven configuration
echo ""
echo "🔍 Checking Maven Configuration:"

if command -v mvn >/dev/null 2>&1; then
    print_status "Maven found"
    
    # Test Maven with Java
    if mvn --version >/dev/null 2>&1; then
        MVN_JAVA=$(mvn --version | grep "Java version" | cut -d: -f2 | tr -d ' ')
        print_info "Maven Java version: $MVN_JAVA"
        
        MVN_JAVA_HOME=$(mvn --version | grep "Java home" | cut -d: -f2 | tr -d ' ')
        print_info "Maven Java home: $MVN_JAVA_HOME"
    else
        print_error "Maven configuration issue"
    fi
    
    # Check if project compiles
    echo ""
    print_info "Testing Maven compilation..."
    if mvn compile -q >/dev/null 2>&1; then
        print_status "Project compiles successfully"
    else
        print_warning "Project compilation issues detected"
        print_info "Run: mvn compile for details"
    fi
else
    print_error "Maven not found"
    print_info "Install Maven: sudo apt install maven"
fi

# Check Git configuration
echo ""
echo "🔍 Checking Git Configuration:"

if command -v git >/dev/null 2>&1; then
    print_status "Git found"
    
    if git config user.name >/dev/null 2>&1; then
        GIT_USER=$(git config user.name)
        print_info "Git user: $GIT_USER"
    else
        print_warning "Git user not configured"
    fi
    
    if git config user.email >/dev/null 2>&1; then
        GIT_EMAIL=$(git config user.email)
        print_info "Git email: $GIT_EMAIL"
    else
        print_warning "Git email not configured"
    fi
else
    print_warning "Git not found"
fi

# Performance recommendations
echo ""
echo "🚀 Performance Recommendations:"

# Check available memory
MEMORY_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEMORY_GB=$((MEMORY_KB / 1024 / 1024))

if [ $MEMORY_GB -ge 8 ]; then
    print_status "Sufficient memory: ${MEMORY_GB}GB"
else
    print_warning "Low memory: ${MEMORY_GB}GB (8GB+ recommended)"
fi

# Check Java heap settings
if grep -q "java.jdt.ls.vmargs" .vscode/settings.json 2>/dev/null; then
    print_status "Java Language Server heap configured"
else
    print_info "Consider configuring Java LS heap:"
    echo '         "java.jdt.ls.vmargs": "-Xmx2G"'
fi

# Summary
echo ""
echo "📋 IDE Setup Summary:"

# Count issues
ERRORS=$(grep -c "❌" /tmp/ide_check.log 2>/dev/null || echo "0")
WARNINGS=$(grep -c "⚠️" /tmp/ide_check.log 2>/dev/null || echo "0")

if [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    print_status "IDE setup is optimal!"
elif [ "$ERRORS" -eq 0 ]; then
    print_warning "IDE setup is good with minor issues"
else
    print_error "IDE setup has critical issues that should be fixed"
fi

echo ""
print_info "To fix all issues automatically:"
echo "  ./scripts/fix_java_environment.sh"
echo ""
print_info "To install missing VSCode extensions:"
echo "  ./scripts/install_vscode_extensions.sh"

echo ""
print_info "IDE setup check complete!"
