#!/bin/bash

# Java Runtime Validation Script for OpenNLP GPU Project
# This script validates that Java is properly configured for both terminal and VSCode

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}                    Java Environment Validation                    ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo -e "\n${BLUE}── $1 ──${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Initialize variables
ERRORS=0
WARNINGS=0

# Function to record error
record_error() {
    print_error "$1"
    ((ERRORS++))
}

# Function to record warning
record_warning() {
    print_warning "$1"
    ((WARNINGS++))
}

print_header

print_section "System Java Installation"

# Check if Java is available
if command -v java &> /dev/null; then
    JAVA_VERSION_OUTPUT=$(java -version 2>&1)
    JAVA_VERSION=$(echo "$JAVA_VERSION_OUTPUT" | head -n1 | sed 's/.*version "\([^"]*\)".*/\1/')
    JAVA_MAJOR_VERSION=$(echo "$JAVA_VERSION" | cut -d'.' -f1)
    
    echo "Java Version Output:"
    echo "$JAVA_VERSION_OUTPUT" | head -n3
    echo ""
    
    if [ "$JAVA_MAJOR_VERSION" -ge 11 ]; then
        print_success "Java $JAVA_VERSION detected (compatible)"
    else
        record_error "Java $JAVA_VERSION is too old (need Java 11+)"
    fi
else
    record_error "Java command not found"
fi

# Check JavaC
if command -v javac &> /dev/null; then
    JAVAC_VERSION=$(javac -version 2>&1)
    print_success "Java Compiler: $JAVAC_VERSION"
else
    record_warning "Java Compiler (javac) not found"
fi

print_section "Environment Variables"

# Check JAVA_HOME
if [ -n "$JAVA_HOME" ]; then
    print_info "JAVA_HOME: $JAVA_HOME"
    
    if [ -d "$JAVA_HOME" ]; then
        print_success "JAVA_HOME directory exists"
        
        if [ -f "$JAVA_HOME/bin/java" ]; then
            JAVA_HOME_VERSION=$("$JAVA_HOME/bin/java" -version 2>&1 | head -n1 | sed 's/.*version "\([^"]*\)".*/\1/')
            JAVA_HOME_MAJOR=$(echo "$JAVA_HOME_VERSION" | cut -d'.' -f1)
            
            if [ "$JAVA_HOME_MAJOR" -ge 11 ]; then
                print_success "JAVA_HOME points to Java $JAVA_HOME_VERSION (compatible)"
            else
                record_error "JAVA_HOME points to Java $JAVA_HOME_VERSION (too old)"
            fi
        else
            record_error "JAVA_HOME/bin/java not found"
        fi
    else
        record_error "JAVA_HOME directory does not exist"
    fi
else
    record_warning "JAVA_HOME not set"
fi

print_section "Available Java Installations"

if command -v update-alternatives &> /dev/null; then
    JAVA_ALTERNATIVES=$(update-alternatives --list java 2>/dev/null || true)
    if [ -n "$JAVA_ALTERNATIVES" ]; then
        echo "Available Java installations:"
        while IFS= read -r java_path; do
            if [ -f "$java_path" ]; then
                VERSION_OUTPUT=$("$java_path" -version 2>&1 | head -n1)
                echo "  $java_path -> $VERSION_OUTPUT"
            fi
        done <<< "$JAVA_ALTERNATIVES"
        
        # Check for Java 17 and 21
        if echo "$JAVA_ALTERNATIVES" | grep -q "java-17"; then
            print_success "Java 17 available"
        else
            record_warning "Java 17 not found in alternatives"
        fi
        
        if echo "$JAVA_ALTERNATIVES" | grep -q "java-21"; then
            print_info "Java 21 available"
        fi
    else
        record_warning "No Java alternatives found"
    fi
else
    record_warning "update-alternatives command not available"
fi

print_section "Maven Configuration"

if command -v mvn &> /dev/null; then
    MVN_VERSION=$(mvn -version 2>&1 | head -n1)
    print_success "Maven: $MVN_VERSION"
    
    # Check Maven's Java version
    MVN_JAVA_INFO=$(mvn -version 2>&1 | grep "Java version")
    print_info "$MVN_JAVA_INFO"
    
    # Check if Maven can detect the project
    if [ -f "pom.xml" ]; then
        print_success "Maven POM found"
        
        # Test Maven compilation (quick check)
        print_info "Testing Maven compilation..."
        if mvn clean compile -q > /dev/null 2>&1; then
            print_success "Maven compilation successful"
        else
            record_warning "Maven compilation failed (run 'mvn clean compile' for details)"
        fi
    else
        record_warning "No pom.xml found in current directory"
    fi
else
    record_warning "Maven not found"
fi

print_section "VSCode Configuration"

# Check VSCode settings
if [ -f ".vscode/settings.json" ]; then
    print_success "VSCode settings.json found"
    
    # Check Java home in VSCode
    VSCODE_JAVA_HOME=$(grep -o '"java.home"[[:space:]]*:[[:space:]]*"[^"]*"' .vscode/settings.json | sed 's/.*"java.home"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || true)
    if [ -n "$VSCODE_JAVA_HOME" ]; then
        print_info "VSCode java.home: $VSCODE_JAVA_HOME"
        
        if [ -d "$VSCODE_JAVA_HOME" ]; then
            if [ -f "$VSCODE_JAVA_HOME/bin/java" ]; then
                VSCODE_JAVA_VERSION=$("$VSCODE_JAVA_HOME/bin/java" -version 2>&1 | head -n1 | sed 's/.*version "\([^"]*\)".*/\1/')
                VSCODE_JAVA_MAJOR=$(echo "$VSCODE_JAVA_VERSION" | cut -d'.' -f1)
                
                if [ "$VSCODE_JAVA_MAJOR" -ge 11 ]; then
                    print_success "VSCode configured with Java $VSCODE_JAVA_VERSION (compatible)"
                else
                    record_error "VSCode configured with Java $VSCODE_JAVA_VERSION (too old)"
                fi
            else
                record_error "VSCode java.home/bin/java not found"
            fi
        else
            record_error "VSCode java.home directory does not exist"
        fi
    else
        record_warning "VSCode java.home not configured"
    fi
    
    # Check XML java home
    XML_JAVA_HOME=$(grep -o '"xml.java.home"[[:space:]]*:[[:space:]]*"[^"]*"' .vscode/settings.json | sed 's/.*"xml.java.home"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' || true)
    if [ -n "$XML_JAVA_HOME" ]; then
        print_info "VSCode xml.java.home: $XML_JAVA_HOME"
        if [ -d "$XML_JAVA_HOME" ]; then
            print_success "XML Java home directory exists"
        else
            record_error "XML Java home directory does not exist"
        fi
    else
        record_warning "VSCode xml.java.home not configured (may cause XML server issues)"
    fi
else
    record_warning "VSCode settings.json not found"
fi

print_section "Project Environment"

# Check .env file
if [ -f ".env" ]; then
    print_success "Project .env file found"
    print_info "Environment configuration:"
    grep "^export" .env || true
else
    record_warning "Project .env file not found"
fi

# Check tasks.json for Java configuration
if [ -f ".vscode/tasks.json" ]; then
    print_success "VSCode tasks.json found"
    
    # Check if Java environment is set in tasks
    if grep -q "JAVA_HOME" .vscode/tasks.json; then
        print_success "JAVA_HOME configured in VSCode tasks"
    else
        record_warning "JAVA_HOME not set in VSCode tasks"
    fi
else
    record_warning "VSCode tasks.json not found"
fi

print_section "Recommendations"

if [ $ERRORS -gt 0 ]; then
    echo ""
    print_error "Found $ERRORS error(s) that need to be fixed:"
    echo ""
    echo "To fix Java issues:"
    echo "  1. Run: ./scripts/setup-java-environment.sh"
    echo "  2. Restart VSCode"
    echo "  3. Run this validation again"
    echo ""
fi

if [ $WARNINGS -gt 0 ]; then
    echo ""
    print_warning "Found $WARNINGS warning(s) - consider addressing these:"
    echo ""
    echo "To optimize setup:"
    echo "  1. Install missing components (Maven, etc.)"
    echo "  2. Update VSCode settings"
    echo "  3. Set up project environment file"
    echo ""
fi

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo ""
    print_success "All checks passed! Java environment is properly configured."
    echo ""
fi

print_section "Summary"

echo "Validation Results:"
if [ $ERRORS -eq 0 ]; then
    print_success "$ERRORS errors"
else
    print_error "$ERRORS errors"
fi

if [ $WARNINGS -eq 0 ]; then
    print_success "$WARNINGS warnings"
else
    print_warning "$WARNINGS warnings"
fi

if [ $ERRORS -eq 0 ]; then
    print_success "Java environment is ready for OpenNLP GPU development!"
    exit 0
else
    print_error "Please fix the errors above before proceeding."
    exit 1
fi
