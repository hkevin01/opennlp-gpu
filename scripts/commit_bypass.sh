#!/bin/bash
# Emergency commit script that bypasses pre-commit hooks
# Customized for OpenNLP GPU project
# Use only when pre-commit is broken and you need to commit urgently

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

PROJECT_NAME="OpenNLP GPU"
PROJECT_PREFIX="[GPU]"

# Default commit message if none provided
DEFAULT_COMMIT_MESSAGE="Emergency commit - bypass pre-commit hooks"

if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No commit message provided. Using default: '$DEFAULT_COMMIT_MESSAGE'${NC}"
    echo -e "${BLUE}Usage: $0 'commit message' [--fix-java] [--check-license] [--phase-tag]${NC}"
    echo -e "${YELLOW}OpenNLP GPU Project Options:${NC}"
    echo -e "  --fix-java        Fix Java code formatting and imports"
    echo -e "  --check-license   Verify Apache license headers are present"
    echo -e "  --phase-tag       Auto-tag commit with current project phase"
    echo -e "  --gpu-test        Run quick GPU availability test before commit"
    echo -e "\n${BLUE}Examples:${NC}"
    echo -e "${YELLOW}  $0 'Add matrix operations' --fix-java --phase-tag${NC}"
    echo -e "${YELLOW}  $0 'Fix memory leak in GPU provider' --check-license${NC}"
    echo -e "${YELLOW}  $0  # Uses default commit message${NC}"
    
    COMMIT_MESSAGE="$DEFAULT_COMMIT_MESSAGE"
else
    COMMIT_MESSAGE="$1"
fi

FIX_JAVA=false
CHECK_LICENSE=false
PHASE_TAG=false
GPU_TEST=false

# Parse all arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix-java)
            FIX_JAVA=true
            shift
            ;;
        --check-license)
            CHECK_LICENSE=true
            shift
            ;;
        --phase-tag)
            PHASE_TAG=true
            shift
            ;;
        --gpu-test)
            GPU_TEST=true
            shift
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            shift
            ;;
    esac
done

echo -e "${PURPLE}🚀 $PROJECT_NAME Emergency Commit Tool${NC}"
echo -e "${YELLOW}⚠️  Bypassing pre-commit hooks for emergency commit${NC}"

# Check if we're in the right project directory
if [ ! -f "pom.xml" ] || ! grep -q "opennlp-gpu" pom.xml 2>/dev/null; then
    echo -e "${RED}❌ Error: Not in OpenNLP GPU project root directory${NC}"
    echo -e "${YELLOW}💡 Please run this script from the project root where pom.xml is located${NC}"
    exit 1
fi

# Quick GPU test if requested
if [ "$GPU_TEST" = true ]; then
    echo -e "${BLUE}🔍 Running quick GPU availability test...${NC}"
    
    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}  ✓ NVIDIA GPU detected${NC}"
    else
        echo -e "${YELLOW}  ⚠ NVIDIA GPU not detected or drivers not installed${NC}"
    fi
    
    # Check if OpenCL is available (basic check)
    if [ -d "/usr/lib/x86_64-linux-gnu" ] && ls /usr/lib/x86_64-linux-gnu/libOpenCL* &> /dev/null; then
        echo -e "${GREEN}  ✓ OpenCL libraries found${NC}"
    else
        echo -e "${YELLOW}  ⚠ OpenCL libraries not found${NC}"
    fi
fi

# Fix Java code formatting if requested
if [ "$FIX_JAVA" = true ]; then
    echo -e "${BLUE}☕ Fixing Java code formatting and imports...${NC}"

    # Get list of staged Java files
    JAVA_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.java$')

    if [ -n "$JAVA_FILES" ]; then
        echo "$JAVA_FILES" | while IFS= read -r file; do
            if [ -f "$file" ]; then
                # Remove trailing whitespace
                sed -i 's/[[:space:]]*$//' "$file"
                
                # Ensure consistent line endings
                sed -i -e '$a\' "$file"
                
                # Fix common Java formatting issues
                # Fix spacing around operators
                sed -i 's/\([^=!<>]\)=\([^=]\)/\1 = \2/g' "$file"
                sed -i 's/\([^=!<>]\)==\([^=]\)/\1 == \2/g' "$file"
                
                # Ensure proper spacing after commas
                sed -i 's/,\([^ ]\)/, \1/g' "$file"
                
                # Fix spacing around opening braces
                sed -i 's/){/) {/g' "$file"
                sed -i 's/else{/else {/g' "$file"
                
                echo -e "${GREEN}  ✓ Fixed: $file${NC}"
            fi
        done

        # Re-stage the fixed files
        echo "$JAVA_FILES" | xargs git add
        echo -e "${GREEN}✅ Java formatting fixed and files re-staged${NC}"
    else
        echo -e "${YELLOW}No staged Java files found${NC}"
    fi
fi

# Check Apache license headers if requested
if [ "$CHECK_LICENSE" = true ]; then
    echo -e "${BLUE}📜 Checking Apache license headers...${NC}"
    
    JAVA_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.java$')
    MISSING_LICENSE=false
    
    if [ -n "$JAVA_FILES" ]; then
        echo "$JAVA_FILES" | while IFS= read -r file; do
            if [ -f "$file" ] && ! grep -q "Licensed to the Apache Software Foundation" "$file"; then
                echo -e "${YELLOW}  ⚠ Missing license header: $file${NC}"
                MISSING_LICENSE=true
            fi
        done
        
        if [ "$MISSING_LICENSE" = true ]; then
            echo -e "${YELLOW}💡 Consider adding Apache license headers to new files${NC}"
            echo -e "${YELLOW}   Template available in existing project files${NC}"
        else
            echo -e "${GREEN}✅ All Java files have proper license headers${NC}"
        fi
    fi
fi

# Auto-tag with project phase if requested
if [ "$PHASE_TAG" = true ]; then
    echo -e "${BLUE}🏷️  Auto-tagging with project phase...${NC}"
    
    # Determine current phase based on file changes
    STAGED_FILES=$(git diff --cached --name-only)
    PHASE=""
    
    if echo "$STAGED_FILES" | grep -q "ml/.*\.java"; then
        PHASE="PHASE2-ML"
    elif echo "$STAGED_FILES" | grep -q "compute/.*\.java"; then
        PHASE="PHASE2-COMPUTE"
    elif echo "$STAGED_FILES" | grep -q "common/.*\.java"; then
        PHASE="PHASE2-COMMON"
    elif echo "$STAGED_FILES" | grep -q "test/.*\.java"; then
        PHASE="PHASE3-TEST"
    elif echo "$STAGED_FILES" | grep -q "docs/.*\.md"; then
        PHASE="PHASE3-DOCS"
    else
        PHASE="PHASE2-CORE"
    fi
    
    COMMIT_MESSAGE="$PROJECT_PREFIX [$PHASE] $COMMIT_MESSAGE"
    echo -e "${GREEN}  ✓ Tagged as: $PHASE${NC}"
fi

# Check for large files that might cause issues
echo -e "${BLUE}📦 Checking for large files...${NC}"
LARGE_FILES=$(git diff --cached --name-only | xargs -I {} find {} -size +10M 2>/dev/null)
if [ -n "$LARGE_FILES" ]; then
    echo -e "${YELLOW}⚠️  Large files detected:${NC}"
    echo "$LARGE_FILES" | while IFS= read -r file; do
        SIZE=$(du -h "$file" | cut -f1)
        echo -e "${YELLOW}  $file ($SIZE)${NC}"
    done
    echo -e "${YELLOW}💡 Consider using Git LFS for large files${NC}"
fi

# Check for sensitive files
echo -e "${BLUE}🔒 Checking for sensitive files...${NC}"
SENSITIVE_PATTERNS="*.key *.p12 *.jks *.keystore *.hprof java_pid*.log"
for pattern in $SENSITIVE_PATTERNS; do
    if git diff --cached --name-only | grep -q "$pattern"; then
        echo -e "${RED}❌ Sensitive file detected: $pattern${NC}"
        echo -e "${RED}   This file should not be committed to version control${NC}"
        exit 1
    fi
done

# Perform the commit
echo -e "${BLUE}💾 Performing emergency commit...${NC}"
git commit --no-verify -m "$COMMIT_MESSAGE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Emergency commit completed for $PROJECT_NAME${NC}"
    echo -e "${BLUE}📝 Commit message: '$COMMIT_MESSAGE'${NC}"
    
    # Show helpful next steps
    echo -e "\n${PURPLE}📋 Next Steps:${NC}"
    echo -e "${YELLOW}1. Verify commit: git log --oneline -1${NC}"
    echo -e "${YELLOW}2. Run tests: mvn test${NC}"
    echo -e "${YELLOW}3. Check build: mvn compile${NC}"
    
    if [ "$FIX_JAVA" = false ]; then
        echo -e "${YELLOW}4. Fix pre-commit issues and amend if needed${NC}"
    fi
    
    # Project-specific reminders
    echo -e "\n${PURPLE}🔧 OpenNLP GPU Reminders:${NC}"
    echo -e "${YELLOW}• Update project progress docs if this completes a milestone${NC}"
    echo -e "${YELLOW}• Test on both CPU and GPU if hardware-related changes${NC}"
    echo -e "${YELLOW}• Update user guide if API changes were made${NC}"
    
else
    echo -e "${RED}❌ Commit failed${NC}"
    echo -e "${YELLOW}💡 Check for remaining issues and try again${NC}"
    exit 1
fi
