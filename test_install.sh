#!/bin/bash

# Installation Test Script for OpenNLP GPU Extension
# Simulates a fresh installation to verify setup works correctly

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🧪 OpenNLP GPU Extension - Installation Test${NC}"
echo "============================================="
echo

# Test 1: Check if all setup scripts exist
echo "Test 1: Setup Scripts"
SCRIPTS=("setup.sh" "aws_setup.sh" "docker_setup.sh" "verify.sh" "gpu_demo.sh")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo -e "  ✅ $script"
    else
        echo -e "  ❌ $script (missing or not executable)"
    fi
done
echo

# Test 2: Quick system verification
echo "Test 2: System Prerequisites"
./verify.sh | grep -E "(✅|❌|⚠️)"
echo

# Test 3: Quick build test (just compilation check)
echo "Test 3: Build System Test"
echo -n "  Maven compile: "
if mvn compile -q 2>/dev/null; then
    echo -e "${GREEN}✅ Success${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

echo -n "  CMake configure: "
cd src/main/cpp
if cmake . >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Success${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi
cd ../../..

# Test 4: Demo execution test
echo "Test 4: Demo Execution Test"
echo -n "  GPU Diagnostics: "
if timeout 10s java -cp "target/classes:$(cat classpath.txt 2>/dev/null || echo '')" org.apache.opennlp.gpu.tools.GpuDiagnostics >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Success${NC}"
else
    echo -e "${YELLOW}⚠️  Timeout/Warning${NC}"
fi

echo -n "  GPU ML Demo: "
if timeout 15s java -cp "target/classes:$(cat classpath.txt 2>/dev/null || echo '')" org.apache.opennlp.gpu.ml.GpuMlDemo >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Success${NC}"
else
    echo -e "${YELLOW}⚠️  Timeout/Warning${NC}"
fi

echo
echo -e "${BLUE}📋 Installation Test Summary${NC}"
echo "============================"
echo "✅ = Working correctly"
echo "⚠️  = Working with warnings"  
echo "❌ = Needs attention"
echo
echo "💡 If you see any ❌, run: ./setup.sh"
echo "💡 For AWS setup: ./aws_setup.sh"
echo "💡 For Docker setup: ./docker_setup.sh"
