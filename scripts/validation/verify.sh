#!/bin/bash

# Quick verification script for OpenNLP GPU Extension
# Checks if everything is properly installed and working

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 OpenNLP GPU Extension - System Verification${NC}"
echo "=============================================="
echo

# Check Java
echo -n "Java 21+: "
if command -v java &> /dev/null; then
    JAVA_VER=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
    if [ "$JAVA_VER" -ge 21 ]; then
        echo -e "${GREEN}✅ Java $JAVA_VER${NC}"
    else
        echo -e "${RED}❌ Java $JAVA_VER (need 21+)${NC}"
    fi
else
    echo -e "${RED}❌ Not found${NC}"
fi

# Check Maven
echo -n "Maven: "
if command -v mvn &> /dev/null; then
    MVN_VER=$(mvn --version 2>/dev/null | head -n 1 | cut -d' ' -f3)
    echo -e "${GREEN}✅ $MVN_VER${NC}"
else
    echo -e "${RED}❌ Not found${NC}"
fi

# Check CMake
echo -n "CMake 3.16+: "
if command -v cmake &> /dev/null; then
    CMAKE_VER=$(cmake --version | head -n 1 | cut -d' ' -f3)
    echo -e "${GREEN}✅ $CMAKE_VER${NC}"
else
    echo -e "${RED}❌ Not found${NC}"
fi

# Check GPU
echo -n "GPU Support: "
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n 1)
    echo -e "${GREEN}✅ NVIDIA: $GPU_INFO${NC}"
elif command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}✅ AMD ROCm${NC}"
elif [ -d "/opt/rocm" ]; then
    echo -e "${GREEN}✅ AMD ROCm (installed)${NC}"
else
    echo -e "${YELLOW}⚠️  CPU-only mode${NC}"
fi

# Check build status
echo -n "Native Library: "
if [ -f "src/main/cpp/libopennlp_gpu.so" ]; then
    echo -e "${GREEN}✅ Built${NC}"
else
    echo -e "${RED}❌ Not built${NC}"
fi

echo -n "Java Project: "
if [ -d "target/classes" ]; then
    echo -e "${GREEN}✅ Built${NC}"
else
    echo -e "${RED}❌ Not built${NC}"
fi

echo
echo "💡 Quick Actions:"
echo "  Run setup:     ./setup.sh"
echo "  Run demo:      ./gpu_demo.sh"
echo "  Build native:  cd src/main/cpp && cmake . && make"
echo "  Build Java:    mvn clean compile"
