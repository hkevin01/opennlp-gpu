#!/bin/bash

# Quick AWS Setup Script for OpenNLP GPU Extension
# Optimized for AWS EC2 instances with minimal manual intervention

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 OpenNLP GPU Extension - AWS Quick Setup${NC}"
echo "============================================="

# Detect AWS instance type
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")

echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

# Check for GPU instances
if [[ "$INSTANCE_TYPE" == *"p2"* ]] || [[ "$INSTANCE_TYPE" == *"p3"* ]] || [[ "$INSTANCE_TYPE" == *"p4"* ]] || [[ "$INSTANCE_TYPE" == *"g3"* ]] || [[ "$INSTANCE_TYPE" == *"g4"* ]] || [[ "$INSTANCE_TYPE" == *"g5"* ]]; then
    echo -e "${GREEN}✅ GPU instance detected${NC}"
    GPU_INSTANCE=true
else
    echo -e "${BLUE}ℹ️  CPU instance - will run in CPU mode${NC}"
    GPU_INSTANCE=false
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -y > /dev/null 2>&1

# Install dependencies based on instance type
if [ "$GPU_INSTANCE" = true ]; then
    echo "🔧 Installing GPU dependencies..."
    
    # Install NVIDIA drivers if not present
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Installing NVIDIA drivers..."
        sudo apt-get install -y nvidia-driver-535 > /dev/null 2>&1 || true
        # May require reboot
    fi
    
    # Install CUDA if not present
    if ! command -v nvcc &> /dev/null; then
        echo "Installing CUDA toolkit..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb > /dev/null 2>&1
        sudo apt-get update > /dev/null 2>&1
        sudo apt-get install -y cuda-toolkit-12-2 > /dev/null 2>&1 || \
        sudo apt-get install -y nvidia-cuda-toolkit > /dev/null 2>&1
    fi
fi

# Install essential tools
echo "🛠️  Installing development tools..."
sudo apt-get install -y openjdk-21-jdk maven cmake build-essential git curl > /dev/null 2>&1

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc

# Clone or update repository if needed
if [ ! -d "opennlp-gpu" ]; then
    echo "📥 Repository not found locally - please ensure you're in the right directory"
    echo "   or clone the repository first"
    exit 1
fi

# Run main setup
echo "🚀 Running main setup script..."
cd opennlp-gpu 2>/dev/null || cd .
./setup.sh

echo -e "${GREEN}✅ AWS setup complete!${NC}"
echo "Run './gpu_demo.sh' to test the installation"
