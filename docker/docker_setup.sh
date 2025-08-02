#!/bin/bash

# Docker-based Setup for OpenNLP GPU Extension
# Provides isolated environment for easy setup

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🐳 OpenNLP GPU Extension - Docker Setup${NC}"
echo "========================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    echo "Installing Docker..."

    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER

    echo -e "${YELLOW}⚠️  Please log out and log back in for Docker permissions to take effect${NC}"
    echo "Then re-run this script"
    exit 1
fi

# Check for GPU support
GPU_SUPPORT=""
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"

    # Check for NVIDIA Container Toolkit
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &>/dev/null; then
        echo "Installing NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        sudo systemctl restart docker
    fi
    GPU_SUPPORT="--gpus all"
elif lspci | grep -i "amd\|ati" &>/dev/null; then
    echo -e "${GREEN}✅ AMD GPU detected${NC}"
fi

# Build Docker image
echo "Building Docker image..."
docker build -t opennlp-gpu -f docker/Dockerfile .

# Run Docker container
echo "Running OpenNLP GPU Extension in Docker..."
docker run --rm -it $GPU_SUPPORT \
    -v $(pwd):/workspace \
    opennlp-gpu \
    /bin/bash -c "cd /workspace && ./scripts/setup.sh"

echo -e "${GREEN}✅ Docker setup complete${NC}"
echo "You can now run:"
echo "  docker run --rm -it $GPU_SUPPORT -v \$(pwd):/workspace opennlp-gpu"
