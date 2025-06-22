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
    GPU_SUPPORT="--device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined"
fi

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-21-jdk \
    maven \
    cmake \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# Install ROCm for AMD GPU support (if needed)
RUN if [ -n "${AMD_GPU}" ]; then \
        wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
        echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ ubuntu main' > /etc/apt/sources.list.d/rocm.list && \
        apt-get update && \
        apt-get install -y rocm-dev hip-dev && \
        echo 'export ROCM_PATH=/opt/rocm' >> /etc/bash.bashrc && \
        echo 'export HIP_PATH=/opt/rocm' >> /etc/bash.bashrc && \
        echo 'export PATH=/opt/rocm/bin:$PATH' >> /etc/bash.bashrc; \
    fi

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Build the project
RUN ./setup.sh || echo "Setup completed with warnings"

# Default command
CMD ["./gpu_demo.sh"]
EOF

# Build Docker image
echo "🔨 Building Docker image..."
if lspci | grep -i "amd\|ati" &>/dev/null; then
    docker build --build-arg AMD_GPU=1 -t opennlp-gpu .
else
    docker build -t opennlp-gpu .
fi

# Create run script
cat > run_docker.sh << EOF
#!/bin/bash
echo "🚀 Running OpenNLP GPU Extension in Docker..."
docker run -it --rm $GPU_SUPPORT -v \$(pwd):/workspace opennlp-gpu bash
EOF

chmod +x run_docker.sh

echo -e "${GREEN}✅ Docker setup complete!${NC}"
echo ""
echo "Usage:"
echo "  ./run_docker.sh          # Run interactive container"
echo "  docker run --rm $GPU_SUPPORT opennlp-gpu  # Run demo directly"
