# OpenNLP GPU Extension - Quick Setup Guide

## 🚀 One-Click Setup

This project includes multiple setup options for different environments:

### 🖥️ Local Machine Setup
```bash
git clone <repository-url>
cd opennlp-gpu
./setup.sh
```

### ☁️ AWS EC2 Setup
```bash
git clone <repository-url>
cd opennlp-gpu
./aws_setup.sh
```

### 🐳 Docker Setup
```bash
git clone <repository-url>
cd opennlp-gpu
./docker_setup.sh
```

## 📋 Prerequisites

The setup scripts will automatically install:
- **Java 21** (OpenJDK)
- **Maven 3.9+**
- **CMake 3.16+**
- **Build tools** (gcc, make, etc.)
- **GPU drivers** (NVIDIA CUDA or AMD ROCm, as needed)

## 🔧 Supported Platforms

### Operating Systems
- ✅ **Ubuntu 20.04/22.04** (Recommended)
- ✅ **Debian 11+**
- ✅ **CentOS 8/9**
- ✅ **RHEL 8/9**
- ✅ **Amazon Linux 2**
- ✅ **macOS** (via Homebrew)
- ✅ **Windows WSL2**

### GPU Support
- ✅ **NVIDIA CUDA** (automatic detection and setup)
- ✅ **AMD ROCm/HIP** (automatic detection and setup)
- ✅ **CPU-only fallback** (when no GPU available)

### Cloud Platforms
- ✅ **AWS EC2** (including GPU instances: p2, p3, p4, g3, g4, g5)
- ✅ **Google Cloud** (including GPU instances)
- ✅ **Microsoft Azure** (including GPU instances)
- ✅ **Local development**

## 🎯 Quick Start

1. **Choose your setup method:**
   ```bash
   # Standard setup (works everywhere)
   ./setup.sh
   
   # AWS-optimized setup
   ./aws_setup.sh
   
   # Docker-based setup
   ./docker_setup.sh
   ```

2. **Run the demo:**
   ```bash
   ./gpu_demo.sh
   ```

3. **Check GPU diagnostics:**
   ```bash
   java -cp "target/classes:$(cat classpath.txt)" org.apache.opennlp.gpu.tools.GpuDiagnostics
   ```

## 🛠️ Manual Setup (if scripts fail)

### 1. Install Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install openjdk-21-jdk maven cmake build-essential git

# CentOS/RHEL
sudo dnf install java-21-openjdk-devel maven cmake gcc-c++ git

# macOS
brew install openjdk@21 maven cmake git
```

### 2. Set Environment Variables
```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export ROCM_PATH=/opt/rocm  # For AMD GPU
```

### 3. Build Native Library
```bash
cd src/main/cpp
cmake .
make -j4
```

### 4. Build Java Project
```bash
mvn clean compile
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Java Not Found
```bash
# Install Java 21
sudo apt-get install openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

#### 2. Maven Build Fails
```bash
# Update Maven dependencies
mvn dependency:resolve
mvn clean compile -U
```

#### 3. CMake Not Found
```bash
# Install newer CMake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
echo "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list
sudo apt-get update
sudo apt-get install cmake
```

#### 4. GPU Not Detected
```bash
# For NVIDIA GPU
nvidia-smi  # Check if drivers are installed
sudo apt-get install nvidia-driver-535

# For AMD GPU
rocm-smi    # Check if ROCm is installed
sudo apt-get install rocm-dev hip-dev
```

#### 5. Permission Issues
```bash
# Make scripts executable
chmod +x setup.sh aws_setup.sh docker_setup.sh

# Add user to docker group (for Docker setup)
sudo usermod -aG docker $USER
# Log out and log back in
```

### Log Files
- **Setup log**: `setup.log`
- **Error log**: `setup-errors.log`
- **Maven log**: Check console output

## 🧪 Testing

### Basic Functionality Test
```bash
# Run all demos
./gpu_demo.sh

# Individual tests
java -cp "target/classes:$(cat classpath.txt)" org.apache.opennlp.gpu.tools.GpuDiagnostics
java -cp "target/classes:$(cat classpath.txt)" org.apache.opennlp.gpu.ml.GpuMlDemo
```

### Performance Benchmarks
```bash
java -cp "target/classes:$(cat classpath.txt)" org.apache.opennlp.gpu.benchmark.PerformanceBenchmark
```

## 📦 AWS EC2 Instance Recommendations

### GPU Instances (for GPU acceleration)
- **p3.2xlarge** - NVIDIA V100, good for development
- **g4dn.xlarge** - NVIDIA T4, cost-effective
- **g5.xlarge** - NVIDIA A10G, latest generation

### CPU Instances (for CPU-only mode)
- **m5.large** - General purpose, 2 vCPUs
- **c5.xlarge** - Compute optimized, 4 vCPUs
- **m5.xlarge** - General purpose, 4 vCPUs

### Setup Commands for AWS
```bash
# Launch instance and connect
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt-get update

# Clone and setup
git clone <repository-url>
cd opennlp-gpu
./aws_setup.sh
```

## 🐳 Docker Quick Reference

### Build Image
```bash
docker build -t opennlp-gpu .
```

### Run with GPU Support
```bash
# NVIDIA GPU
docker run --gpus all -it opennlp-gpu

# AMD GPU
docker run --device=/dev/kfd --device=/dev/dri -it opennlp-gpu

# CPU only
docker run -it opennlp-gpu
```

## 🎯 Environment-Specific Instructions

### Ubuntu 22.04 (Recommended)
```bash
./setup.sh  # Everything should work out of the box
```

### Amazon Linux 2
```bash
sudo yum update
sudo yum install java-21-amazon-corretto-devel
./setup.sh
```

### Windows WSL2
```bash
# In WSL2 terminal
sudo apt-get update
./setup.sh
# GPU support requires Windows 11 and WSL2 GPU drivers
```

### macOS
```bash
# Install Homebrew first: https://brew.sh
./setup.sh
# Note: GPU acceleration not available on macOS
```

## 💡 Tips

1. **First time setup**: Use `./setup.sh` - it handles all edge cases
2. **AWS users**: Use `./aws_setup.sh` for optimized AWS setup
3. **Isolated environment**: Use `./docker_setup.sh` for containerized setup
4. **Check logs**: Review `setup.log` and `setup-errors.log` if issues occur
5. **Rerun setup**: Safe to run setup scripts multiple times
6. **Update environment**: Source `~/.bashrc` after setup for environment variables

## 🆘 Support

If you encounter issues:

1. **Check logs**: `cat setup.log` and `cat setup-errors.log`
2. **Verify Java**: `java -version` (should be 21+)
3. **Verify Maven**: `mvn --version` (should be 3.6+)
4. **Verify CMake**: `cmake --version` (should be 3.16+)
5. **Check GPU**: `nvidia-smi` or `rocm-smi`
6. **Re-run setup**: `./setup.sh` (safe to run multiple times)

The setup scripts are designed to be robust and handle most common issues automatically. They will attempt alternative installation methods if the primary method fails.
