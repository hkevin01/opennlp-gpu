# GPU Prerequisites Guide for OpenNLP GPU

## Overview

This guide provides detailed instructions for setting up GPU acceleration prerequisites for OpenNLP GPU. The project supports multiple GPU platforms: NVIDIA (CUDA), AMD (ROCm), and Intel (OpenCL).

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **GPU**: Any modern GPU with compute capability
- **Memory**: 4GB system RAM, 2GB GPU memory
- **Storage**: 5GB free space for drivers and SDKs
- **CPU**: Multi-core processor (4+ cores recommended)

#### Recommended Requirements
- **GPU**: NVIDIA RTX 3060+ / AMD RX 6600+ / Intel Arc A750+
- **Memory**: 16GB system RAM, 8GB+ GPU memory
- **Storage**: 10GB+ free space
- **CPU**: 8+ cores for optimal performance

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+), Windows 10+, macOS 12+
- **Java**: OpenJDK 11+ or Oracle JDK 11+ (Java 17+ recommended)
- **Build Tools**: Maven 3.6+ or Gradle 7.0+

## NVIDIA CUDA Setup

### 1. Check GPU Compatibility

```bash
# Check if NVIDIA GPU is detected
lspci | grep -i nvidia

# Check current driver version
nvidia-smi
```

### 2. Install NVIDIA Drivers

#### Ubuntu/Debian
```bash
# Add NVIDIA repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install latest stable driver
sudo apt install nvidia-driver-535

# Alternative: Install specific version
sudo apt install nvidia-driver-470  # For older GPUs
```

#### CentOS/RHEL
```bash
# Enable EPEL repository
sudo yum install epel-release

# Install NVIDIA driver
sudo yum install nvidia-driver
```

#### Windows
1. Download driver from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Run installer as administrator
3. Restart system

### 3. Install CUDA Toolkit

#### Ubuntu/Debian
```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

#### Manual Installation
```bash
# Download CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run

# Install
sudo sh cuda_12.2.0_535.54.03_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Run CUDA samples
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
./deviceQuery

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

## AMD ROCm Setup

### 1. Check GPU Compatibility

```bash
# Check if AMD GPU is detected
lspci | grep -i amd

# Check current driver
glxinfo | grep "OpenGL vendor"
```

### 2. Install ROCm

#### Ubuntu 20.04+
```bash
# Add ROCm repository
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm
sudo apt update
sudo apt install rocm-dev rocm-libs

# Add user to video group
sudo usermod -a -G video $LOGNAME
```

#### CentOS/RHEL
```bash
# Add ROCm repository
sudo yum-config-manager --add-repo https://repo.radeon.com/rocm/yum/rpm
sudo yum install rocm-dev rocm-libs
```

### 3. Verify ROCm Installation

```bash
# Check ROCm installation
rocm-smi --showproductname

# Check GPU memory
rocm-smi --showmeminfo

# Check ROCm version
cat /opt/rocm/.info/version
```

## Intel OpenCL Setup

### 1. Check GPU Compatibility

```bash
# Check if Intel GPU is detected
lspci | grep -i intel

# Check current driver
glxinfo | grep "OpenGL vendor"
```

### 2. Install Intel OpenCL Runtime

#### Ubuntu/Debian
```bash
# Install Intel OpenCL runtime
sudo apt install intel-opencl-icd

# Install Intel Media Driver (for newer GPUs)
sudo apt install intel-media-va-driver-non-free
```

#### CentOS/RHEL
```bash
# Install Intel OpenCL runtime
sudo yum install intel-opencl
```

### 3. Verify OpenCL Installation

```bash
# Install clinfo tool
sudo apt install clinfo

# Check OpenCL platforms and devices
clinfo | grep -E "(Platform|Device)"

# Check Intel OpenCL specifically
clinfo | grep -A 10 "Intel"
```

## Cross-Platform Setup

### 1. Install OpenCL Headers

```bash
# Ubuntu/Debian
sudo apt install opencl-headers ocl-icd-opencl-dev

# CentOS/RHEL
sudo yum install opencl-headers
```

### 2. Install JOCL (Java OpenCL)

The project uses JOCL for OpenCL integration. It's included as a Maven dependency:

```xml
<dependency>
    <groupId>org.jocl</groupId>
    <artifactId>jocl</artifactId>
    <version>2.0.5</version>
</dependency>
```

## Environment Configuration

### 1. Set Environment Variables

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ROCm
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# OpenCL
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors
```

### 2. Java Environment

```bash
# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Set JVM options for GPU
export JAVA_OPTS="-Xmx4g -XX:+UseG1GC"
```

## Verification and Testing

### 1. Run GPU Diagnostics

```bash
# Build the project
mvn clean compile

# Run diagnostics
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"
```

### 2. Run Performance Tests

```bash
# Run GPU performance benchmarks
mvn test -Dtest=GpuPerformanceBenchmark

# Run stress tests
mvn test -Dtest=GpuStressTest
```

### 3. Test Examples

```bash
# Run quick start examples
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.GpuQuickStartDemo"

# Run specific examples
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.ner.GpuNamedEntityRecognition"
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check GPU hardware
lspci | grep -i vga

# Check driver status
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD
clinfo      # For OpenCL
```

#### 2. CUDA Installation Issues
```bash
# Check CUDA installation
ls -la /usr/local/cuda

# Check CUDA libraries
ldconfig -p | grep cuda

# Reinstall CUDA if needed
sudo apt remove --purge nvidia-cuda-toolkit
sudo apt install nvidia-cuda-toolkit
```

#### 3. ROCm Installation Issues
```bash
# Check ROCm installation
ls -la /opt/rocm

# Check user groups
groups $USER

# Reinstall ROCm if needed
sudo apt remove --purge rocm-dev rocm-libs
sudo apt install rocm-dev rocm-libs
```

#### 4. OpenCL Issues
```bash
# Check OpenCL installation
clinfo

# Check Intel OpenCL specifically
ls -la /usr/lib/x86_64-linux-gnu/intel-opencl

# Reinstall if needed
sudo apt remove --purge intel-opencl-icd
sudo apt install intel-opencl-icd
```

### Performance Issues

#### 1. Low GPU Utilization
- Check thermal throttling: `nvidia-smi -l 1` or `rocm-smi -l 1`
- Ensure adequate cooling
- Check power management settings

#### 2. Memory Issues
- Monitor GPU memory usage
- Reduce batch size in configuration
- Use CPU fallback for large models

#### 3. Driver Issues
- Update to latest stable drivers
- Check for known issues in driver release notes
- Consider rolling back to previous stable version

## Platform-Specific Notes

### Windows
- Install Visual Studio Build Tools for CUDA compilation
- Use Windows Subsystem for Linux (WSL) for development
- Ensure proper PATH configuration

### macOS
- Limited GPU support (Metal only)
- Use Docker for development
- Consider cloud-based GPU instances

### Cloud Platforms
- **AWS**: Use g4dn, p3, or p4 instances
- **Google Cloud**: Use T4, V100, or A100 instances
- **Azure**: Use NC, ND, or NV instances

## Next Steps

After completing the GPU setup:

1. **Build the Project**: `mvn clean install`
2. **Run Tests**: `mvn test`
3. **Explore Examples**: Check the `examples/` directory
4. **Read Documentation**: Review API documentation
5. **Join Community**: Participate in discussions and contribute

## Support Resources

- **NVIDIA CUDA**: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- **AMD ROCm**: [https://rocmdocs.amd.com/](https://rocmdocs.amd.com/)
- **Intel OpenCL**: [https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/overview.html](https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/overview.html)
- **OpenNLP Community**: [https://opennlp.apache.org/community.html](https://opennlp.apache.org/community.html)
