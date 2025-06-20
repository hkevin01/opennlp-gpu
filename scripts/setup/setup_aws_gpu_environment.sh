#!/bin/bash

# OpenNLP GPU - AWS GPU Environment Setup Script
# Configures AWS EC2 instances for optimal GPU acceleration

set -e

echo "☁️ OpenNLP GPU - AWS GPU Environment Setup"
echo "==========================================="

# Function to detect AWS instance type
detect_aws_instance() {
    if command -v curl &> /dev/null && curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null; then
        local instance_type=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
        echo "🖥️ Detected AWS Instance: $instance_type"
        echo "$instance_type"
    else
        echo "⚠️ Not running on AWS or metadata service unavailable"
        return 1
    fi
}

# Function to detect GPU hardware
detect_gpu_hardware() {
    echo "🔍 Detecting GPU hardware..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        return 0
    elif command -v rocm-smi &> /dev/null; then
        echo "✅ AMD GPU detected:"
        rocm-smi --showproductname
        return 0
    else
        echo "❌ No GPU detected or drivers not installed"
        return 1
    fi
}

# Function to install NVIDIA drivers and CUDA
install_nvidia_cuda() {
    echo "📦 Installing NVIDIA drivers and CUDA..."
    
    # Detect Ubuntu version
    local ubuntu_version=$(lsb_release -rs)
    echo "🐧 Ubuntu version: $ubuntu_version"
    
    # Install NVIDIA drivers
    sudo apt-get update
    sudo apt-get install -y nvidia-driver-535
    
    # Install CUDA toolkit
    if [[ "$ubuntu_version" == "22.04" ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
    elif [[ "$ubuntu_version" == "20.04" ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
    fi
    
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-2
    
    # Set environment variables
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    echo "✅ NVIDIA CUDA installation completed"
    echo "⚠️ System reboot recommended to load new drivers"
}

# Function to install Java optimized for AWS
install_java_aws() {
    echo "☕ Installing Java optimized for AWS..."
    
    # Install Amazon Corretto (AWS optimized OpenJDK)
    curl -fsSL https://apt.corretto.aws/corretto.key | sudo apt-key add -
    echo "deb https://apt.corretto.aws stable main" | sudo tee /etc/apt/sources.list.d/corretto.list
    
    sudo apt-get update
    sudo apt-get install -y java-17-amazon-corretto-jdk
    
    # Set JAVA_HOME
    echo 'export JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto' >> ~/.bashrc
    
    echo "✅ Amazon Corretto Java 17 installed"
}

# Function to install Maven
install_maven() {
    echo "📦 Installing Maven..."
    sudo apt-get install -y maven
    echo "✅ Maven installed"
}

# Function to optimize AWS instance for GPU workloads
optimize_aws_instance() {
    echo "⚡ Optimizing AWS instance for GPU workloads..."
    
    # Increase file limits for large datasets
    echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
    echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
    
    # Optimize network settings for S3 access
    echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
    echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
    echo 'net.ipv4.tcp_rmem = 4096 87380 16777216' | sudo tee -a /etc/sysctl.conf
    echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' | sudo tee -a /etc/sysctl.conf
    
    # Apply sysctl changes
    sudo sysctl -p
    
    echo "✅ AWS instance optimization completed"
}

# Function to setup S3 integration
setup_s3_integration() {
    echo "📁 Setting up S3 integration..."
    
    # Install AWS CLI if not present
    if ! command -v aws &> /dev/null; then
        echo "📦 Installing AWS CLI..."
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
        echo "✅ AWS CLI installed"
    else
        echo "✅ AWS CLI already installed"
    fi
    
    # Create sample S3 integration script
    cat > ~/s3_nlp_processor.sh << 'EOF'
#!/bin/bash
# Sample S3 NLP processing script

BUCKET_NAME="$1"
INPUT_PREFIX="$2"
OUTPUT_PREFIX="$3"

if [[ -z "$BUCKET_NAME" || -z "$INPUT_PREFIX" || -z "$OUTPUT_PREFIX" ]]; then
    echo "Usage: $0 <bucket-name> <input-prefix> <output-prefix>"
    exit 1
fi

echo "Processing documents from s3://$BUCKET_NAME/$INPUT_PREFIX"

# Download documents from S3
aws s3 sync s3://$BUCKET_NAME/$INPUT_PREFIX/ ./input/

# Process with OpenNLP GPU
java -jar /path/to/opennlp-gpu.jar --batch-process ./input ./output

# Upload results to S3
aws s3 sync ./output/ s3://$BUCKET_NAME/$OUTPUT_PREFIX/

echo "Results uploaded to s3://$BUCKET_NAME/$OUTPUT_PREFIX"
EOF
    
    chmod +x ~/s3_nlp_processor.sh
    echo "✅ S3 integration script created at ~/s3_nlp_processor.sh"
}

# Function to setup AWS Batch integration
setup_aws_batch() {
    echo "🎯 Setting up AWS Batch integration..."
    
    # Create Dockerfile for AWS Batch
    cat > ~/Dockerfile.aws-batch << 'EOF'
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Install Java and dependencies
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    maven \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Copy OpenNLP GPU project
COPY . /app
WORKDIR /app

# Build the project
RUN mvn clean package

# Set entrypoint
ENTRYPOINT ["java", "-jar", "target/opennlp-gpu-1.0-SNAPSHOT.jar"]
EOF
    
    echo "✅ AWS Batch Dockerfile created at ~/Dockerfile.aws-batch"
    
    # Create AWS Batch job definition template
    cat > ~/aws-batch-job-definition.json << 'EOF'
{
    "jobDefinitionName": "opennlp-gpu-processing",
    "type": "container",
    "containerProperties": {
        "image": "your-account.dkr.ecr.region.amazonaws.com/opennlp-gpu:latest",
        "vcpus": 4,
        "memory": 8192,
        "resourceRequirements": [
            {
                "type": "GPU",
                "value": "1"
            }
        ],
        "jobRoleArn": "arn:aws:iam::your-account:role/BatchJobRole",
        "environment": [
            {
                "name": "AWS_DEFAULT_REGION",
                "value": "us-east-1"
            }
        ]
    },
    "retryStrategy": {
        "attempts": 3
    },
    "timeout": {
        "attemptDurationSeconds": 3600
    }
}
EOF
    
    echo "✅ AWS Batch job definition template created at ~/aws-batch-job-definition.json"
}

# Function to run final diagnostics
run_aws_diagnostics() {
    echo "🔍 Running AWS-specific diagnostics..."
    
    # Check instance metadata
    if curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id &>/dev/null; then
        echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
        echo "Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
        echo "Availability Zone: $(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)"
    fi
    
    # Check GPU
    detect_gpu_hardware
    
    # Check Java
    java -version
    
    # Check Maven
    mvn -version
    
    # Check AWS CLI
    aws --version
    
    echo "✅ AWS diagnostics completed"
}

# Main execution function
main() {
    echo "🚀 Starting AWS GPU environment setup..."
    echo ""
    
    # Detect if running on AWS
    local instance_type=""
    if instance_type=$(detect_aws_instance); then
        echo "✅ Running on AWS instance: $instance_type"
    else
        echo "⚠️ Not detected as AWS instance, continuing anyway..."
    fi
    
    # Update system
    echo "📦 Updating system packages..."
    sudo apt-get update
    sudo apt-get upgrade -y
    
    # Install basic dependencies
    sudo apt-get install -y \
        build-essential \
        curl \
        wget \
        unzip \
        git \
        htop \
        bc \
        lsb-release
    
    # Install Java optimized for AWS
    install_java_aws
    
    # Install Maven
    install_maven
    
    # Check for GPU and install drivers
    if detect_gpu_hardware; then
        echo "✅ GPU already configured"
    else
        echo "⚠️ GPU not detected, installing NVIDIA drivers..."
        install_nvidia_cuda
    fi
    
    # Optimize AWS instance
    optimize_aws_instance
    
    # Setup integrations
    setup_s3_integration
    setup_aws_batch
    
    # Run diagnostics
    run_aws_diagnostics
    
    echo ""
    echo "🎉 AWS GPU environment setup completed!"
    echo ""
    echo "✅ Ready for OpenNLP GPU acceleration on AWS"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Reboot instance if NVIDIA drivers were installed"
    echo "  2. Clone and build OpenNLP GPU project"
    echo "  3. Run: ./scripts/run_all_demos.sh"
    echo "  4. Configure S3 access for your data processing needs"
    echo "  5. Use AWS Batch for large-scale processing jobs"
    echo ""
    echo "🔧 AWS-specific features configured:"
    echo "  • Amazon Corretto Java 17 (AWS optimized)"
    echo "  • NVIDIA CUDA drivers and toolkit"
    echo "  • S3 integration script: ~/s3_nlp_processor.sh"
    echo "  • AWS Batch Dockerfile: ~/Dockerfile.aws-batch"
    echo "  • Instance optimization for GPU workloads"
    echo ""
    echo "💰 Cost optimization tips:"
    echo "  • Use Spot instances for batch processing (up to 70% savings)"
    echo "  • Scale down or stop instances when not processing"
    echo "  • Use S3 lifecycle policies for data archival"
    echo "  • Monitor GPU utilization with CloudWatch"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --no-reboot    Skip reboot recommendation"
        echo ""
        echo "This script configures AWS EC2 instances for OpenNLP GPU acceleration."
        echo "Optimized for p3, p4d, and g4dn instance types."
        exit 0
        ;;
esac

# Execute main function
main "$@"
