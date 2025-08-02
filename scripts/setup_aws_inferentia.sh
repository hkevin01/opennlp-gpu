#!/bin/bash

# AWS Inferentia/Trainium Setup Script for OpenNLP GPU Extension
# Copyright 2025 OpenNLP GPU Extension Contributors
# Licensed under the Apache License, Version 2.0

set -e  # Exit on any error

echo "🚀 Setting up AWS Inferentia/Trainium environment for OpenNLP GPU Extension..."

# Color output functions
print_info() {
    echo -e "\e[34mℹ️  $1\e[0m"
}

print_success() {
    echo -e "\e[32m✅ $1\e[0m"
}

print_warning() {
    echo -e "\e[33m⚠️  $1\e[0m"
}

print_error() {
    echo -e "\e[31m❌ $1\e[0m"
}

# Check if running on AWS
check_aws_environment() {
    print_info "Checking AWS environment..."

    # Check for AWS metadata service
    if curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type > /dev/null 2>&1; then
        INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
        print_success "Running on AWS instance: $INSTANCE_TYPE"

        # Check if it's an Inferentia instance
        if [[ "$INSTANCE_TYPE" == inf1.* ]] || [[ "$INSTANCE_TYPE" == inf2.* ]]; then
            print_success "Inferentia instance detected: $INSTANCE_TYPE"
            return 0
        elif [[ "$INSTANCE_TYPE" == trn1.* ]]; then
            print_success "Trainium instance detected: $INSTANCE_TYPE"
            return 0
        else
            print_warning "Not an Inferentia/Trainium instance, but AWS environment detected"
            return 1
        fi
    else
        print_warning "Not running on AWS or metadata service unavailable"
        return 1
    fi
}

# Install AWS Neuron SDK
install_neuron_sdk() {
    print_info "Installing AWS Neuron SDK..."

    # Add Neuron repository
    echo "deb https://apt.repos.neuron.amazonaws.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/neuron.list

    # Import GPG key
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-NEURON.PUB | sudo apt-key add -

    # Update package list
    sudo apt-get update

    # Install Neuron SDK components
    sudo apt-get install -y \
        aws-neuronx-tools \
        aws-neuronx-runtime-lib \
        aws-neuronx-collectives

    print_success "Neuron SDK installed successfully"
}

# Install Python packages for Neuron
install_python_packages() {
    print_info "Installing Python packages for Neuron..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install Neuron packages
    pip install --upgrade pip
    pip install neuronx-cc
    pip install torch-neuronx
    pip install transformers

    print_success "Python packages installed successfully"
}

# Configure environment variables
configure_environment() {
    print_info "Configuring environment variables..."

    # Create environment configuration
    cat > neuron_env.sh << 'EOF'
#!/bin/bash
# AWS Neuron Environment Configuration

export NEURON_CC_FLAGS="--framework XLA"
export NEURON_RT_NUM_CORES=4
export NEURON_RT_LOG_LEVEL=INFO

# Add Neuron tools to PATH
export PATH=/opt/aws/neuron/bin:$PATH

# Neuron runtime library path
export LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH

echo "AWS Neuron environment configured"
EOF

    chmod +x neuron_env.sh

    # Add to bashrc for persistence
    echo "source $(pwd)/neuron_env.sh" >> ~/.bashrc

    print_success "Environment variables configured"
}

# Test Neuron installation
test_neuron_installation() {
    print_info "Testing Neuron installation..."

    # Check if neuron-ls is available
    if command -v neuron-ls > /dev/null 2>&1; then
        print_info "Neuron devices:"
        neuron-ls || print_warning "No Neuron devices found (normal on non-Inferentia instances)"
    else
        print_warning "neuron-ls command not found"
    fi

    # Test Python integration
    if [ -d "venv" ]; then
        source venv/bin/activate
        python3 -c "
try:
    import neuronx_cc
    print('✅ neuronx-cc imported successfully')
except ImportError as e:
    print(f'❌ Failed to import neuronx-cc: {e}')

try:
    import torch_neuronx
    print('✅ torch-neuronx imported successfully')
except ImportError as e:
    print(f'❌ Failed to import torch-neuronx: {e}')
"
    fi

    print_success "Neuron installation test completed"
}

# Create enhanced OpenNLP configuration
create_opennlp_config() {
    print_info "Creating OpenNLP Inferentia configuration..."

    cat > aws_inferentia_config.properties << 'EOF'
# OpenNLP GPU Extension - AWS Inferentia Configuration

# Enable cloud accelerator support
gpu.cloud.enabled=true

# AWS Inferentia settings
gpu.cloud.provider=aws_inferentia
gpu.cloud.inferentia.enabled=true
gpu.cloud.inferentia.cores=4
gpu.cloud.inferentia.memory_gb=16

# Inference optimization
gpu.inference.batch_size=32
gpu.inference.optimization_level=3
gpu.inference.precision=fp16

# Fallback configuration
gpu.fallback.enabled=true
gpu.fallback.provider=cpu

# Logging
gpu.logging.level=INFO
gpu.logging.metrics=true
EOF

    print_success "OpenNLP Inferentia configuration created"
}

# Main setup function
main() {
    print_info "Starting AWS Inferentia/Trainium setup for OpenNLP GPU Extension"
    print_info "================================================================="

    # Check prerequisites
    if ! command -v curl > /dev/null 2>&1; then
        print_error "curl is required but not installed"
        exit 1
    fi

    if ! command -v python3 > /dev/null 2>&1; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi

    # Check AWS environment
    check_aws_environment
    AWS_ENV=$?

    # Install components
    install_neuron_sdk
    install_python_packages
    configure_environment

    # Test installation
    test_neuron_installation

    # Create OpenNLP configuration
    create_opennlp_config

    print_success "================================================================="
    print_success "AWS Inferentia/Trainium setup completed successfully!"
    print_info ""
    print_info "Next steps:"
    print_info "1. Source the environment: source neuron_env.sh"
    print_info "2. Activate Python virtual environment: source venv/bin/activate"
    print_info "3. Compile your OpenNLP project with: mvn clean compile"
    print_info "4. Run with Inferentia acceleration enabled"
    print_info ""

    if [ $AWS_ENV -eq 0 ]; then
        print_success "Inferentia/Trainium instance detected - hardware acceleration ready!"
    else
        print_warning "Not on Inferentia/Trainium instance - software simulation mode available"
    fi
}

# Run main function
main "$@"
