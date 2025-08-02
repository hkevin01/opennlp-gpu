#!/bin/bash

# Google TPU Setup Script for OpenNLP GPU Extension
# Copyright 2025 OpenNLP GPU Extension Contributors
# Licensed under the Apache License, Version 2.0

set -e  # Exit on any error

echo "🚀 Setting up Google TPU environment for OpenNLP GPU Extension..."

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

# Check if running on GCP
check_gcp_environment() {
    print_info "Checking GCP environment..."

    # Check for GCP metadata service
    if curl -s --max-time 2 -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type > /dev/null 2>&1; then
        MACHINE_TYPE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type)
        print_success "Running on GCP instance: $(basename $MACHINE_TYPE)"

        # Check if it's a TPU instance
        if echo "$MACHINE_TYPE" | grep -i "tpu" > /dev/null; then
            print_success "TPU instance detected: $(basename $MACHINE_TYPE)"
            return 0
        else
            print_warning "Not a TPU instance, but GCP environment detected"
            return 1
        fi
    else
        print_warning "Not running on GCP or metadata service unavailable"
        return 1
    fi
}

# Install JAX with TPU support
install_jax_tpu() {
    print_info "Installing JAX with TPU support..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install JAX with TPU support
    pip install --upgrade pip
    pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install jaxlib
    pip install flax
    pip install optax

    print_success "JAX with TPU support installed successfully"
}

# Install TensorFlow with TPU support
install_tensorflow_tpu() {
    print_info "Installing TensorFlow with TPU support..."

    # Activate virtual environment
    source venv/bin/activate

    # Install TensorFlow
    pip install tensorflow
    pip install tensorflow-text
    pip install cloud-tpu-client

    print_success "TensorFlow with TPU support installed successfully"
}

# Install additional ML libraries
install_ml_libraries() {
    print_info "Installing additional ML libraries..."

    # Activate virtual environment
    source venv/bin/activate

    # Install supporting libraries
    pip install transformers
    pip install datasets
    pip install tokenizers
    pip install scikit-learn
    pip install numpy
    pip install pandas

    print_success "Additional ML libraries installed successfully"
}

# Configure TPU environment
configure_tpu_environment() {
    print_info "Configuring TPU environment variables..."

    # Create environment configuration
    cat > tpu_env.sh << 'EOF'
#!/bin/bash
# Google TPU Environment Configuration

# TPU settings
export TPU_NAME=local
export TPU_LOAD_LIBRARY=0
export JAX_PLATFORMS=tpu

# XLA settings for optimization
export XLA_FLAGS="--xla_force_host_platform_device_count=8"

# JAX configuration
export JAX_ENABLE_X64=True

# TensorFlow TPU settings
export TF_CPP_MIN_LOG_LEVEL=1

echo "Google TPU environment configured"
EOF

    chmod +x tpu_env.sh

    # Add to bashrc for persistence
    echo "source $(pwd)/tpu_env.sh" >> ~/.bashrc

    print_success "TPU environment variables configured"
}

# Test TPU installation
test_tpu_installation() {
    print_info "Testing TPU installation..."

    if [ -d "venv" ]; then
        source venv/bin/activate

        # Test JAX TPU integration
        python3 -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')

try:
    tpu_devices = jax.devices('tpu')
    if tpu_devices:
        print(f'✅ TPU devices found: {len(tpu_devices)} devices')
        for i, device in enumerate(tpu_devices):
            print(f'  TPU {i}: {device}')
    else:
        print('⚠️  No TPU devices found (normal on non-TPU instances)')
except Exception as e:
    print(f'⚠️  TPU detection error: {e}')

# Test basic JAX operation
try:
    import jax.numpy as jnp
    x = jnp.array([1, 2, 3, 4])
    y = jnp.sum(x)
    print(f'✅ JAX basic operation successful: sum([1,2,3,4]) = {y}')
except Exception as e:
    print(f'❌ JAX operation failed: {e}')
"

        # Test TensorFlow TPU integration
        python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')

try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print(f'✅ TPU strategy created with {strategy.num_replicas_in_sync} replicas')
except Exception as e:
    print(f'⚠️  TPU strategy creation failed: {e} (normal on non-TPU instances)')
"
    fi

    print_success "TPU installation test completed"
}

# Create enhanced OpenNLP configuration
create_opennlp_config() {
    print_info "Creating OpenNLP TPU configuration..."

    cat > google_tpu_config.properties << 'EOF'
# OpenNLP GPU Extension - Google TPU Configuration

# Enable cloud accelerator support
gpu.cloud.enabled=true

# Google TPU settings
gpu.cloud.provider=google_tpu
gpu.cloud.tpu.enabled=true
gpu.cloud.tpu.cores=8
gpu.cloud.tpu.memory_gb=32

# Training optimization
gpu.training.batch_size=128
gpu.training.optimization_level=3
gpu.training.precision=bfloat16
gpu.training.gradient_accumulation=4

# Inference optimization
gpu.inference.batch_size=64
gpu.inference.optimization_level=3
gpu.inference.precision=fp16

# Fallback configuration
gpu.fallback.enabled=true
gpu.fallback.provider=cpu

# Logging
gpu.logging.level=INFO
gpu.logging.metrics=true
gpu.logging.profiling=true
EOF

    print_success "OpenNLP TPU configuration created"
}

# Setup TPU utilities
create_tpu_utilities() {
    print_info "Creating TPU utility scripts..."

    # Create TPU status script
    cat > check_tpu_status.py << 'EOF'
#!/usr/bin/env python3
"""
TPU Status Checker for OpenNLP GPU Extension
"""

import jax
import tensorflow as tf

def check_jax_tpu():
    print("=== JAX TPU Status ===")
    print(f"JAX version: {jax.__version__}")

    try:
        devices = jax.devices()
        print(f"Available devices: {len(devices)}")

        tpu_devices = jax.devices('tpu')
        if tpu_devices:
            print(f"TPU devices: {len(tpu_devices)}")
            for i, device in enumerate(tpu_devices):
                print(f"  TPU {i}: {device}")
        else:
            print("No TPU devices found")

    except Exception as e:
        print(f"Error checking JAX TPU: {e}")

def check_tensorflow_tpu():
    print("\n=== TensorFlow TPU Status ===")
    print(f"TensorFlow version: {tf.__version__}")

    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f"TPU cluster: {resolver.cluster_spec()}")

        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"TPU strategy: {strategy.num_replicas_in_sync} replicas")

    except Exception as e:
        print(f"Error checking TensorFlow TPU: {e}")

if __name__ == "__main__":
    check_jax_tpu()
    check_tensorflow_tpu()
EOF

    chmod +x check_tpu_status.py

    print_success "TPU utility scripts created"
}

# Main setup function
main() {
    print_info "Starting Google TPU setup for OpenNLP GPU Extension"
    print_info "=================================================="

    # Check prerequisites
    if ! command -v curl > /dev/null 2>&1; then
        print_error "curl is required but not installed"
        exit 1
    fi

    if ! command -v python3 > /dev/null 2>&1; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi

    # Check GCP environment
    check_gcp_environment
    GCP_ENV=$?

    # Install components
    install_jax_tpu
    install_tensorflow_tpu
    install_ml_libraries
    configure_tpu_environment

    # Test installation
    test_tpu_installation

    # Create configuration and utilities
    create_opennlp_config
    create_tpu_utilities

    print_success "=================================================="
    print_success "Google TPU setup completed successfully!"
    print_info ""
    print_info "Next steps:"
    print_info "1. Source the environment: source tpu_env.sh"
    print_info "2. Activate Python virtual environment: source venv/bin/activate"
    print_info "3. Check TPU status: python3 check_tpu_status.py"
    print_info "4. Compile your OpenNLP project with: mvn clean compile"
    print_info "5. Run with TPU acceleration enabled"
    print_info ""

    if [ $GCP_ENV -eq 0 ]; then
        print_success "TPU instance detected - hardware acceleration ready!"
    else
        print_warning "Not on TPU instance - software simulation mode available"
    fi
}

# Run main function
main "$@"
