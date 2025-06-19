# Multi-Platform Docker Testing for OpenNLP GPU
# CentOS 8 Stream test environment

FROM quay.io/centos/centos:stream8

# Install basic system dependencies
RUN dnf update -y && dnf install -y \
    curl \
    wget \
    git \
    unzip \
    gcc \
    gcc-c++ \
    make \
    bc \
    which \
    && dnf clean all

# Install Java 17
RUN dnf install -y java-17-openjdk-devel
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Maven
RUN dnf install -y maven

# Install GPU detection tools
RUN dnf install -y \
    pciutils \
    mesa-libGL \
    && dnf clean all

# Create working directory
WORKDIR /app

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Set up environment variables for testing
ENV CI=true
ENV DOCKER_ENV=centos8

# Default command runs the compatibility tests
CMD ["./scripts/test_cross_platform_compatibility.sh"]
