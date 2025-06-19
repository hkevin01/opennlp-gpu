# Multi-Platform Docker Testing for OpenNLP GPU
# Ubuntu 22.04 LTS test environment

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Install Java 17
RUN apt-get update && apt-get install -y openjdk-17-jdk
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Maven
RUN apt-get update && apt-get install -y maven

# Install GPU detection tools (for testing GPU detection scripts)
RUN apt-get update && apt-get install -y \
    pciutils \
    clinfo \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Set up environment variables for testing
ENV CI=true
ENV DOCKER_ENV=ubuntu22.04

# Default command runs the compatibility tests
CMD ["./scripts/test_cross_platform_compatibility.sh"]
