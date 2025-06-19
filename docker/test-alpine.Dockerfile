# Multi-Platform Docker Testing for OpenNLP GPU
# Alpine Linux test environment

FROM alpine:3.18

# Install basic system dependencies
RUN apk update && apk add --no-cache \
    curl \
    wget \
    git \
    unzip \
    build-base \
    bash \
    bc \
    coreutils \
    pciutils

# Install Java 17
RUN apk add --no-cache openjdk17
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Maven
RUN apk add --no-cache maven

# Create working directory
WORKDIR /app

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Set up environment variables for testing
ENV CI=true
ENV DOCKER_ENV=alpine3.18

# Default command runs the compatibility tests
CMD ["./scripts/test_cross_platform_compatibility.sh"]
