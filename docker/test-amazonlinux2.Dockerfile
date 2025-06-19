# Multi-Platform Docker Testing for OpenNLP GPU
# Amazon Linux 2 test environment

FROM amazonlinux:2

# Install basic system dependencies
RUN yum update -y && yum install -y \
    curl \
    wget \
    git \
    unzip \
    gcc \
    gcc-c++ \
    make \
    bc \
    which \
    tar \
    && yum clean all

# Install Java 17
RUN yum install -y java-17-amazon-corretto-devel
ENV JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Maven manually (not available in default repos)
RUN cd /opt && \
    wget https://archive.apache.org/dist/maven/maven-3/3.9.4/binaries/apache-maven-3.9.4-bin.tar.gz && \
    tar xzf apache-maven-3.9.4-bin.tar.gz && \
    ln -s /opt/apache-maven-3.9.4 /opt/maven && \
    rm apache-maven-3.9.4-bin.tar.gz
ENV MAVEN_HOME=/opt/maven
ENV PATH=$MAVEN_HOME/bin:$PATH

# Install GPU detection tools
RUN yum install -y pciutils && yum clean all

# Create working directory
WORKDIR /app

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Set up environment variables for testing
ENV CI=true
ENV DOCKER_ENV=amazonlinux2

# Default command runs the compatibility tests
CMD ["./scripts/test_cross_platform_compatibility.sh"]
