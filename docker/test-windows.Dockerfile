# Windows Server Core with Java for OpenNLP GPU testing
# escape=`

FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set shell to PowerShell
SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Install Chocolatey package manager
RUN Set-ExecutionPolicy Bypass -Scope Process -Force; \
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; \
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Java 17, Maven, and Git
RUN choco install -y openjdk17 maven git

# Refresh environment variables
RUN refreshenv

# Set JAVA_HOME environment variable
RUN setx JAVA_HOME "C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.9.9-hotspot" /M

# Add Java and Maven to PATH
RUN setx PATH "%PATH%;%JAVA_HOME%\\bin;C:\\ProgramData\\chocolatey\\lib\\maven\\apache-maven-3.9.5\\bin" /M

# Create working directory
WORKDIR C:\\opennlp-gpu

# Copy project files
COPY . .

# Install any Windows-specific dependencies
RUN if (Test-Path "scripts\\setup_windows_environment.ps1") { \
    PowerShell -ExecutionPolicy Bypass -File "scripts\\setup_windows_environment.ps1" \
    }

# Set environment variables for testing
ENV OPENNLP_GPU_TEST_MODE=1
ENV JAVA_OPTS="-Xmx2g -XX:+UseG1GC"

# Health check to verify Java and Maven installation
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD powershell -Command "java -version; if ($$LASTEXITCODE -ne 0) { exit 1 }; mvn -version; if ($$LASTEXITCODE -ne 0) { exit 1 }"

# Default command - run compatibility tests
CMD powershell -Command \
    "Write-Host 'Starting OpenNLP GPU Windows compatibility test...'; \
    java -version; \
    mvn -version; \
    if (Test-Path 'scripts\\test_cross_platform_compatibility.sh') { \
    bash scripts\\test_cross_platform_compatibility.sh \
    } else { \
    Write-Host 'Running basic Maven test...'; \
    mvn clean compile test -Dtest=GpuDemoApplication \
    }"
