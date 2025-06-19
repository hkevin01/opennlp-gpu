# Windows Nano Server with Java for lightweight testing
FROM mcr.microsoft.com/windows/nanoserver:ltsc2022

# Note: Windows Nano Server has limited capabilities
# This is mainly for basic compatibility testing

# Set working directory
WORKDIR C:\\opennlp-gpu

# Copy project files
COPY . .

# Copy a pre-built Java runtime (would need to be included in build context)
# COPY jre C:\\java

# Basic test command - just check if files are copied correctly
CMD ["cmd", "/c", "dir && echo Windows Nano Server test completed"]
