# Troubleshooting Guide

## Build Issues
- Ensure Java 11+ and Maven are installed
- Clean build: `mvn clean install`
- For native errors: check CMake, CUDA/ROCm, and driver versions

## Runtime Issues
- "No GPU detected": Check driver installation and hardware
- "UnsatisfiedLinkError": Native library not found; rebuild with `mvn install`

## Test Failures
- Out of memory: Reduce test matrix size in stress tests
- Timeout: Run tests on a machine with more resources or reduce test workload

## Getting Help
- See FAQ.md
- Open an issue with logs and environment details 