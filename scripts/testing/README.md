# Testing Scripts

This directory contains comprehensive testing and demo execution scripts for the OpenNLP GPU acceleration project.

## Purpose

These scripts provide automated testing, demo execution, and validation workflows to ensure GPU acceleration functionality works correctly across different configurations and environments.

## Available Scripts

### 🚀 `run_maven_demos.sh` - Recommended Demo Runner
**Purpose**: Reliable Maven-based demo execution using `mvn exec:java`

**Why this is recommended**:
- ✅ Uses Maven exec plugin for consistent classpath handling
- ✅ Works reliably across all environments and IDEs
- ✅ Proper dependency resolution and resource loading
- ✅ Colored output with comprehensive error reporting
- ✅ Individual demo success/failure tracking

**What it runs**:
```bash
# Demo 1: Simple GPU Demo (basic functionality)
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.SimpleGpuDemo"

# Demo 2: Comprehensive Test Suite (all configurations)  
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Demo 3: GPU Demo Application (full feature demo)
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.GpuDemoApplication"
```

**Usage**:
```bash
./scripts/testing/run_maven_demos.sh

# Expected output:
🚀 OpenNLP GPU - Maven Demo Runner
=================================
🧪 Running: Simple GPU Demo
✅ Simple GPU Demo - SUCCESS
🧪 Running: Comprehensive Demo Test Suite  
✅ Comprehensive Demo Test Suite - SUCCESS
🧪 Running: GPU Demo Application
✅ GPU Demo Application - SUCCESS
🎉 All Maven demos executed successfully!
```

### 🎨 `run_all_demos.sh` - Legacy Demo Runner
**Purpose**: Legacy demo execution with comprehensive testing configurations

**What it tests**:
- ✅ **Basic Demo**: Standard functionality (`mvn test -Dtest=GpuDemoApplication`)
- ✅ **OpenCL Configuration**: GPU backend testing (`-Dgpu.backend=opencl`)
- ✅ **Debug Mode**: Verbose logging (`-Dgpu.debug=true`)
- ✅ **Comprehensive Mode**: All features (`-Dcomprehensive=true`)
- ✅ **Performance Mode**: Benchmark focus (`-Dperformance.only=true`)
- ✅ **Combined Configuration**: Multiple settings simultaneously

**Usage**:
```bash
./scripts/testing/run_all_demos.sh

# Runs 6 different test configurations with timing and success tracking
```

**Execution Flow**:
