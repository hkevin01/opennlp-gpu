# GPU Acceleration Demo Applications

This directory contains interactive demonstration applications that showcase OpenNLP GPU acceleration capabilities through complete examples and tutorials.

## 🎯 Demo Files Overview

### `GpuDemoApplication.java`
**Purpose**: Complete demonstration of GPU acceleration features with tests and benchmarks  
**What it demonstrates**: Full GPU acceleration pipeline, testing framework, performance comparison

**Run the complete demo:**
```bash
# Run comprehensive demo application
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.GpuDemoApplication"

# Run demo with custom GPU settings
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.GpuDemoApplication" \
  -Dgpu.enabled=true -Dgpu.memory.limit=2048

# Run with verbose output
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.GpuDemoApplication" \
  -Dlogging.level=DEBUG
```

**Run via test framework:**
```bash
# Run as a test
mvn test -Dtest=GpuDemoApplication

# Run with specific configuration
mvn test -Dtest=GpuDemoApplication -Dgpu.backend=opencl

# Run with debug output
mvn test -Dtest=GpuDemoApplication -Dgpu.debug=true

# Run with comprehensive testing
mvn test -Dtest=GpuDemoApplication -Dcomprehensive=true

# Run with performance focus
mvn test -Dtest=GpuDemoApplication -Dperformance.only=true
```

**Expected Output:**
````
