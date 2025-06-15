# Core GPU Acceleration Tests

This directory contains the core unit and system tests for OpenNLP GPU acceleration components.

## 🎯 Test Files Overview

### `GpuTestSuite.java`
**Purpose**: Comprehensive test suite covering all GPU acceleration components  
**What it tests**: Matrix operations, feature extraction, neural networks, memory management, error handling

**Run individually:**
```bash
# Run complete test suite
mvn test -Dtest=GpuTestSuite

# Run with verbose output
mvn test -Dtest=GpuTestSuite -Dtest.verbose=true

# Run with specific GPU backend
mvn test -Dtest=GpuTestSuite -Dgpu.backend=opencl
```

**Run from IDE:**
```java
// In your IDE, run the main method:
GpuTestSuite testSuite = new GpuTestSuite();
GpuTestSuite.TestResults results = testSuite.runAllTests();
System.out.println(results.getReport());
```

**Expected Output:**
