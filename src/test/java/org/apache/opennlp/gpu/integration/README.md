# OpenNLP Integration Tests

This directory contains integration tests that validate GPU acceleration using real OpenNLP models, datasets, and workflows.

## 🎯 Integration Test Files Overview

### `OpenNLPTestDataIntegration.java`
**Purpose**: Comprehensive integration testing with real OpenNLP datasets and models  
**What it tests**: Real-world performance, accuracy validation, data compatibility, model integration

**Run complete integration tests:**
```bash
# Run all integration tests
mvn test -Dtest=OpenNLPTestDataIntegration

# Run with real OpenNLP data download
mvn test -Dtest=OpenNLPTestDataIntegration -Ddownload.real.data=true

# Run with specific dataset sizes
mvn test -Dtest=OpenNLPTestDataIntegration -Dtest.sizes=10,50,100,500
```

**Run programmatically:**
```java
// Execute integration tests from code
OpenNLPTestDataIntegration integration = new OpenNLPTestDataIntegration();
integration.runRealModelTests();
```

**Expected Output:**
