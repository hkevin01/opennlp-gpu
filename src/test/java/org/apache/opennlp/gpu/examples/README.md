# GPU Acceleration Code Examples

This directory contains practical code examples and tutorials demonstrating how to use OpenNLP GPU acceleration in real-world scenarios.

## 🎯 Example Files Overview

### `GpuQuickStartDemo.java`
**Purpose**: Comprehensive tutorial showing all major GPU acceleration features  
**What it demonstrates**: Matrix operations, feature extraction, neural networks, OpenNLP integration

**Run the complete tutorial:**
```bash
# Run quick start demo
mvn test -Dtest=GpuQuickStartDemo

# Run with verbose output
mvn test -Dtest=GpuQuickStartDemo -Dverbose=true

# Run specific demo sections
mvn test -Dtest=GpuQuickStartDemo -Ddemo.sections=matrix,features,neural,integration
```

**Run from IDE:**
```java
// Execute in your IDE
public class MyTest {
    public static void main(String[] args) {
        GpuQuickStartDemo.main(args);
    }
}
```

**Expected Output:**
