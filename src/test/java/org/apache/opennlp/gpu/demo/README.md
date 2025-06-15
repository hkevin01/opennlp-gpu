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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

**🚀 Run All Demo Configurations:**

**Using Shell Script (Recommended):**
```bash
# Make script executable and run all demos
chmod +x scripts/run_all_demos.sh
./scripts/run_all_demos.sh

# Or run directly
bash scripts/run_all_demos.sh
```

**Using JUnit Test Suite:**
```bash
# Run comprehensive JUnit test suite
mvn test -Dtest=ComprehensiveDemoTestSuite

# Run programmatically
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite"

# Run individual test classes
mvn test -Dtest=ComprehensiveDemoTestSuite.BasicDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.OpenClDemoTest
mvn test -Dtest=ComprehensiveDemoTestSuite.DebugDemoTest
```

**Quick Start Commands:**
```bash
# One-liner to run everything
./scripts/run_all_demos.sh && echo "🎉 All demos completed!"

# Run with timing
time ./scripts/run_all_demos.sh

# Run and save output to log
./scripts/run_all_demos.sh 2>&1 | tee demo-results.log
```

## 🎯 Running the Comprehensive Demo Test Suite

### 🖱️ **IDE Right-Click Options (Recommended)**

**Option 1: Run as Java Application**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click anywhere in the file
3. Select **"Run 'ComprehensiveDemoTestSuite.main()'"**
4. ✅ Runs all demos with standalone output

**Option 2: Run as JUnit Test**
1. Open `ComprehensiveDemoTestSuite.java` in your IDE
2. Right-click on the class name
3. Select **"Run 'ComprehensiveDemoTestSuite'"**
4. ✅ Runs all demos through JUnit framework

**Option 3: Run Individual Tests**
1. Expand the test class in your IDE
2. Right-click on specific test (e.g., `BasicDemoTest`)
3. Select **"Run 'BasicDemoTest'"**
4. ✅ Runs only that specific demo configuration

### 🔧 **Before Running - Make Sure Project is Compiled**

```bash
# In terminal (required first time or after changes)
mvn clean compile

# Then you can right-click and run in IDE
```

### 📊 **What You'll See When Running**

**Via main() method (Option 1):**
````markdown
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

#