# Deployment Scripts

This directory contains scripts for production deployment preparation, release packaging, and build validation for the OpenNLP GPU acceleration project.

## Purpose

These scripts ensure production-ready quality, handle release packaging, and validate deployment readiness through comprehensive quality gates and automated checks.

## Available Scripts

### 🏗️ `validate_build.sh` - Pre-Deployment Validation
**Purpose**: Comprehensive build and quality validation before deployment

**What it validates**:
- ✅ **Clean Compilation**: Zero build errors across all source files
- ✅ **Test Suite Execution**: All unit, integration, and demo tests pass
- ✅ **Performance Benchmarks**: GPU acceleration performance meets targets
- ✅ **Documentation Completeness**: All required documentation present
- ✅ **Code Quality**: Static analysis and code coverage metrics
- ✅ **Dependency Security**: Vulnerability scanning and license compliance

**Quality Gates**:
```bash
# Build Quality Gates
✅ Maven clean compile: 0 errors, 0 warnings
✅ Test execution: >95% pass rate
✅ Performance: >2x GPU speedup on target operations
✅ Coverage: >90% line coverage, >85% branch coverage  
✅ Documentation: README, API docs, examples complete
✅ Security: No critical vulnerabilities detected
```

**Usage**:
```bash
./scripts/deployment/validate_build.sh

# Example output:
🏗️ OpenNLP GPU - Build Validation
==================================
✅ Clean compilation successful (59 source, 11 test files)
✅ All test suites passed (847 tests, 99.2% pass rate)
✅ Performance benchmarks meet targets (3.4x max speedup)
✅ Code coverage exceeds thresholds (92% line, 86% branch)
✅ Documentation validation complete
✅ Dependency security scan passed
🎉 Build ready for production deployment!
```

### 📦 `package_release.sh` - Release Packaging
**Purpose**: Create production-ready release packages with all dependencies

**What it packages**:
- ✅ **JAR Files**: Main application JAR with dependencies
- ✅ **Documentation**: API documentation, user guides, examples
- ✅ **Scripts**: Installation and setup scripts
- ✅ **Test Data**: Sample datasets and validation files  
- ✅ **GPU Kernels**: OpenCL/CUDA kernel files
- ✅ **Configuration**: Default configuration templates

**Package Formats**:
- 📦 **Standalone JAR**: `opennlp-gpu-{version}-standalone.jar`
- 📦 **Source Distribution**: `opennlp-gpu-{version}-src.tar.gz`
- 📦 **Binary Distribution**: `opennlp-gpu-{version}-bin.tar.gz`
- 📦 **Docker Image**: `opennlp-gpu:{version}` (planned)

**Usage**:
```bash
./scripts/deployment/package_release.sh

# Creates release packages in target/releases/
# Validates package integrity and completeness
# Generates checksums and signatures
```

**Package Structure**:
