# Contributing to OpenNLP GPU

Thank you for your interest in contributing to the OpenNLP GPU project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [GPU-Specific Considerations](#gpu-specific-considerations)

## Code of Conduct

This project follows the [Apache Software Foundation Code of Conduct](https://www.apache.org/foundation/policies/conduct.html). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/opennlp-gpu.git
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/original-repo/opennlp-gpu.git
   ```
4. **Set up the development environment**:
   - Install JDK 8 or higher
   - Install Maven 3.6 or higher
   - Install GPU drivers and CUDA/OpenCL as needed

## Development Workflow

1. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   or
   ```bash
   git checkout -b fix/issue-description
   ```

2. **Make your changes**:
   - Follow the coding standards (see below)
   - Add appropriate tests
   - Update documentation as needed

3. **Build and test locally**:
   ```bash
   mvn clean install
   ```

4. **Commit your changes** with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a pull request** against the main branch of the upstream repository

## Pull Request Process

1. **Ensure your code follows our standards** and passes all tests
2. **Update documentation** relevant to the changes you're making
3. **Include screenshots or output examples** for UI or behavior changes
4. **Describe your changes** in detail in the PR description
5. **Link any related issues** using GitHub's issue linking syntax
6. **Be responsive to feedback** and be prepared to make requested changes

PRs will be merged once they've been approved by at least one maintainer.

## Coding Standards

This project follows the [Apache OpenNLP coding standards](https://opennlp.apache.org/dev-contribution.html#coding-standards):

- Use 2 spaces for indentation (no tabs)
- Use meaningful variable and method names
- Add JavaDoc comments for all public methods and classes
- Keep methods small and focused
- Follow the Java naming conventions

Additionally, for GPU-specific code:
- Always provide CPU fallback implementations
- Handle device memory carefully to avoid leaks
- Document performance characteristics
- Include benchmarking code for GPU operations

## Testing Guidelines

All contributions should include appropriate tests:

- **Unit tests** for individual classes and methods
- **Integration tests** for component interactions
- **GPU-specific tests** that verify correctness on GPU hardware
- **Performance tests** for critical operations

For GPU tests, include conditional logic to skip tests when appropriate hardware is not available.

## Documentation

Good documentation is crucial, especially for GPU-accelerated code:

- **JavaDoc** for all public classes and methods
- **Implementation notes** for complex algorithms
- **Performance guidance** in relevant sections
- **Hardware requirements** clearly stated
- **Examples** demonstrating correct usage

## GPU-Specific Considerations

When working with GPU acceleration:

1. **Resource management** is critical - always release GPU resources
2. **Error handling** should be robust with appropriate fallbacks
3. **Configuration options** should allow fine-tuning for different hardware
4. **Precision differences** between CPU and GPU implementations should be documented
5. **Memory transfers** should be minimized and optimized
6. **Batch processing** should be used where appropriate

## Project Structure

The project follows this high-level structure:

- `src/main/java/org/apache/opennlp/gpu/common` - Common abstractions and interfaces
- `src/main/java/org/apache/opennlp/gpu/compute` - Computational operations
- `src/main/java/org/apache/opennlp/gpu/ml` - Machine learning implementations
- `src/test` - Test code and resources

## Need Help?

If you need help or have questions, please:

1. Check the [documentation](docs/)
2. Look for existing [issues](https://github.com/original-repo/opennlp-gpu/issues)
3. Reach out to the maintainers

Thank you for contributing to OpenNLP GPU!
