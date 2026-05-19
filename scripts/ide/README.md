# IDE Configuration Scripts

This directory contains scripts for setting up and optimizing Integrated Development Environment (IDE) configurations for the OpenNLP GPU project.

## Purpose

These scripts solve common IDE setup challenges, Java version conflicts, and development environment optimization for optimal GPU acceleration development.

## Available Scripts

### 🔧 `setup_vscode.sh` - Complete VS Code Setup
**Purpose**: Comprehensive VS Code configuration for Java 11+ development

**What it does**:
- ✅ Detects and installs Java 11+ automatically (OpenJDK)
- ✅ Configures VS Code settings.json with optimal Java runtime
- ✅ Creates launch.json with all demo application configurations
- ✅ Sets up tasks.json for Maven build automation
- ✅ Installs recommended Java extensions
- ✅ Fixes "Java version older than Java 11" warnings
- ✅ Configures XML language server to use correct Java

**Usage**:
```bash
# Run comprehensive VS Code setup
./scripts/ide/setup_vscode.sh

# What it creates:
# - .vscode/settings.json (Java runtime configuration)
# - .vscode/launch.json (Demo run configurations)  
# - .vscode/tasks.json (Maven tasks)
# - .vscode/extensions.json (Recommended extensions)
```

**Before/After**: