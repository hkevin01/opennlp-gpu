# OpenNLP GPU Scripts

This directory contains utility scripts for the OpenNLP GPU acceleration project.

## Available Scripts

### 🔧 `check_ide_setup.sh`
**Purpose**: Comprehensive IDE setup checker and fixer

**What it does**:
- ✅ Verifies Maven and Java installation
- ✅ Checks project compilation status
- ✅ Validates Maven dependencies
- ✅ Detects IDE configuration files
- ✅ Creates missing VS Code configurations
- ✅ Tests demo class availability
- ✅ Provides IDE-specific fix instructions
- ✅ Generates quick fix scripts

**Usage**:
```bash
# Make executable and run
chmod +x scripts/check_ide_setup.sh
./scripts/check_ide_setup.sh

# Or run directly
bash scripts/check_ide_setup.sh
```

**Expected Output**:
````
# OpenNLP GPU Build Scripts

This directory contains scripts to help fix common build issues with the OpenNLP GPU project.

## Available Scripts

### fix_plugin_conflict.sh

Fixes the conflict between the Java and C++ Gradle plugins by:
- Removing the `cpp-library` and `cpp-unit-test` plugins
- Configuring a custom native build approach using CMake directly
- Preserving native build integration with Java compilation

Usage:
```bash
./scripts/fix_plugin_conflict.sh
```

### upgrade_gradle.sh

Upgrades the Gradle wrapper to version 8.10 which supports Java 21:
- Updates the Gradle wrapper properties
- Downloads the new Gradle wrapper if needed
- Configures the wrapper to use Gradle 8.10

Usage:
```bash
./scripts/upgrade_gradle.sh
```

### make_executable.sh

Makes the other scripts executable:

Usage:
```bash
bash scripts/make_executable.sh
```

## Common Issues Fixed

1. **Plugin Conflict**: 
   ```
   A problem occurred configuring root project 'opennlp-gpu'.
   An exception occurred applying plugin request [id: 'cpp-library']
   Failed to apply plugin 'org.gradle.cpp-library'.
   Could not create an instance of type org.gradle.language.cpp.internal.DefaultCppLibrary.
   Could not create an instance of type org.gradle.language.internal.DefaultLibraryDependencies.
   Cannot add a configuration with name 'implementation' as a configuration with that name already exists.
   ```

2. **Java Version Compatibility**:
   ```
   Unsupported Java Runtime: The Java version: 21, that is selected for the project is not supported by Gradle 8.4.
   The IDE will attempt to use Gradle 8.10 to gather the project information.
   ```

## After Running the Scripts

After running these scripts, rebuild your project with:

```bash
./gradlew clean build
```
