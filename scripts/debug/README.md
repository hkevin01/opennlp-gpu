# Debug Scripts and Tools

This directory contains debugging utilities and diagnostic tools for the OpenNLP GPU project.

## Files

- `debug_lspci.java` - Java source for GPU detection debugging
- `debug_lspci.class` - Compiled Java class for GPU debugging

## Usage

These debug tools are used during development and troubleshooting to:

- Diagnose GPU detection issues
- Debug system configuration problems
- Troubleshoot hardware compatibility

## Running Debug Tools

```bash
# Compile and run the GPU debug tool
cd scripts/debug
javac debug_lspci.java
java debug_lspci
```

These tools are automatically used by the main setup scripts when diagnosing system issues.
