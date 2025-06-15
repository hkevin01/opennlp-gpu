# Legacy Scripts

This directory contains historical scripts from the early development phase when the project used Gradle as the build system. These scripts are maintained for reference and educational purposes.

## ⚠️ **IMPORTANT**: These Scripts are DEPRECATED

**Current Build System**: Maven  
**Legacy Build System**: Gradle (no longer used)  
**Status**: ✅ **PRESERVED FOR HISTORICAL REFERENCE**

## Purpose

These scripts solved build system conflicts during the early project development when we experimented with Gradle for Java/C++ hybrid builds. The project now uses Maven exclusively for better OpenNLP integration.

## Historical Context

### Original Problem (2023)
The project initially attempted to use Gradle for building both Java GPU acceleration code and native C++/CUDA kernels in a single build system. This created complex plugin conflicts:

