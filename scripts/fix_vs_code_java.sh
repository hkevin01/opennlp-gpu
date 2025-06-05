#!/bin/bash

echo "Fixing VS Code Java and task provider issues..."

# Function to clean VS Code workspace
clean_vscode_workspace() {
    echo "Cleaning VS Code workspace cache..."
    
    # Find and clean VS Code workspace storage
    if [ -d "$HOME/.config/Code/User/workspaceStorage" ]; then
        echo "Clearing workspace storage..."
        rm -rf "$HOME/.config/Code/User/workspaceStorage"/*
    fi
    
    # Clean VS Code cache
    if [ -d "$HOME/.config/Code/Cache" ]; then
        echo "Clearing VS Code cache..."
        rm -rf "$HOME/.config/Code/Cache"/*
    fi
    
    if [ -d "$HOME/.config/Code/CachedData" ]; then
        echo "Clearing VS Code cached data..."
        rm -rf "$HOME/.config/Code/CachedData"/*
    fi
}

# Function to fix VS Code settings
fix_vscode_settings() {
    echo "Creating optimized VS Code settings for Java..."
    
    # Create .vscode directory if it doesn't exist
    mkdir -p .vscode
    
    # Create settings.json with Java-specific optimizations
    cat << 'EOF' > .vscode/settings.json
{
    // Java settings
    "java.autobuild.enabled": false,
    "java.debug.settings.enableRunDebugCodeLens": false,
    "java.configuration.updateBuildConfiguration": "automatic",
    "java.server.launchMode": "Standard",
    "java.compile.nullAnalysis.mode": "automatic",
    
    // Disable problematic task auto-detection
    "task.autoDetect": "off",
    "gradle.autoDetect": "off",
    "npm.autoDetect": "off",
    "gulp.autoDetect": "off",
    "grunt.autoDetect": "off",
    "jake.autoDetect": "off",
    "typescript.tsc.autoDetect": "off",
    
    // Java project settings
    "java.project.importOnFirstTimeStartup": "automatic",
    "java.project.referencedLibraries": [
        "lib
