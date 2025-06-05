#!/bin/bash

# Exit on error
set -e

echo "===== Fixing Lombok imports and logger declarations ====="

PROJECT_ROOT="$(pwd)"

# Fix Lombok imports
fix_lombok_imports() {
    echo "Fixing Lombok imports in Java files..."
    
    find "$PROJECT_ROOT/src" -name "*.java" | while read -r file; do
        # Fix incorrect Slf4j imports
        if grep -q "import lombok.extern/slf4j" "$file"; then
            echo "  Fixing incorrect SLF4J import in $(basename "$file")"
            sed -i 's|import lombok.extern/slf4j.*|import lombok.extern.slf4j.Slf4j;|g' "$file"
        elif grep -q "import lombok.extern.slf4j/log4j" "$file"; then
            echo "  Fixing incorrect SLF4J import in $(basename "$file")"
            sed -i 's|import lombok.extern.slf4j/log4j.*|import lombok.extern.slf4j.Slf4j;|g' "$file"
        fi
        
        # Check if file uses loggers but has no @Slf4j annotation
        if grep -q "log\." "$file" || grep -q "logger\." "$file"; then
            if ! grep -q "@Slf4j" "$file" && ! grep -q "Logger.*=.*LoggerFactory" "$file"; then
                echo "  Adding @Slf4j annotation to $(basename "$file")"
                
                # Add import if missing
                if ! grep -q "import lombok.extern.slf4j.Slf4j" "$file"; then
                    sed -i '/package /a import lombok.extern.slf4j.Slf4j;' "$file"
                fi
                
                # Add annotation before class declaration
                sed -i 's/public class/\@Slf4j\npublic class/g' "$file"
            fi
        fi
        
        # Replace logger. with log. if @Slf4j is used
        if grep -q "@Slf4j" "$file" && grep -q "logger\." "$file"; then
            echo "  Converting logger to log in $(basename "$file")"
            sed -i 's/logger\./log\./g' "$file"
        fi
    done
    
    echo "Lombok imports fixed."
}

# Create VS Code settings for better Lombok support
create_vscode_settings() {
    echo "Creating VS Code settings for Lombok support..."
    
    mkdir -p "$PROJECT_ROOT/.vscode"
    
    cat << EOF > "$PROJECT_ROOT/.vscode/settings.json"
{
    "java.configuration.updateBuildConfiguration": "automatic",
    "java.jdt.ls.lombokSupport": true,
    "java.format.enabled": true,
    "java.import.gradle.enabled": false,
    "java.import.maven.enabled": true,
    "java.completion.importOrder": [
        "java",
        "javax",
        "org",
        "com",
        "",
        "#"
    ],
    "files.exclude": {
        "**/.git": true,
        "**/.svn": true,
        "**/.hg": true,
        "**/CVS": true,
        "**/.DS_Store": true,
        "**/target": false,
        "**/bin": false
    }
}
EOF
    
    echo "VS Code settings created."
}

# Clean up Maven project
clean_project() {
    echo "Cleaning and rebuilding project..."
    
    if command -v mvn &>/dev/null; then
        mvn clean
        mvn compile -Dmaven.compiler.showWarnings=true
        echo "Project cleaned and rebuilt."
    else
        echo "Maven not found. Please run 'mvn clean compile' manually."
    fi
}

# Run the fixes
fix_lombok_imports
create_vscode_settings
clean_project

echo "===== Lombok and SLF4J configuration fixed ====="
echo "You may need to restart your IDE for changes to take effect."
chmod +x "$0"
