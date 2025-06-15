#!/bin/bash

echo "Comprehensive Gradle Removal & Maven Setup Script"
echo "================================================"
echo ""
echo "This script will:"
echo "1. Remove ALL Gradle-related files and folders"
echo "2. Create a complete Maven project structure"
echo ""

# Function to find all Gradle files with explicit checks
find_all_gradle_files() {
    echo "Searching for all Gradle-related files and folders..."
    echo ""
    
    # Arrays to store found files
    GRADLE_FILES=()
    GRADLE_DIRS=()
    WRAPPER_FILES=()
    
    # Explicitly check for gradle wrapper files
    [ -f "gradlew" ] && WRAPPER_FILES+=("gradlew")
    [ -f "gradlew.bat" ] && WRAPPER_FILES+=("gradlew.bat")
    [ -f "gradle/wrapper/gradle-wrapper.jar" ] && WRAPPER_FILES+=("gradle/wrapper/gradle-wrapper.jar")
    [ -f "gradle/wrapper/gradle-wrapper.properties" ] && WRAPPER_FILES+=("gradle/wrapper/gradle-wrapper.properties")
    
    # Find all .gradle files
    while IFS= read -r -d '' file; do
        GRADLE_FILES+=("$file")
    done < <(find . -name "*.gradle" -type f ! -path "./gradle_backup_*" -print0 2>/dev/null)
    
    # Find all .gradle.kts files
    while IFS= read -r -d '' file; do
        GRADLE_FILES+=("$file")
    done < <(find . -name "*.gradle.kts" -type f ! -path "./gradle_backup_*" -print0 2>/dev/null)
    
    # Find specific files
    [ -f "gradle.properties" ] && GRADLE_FILES+=("gradle.properties")
    [ -f "settings.gradle" ] && GRADLE_FILES+=("settings.gradle")
    [ -f "settings.gradle.kts" ] && GRADLE_FILES+=("settings.gradle.kts")
    [ -f "build.gradle" ] && GRADLE_FILES+=("build.gradle")
    [ -f "build.gradle.kts" ] && GRADLE_FILES+=("build.gradle.kts")
    [ -f "rocm.gradle" ] && GRADLE_FILES+=("rocm.gradle")
    [ -f "dependencies.gradle" ] && GRADLE_FILES+=("dependencies.gradle")
    [ -f "lombok.config" ] && GRADLE_FILES+=("lombok.config")
    
    # Find backup files
    while IFS= read -r -d '' file; do
        GRADLE_FILES+=("$file")
    done < <(find . -regex ".*gradle.*\.\(bak\|backup\|old\|orig\|save\|[0-9]+\).*" -type f -print0 2>/dev/null)
    
    # Explicitly check for Gradle directories
    [ -d "gradle" ] && GRADLE_DIRS+=("gradle")
    [ -d ".gradle" ] && GRADLE_DIRS+=(".gradle")
    [ -d "build" ] && GRADLE_DIRS+=("build")
    [ -d "buildSrc" ] && GRADLE_DIRS+=("buildSrc")
    [ -d ".gradle-wrapper" ] && GRADLE_DIRS+=(".gradle-wrapper")
    
    # Find .gradle directories in subdirectories
    while IFS= read -r -d '' dir; do
        GRADLE_DIRS+=("$dir")
    done < <(find . -name ".gradle" -type d ! -path "./gradle_backup_*" -print0 2>/dev/null)
    
    # Find build directories in subdirectories
    while IFS= read -r -d '' dir; do
        GRADLE_DIRS+=("$dir")
    done < <(find . -name "build" -type d ! -path "./gradle_backup_*" ! -path "./src/*" ! -path "./target/*" -print0 2>/dev/null)
    
    # IDE specific files
    [ -f ".idea/gradle.xml" ] && GRADLE_FILES+=(".idea/gradle.xml")
    [ -d ".idea/libraries" ] && find .idea/libraries -name "Gradle__*" -type f | while read -r file; do
        GRADLE_FILES+=("$file")
    done
}

# Function to display found files
display_found_files() {
    echo "Files and directories found:"
    echo "============================"
    
    if [ ${#WRAPPER_FILES[@]} -gt 0 ]; then
        echo ""
        echo "Gradle Wrapper Files:"
        echo "--------------------"
        printf '%s\n' "${WRAPPER_FILES[@]}" | while read -r file; do
            echo "  ✓ $file"
        done
    fi
    
    if [ ${#GRADLE_FILES[@]} -gt 0 ]; then
        echo ""
        echo "Gradle Build Files:"
        echo "------------------"
        printf '%s\n' "${GRADLE_FILES[@]}" | sort -u | while read -r file; do
            echo "  ✓ $file"
        done
    fi
    
    if [ ${#GRADLE_DIRS[@]} -gt 0 ]; then
        echo ""
        echo "Gradle Directories:"
        echo "------------------"
        printf '%s\n' "${GRADLE_DIRS[@]}" | sort -u | while read -r dir; do
            SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "  ✓ $dir/ ($SIZE)"
        done
    fi
    
    echo ""
    TOTAL_FILES=$((${#GRADLE_FILES[@]} + ${#WRAPPER_FILES[@]}))
    TOTAL_ITEMS=$((TOTAL_FILES + ${#GRADLE_DIRS[@]}))
    echo "Total: $TOTAL_FILES files and ${#GRADLE_DIRS[@]} directories found"
    echo ""
}

# Function to create backup
create_backup() {
    echo "Creating backup of all Gradle files..."
    
    BACKUP_DIR="gradle_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup all files preserving structure
    for file in "${GRADLE_FILES[@]}" "${WRAPPER_FILES[@]}"; do
        if [ -f "$file" ]; then
            DIR=$(dirname "$file")
            mkdir -p "$BACKUP_DIR/$DIR"
            cp -p "$file" "$BACKUP_DIR/$file"
        fi
    done
    
    # Backup directories
    for dir in "${GRADLE_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            cp -rp "$dir" "$BACKUP_DIR/"
        fi
    done
    
    echo "✓ Backup created in: $BACKUP_DIR/"
    echo ""
}

# Function to remove all Gradle files
remove_all_gradle_files() {
    echo "Removing all Gradle files and directories..."
    echo ""
    
    # Remove wrapper files first
    for file in "${WRAPPER_FILES[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file" && echo "✓ Removed wrapper file: $file"
        fi
    done
    
    # Remove the gradle directory (including wrapper subdirectory)
    if [ -d "gradle" ]; then
        echo "✓ Removing gradle/ directory and all contents..."
        rm -rf "gradle"
    fi
    
    # Remove .gradle directory
    if [ -d ".gradle" ]; then
        echo "✓ Removing .gradle/ directory..."
        rm -rf ".gradle"
    fi
    
    # Remove other Gradle files
    for file in "${GRADLE_FILES[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file" && echo "✓ Removed: $file"
        fi
    done
    
    # Remove other Gradle directories
    for dir in "${GRADLE_DIRS[@]}"; do
        if [ -d "$dir" ] && [ "$dir" != "gradle" ] && [ "$dir" != ".gradle" ]; then
            rm -rf "$dir" && echo "✓ Removed: $dir/"
        fi
    done
    
    # Clean VS Code settings
    if [ -f ".vscode/settings.json" ]; then
        if grep -q "gradle" ".vscode/settings.json" 2>/dev/null; then
            echo ""
            echo "Cleaning Gradle settings from VS Code configuration..."
            if command -v jq &> /dev/null; then
                jq 'del(.["gradle.autoDetect"], .["gradle.nestedProjects"], .["java.import.gradle.enabled"], .["java.import.gradle.wrapper.enabled"], .["java.import.gradle.version"], .["java.import.gradle.home"], .["java.import.gradle.java.home"], .["java.import.gradle.offline.enabled"], .["java.import.gradle.arguments"], .["java.import.gradle.jvmArguments"], .["java.import.gradle.user.home"])' .vscode/settings.json > .vscode/settings.json.tmp && mv .vscode/settings.json.tmp .vscode/settings.json
                echo "✓ Cleaned Gradle settings from VS Code"
            fi
        fi
    fi
    
    echo ""
}

# Function to create complete Maven project structure
create_maven_structure() {
    echo "Creating complete Maven project structure..."
    echo ""
    
    # Main source directories
    mkdir -p src/main/java
    mkdir -p src/main/resources
    mkdir -p src/main/webapp/WEB-INF
    mkdir -p src/main/filters
    mkdir -p src/main/assembly
    
    # Test directories
    mkdir -p src/test/java
    mkdir -p src/test/resources
    mkdir -p src/test/filters
    
    # Integration test directories
    mkdir -p src/it/java
    mkdir -p src/it/resources
    
    # Site documentation
    mkdir -p src/site
    mkdir -p src/site/markdown
    mkdir -p src/site/resources
    
    # Additional Maven directories
    mkdir -p src/main/scripts
    mkdir -p src/main/config
    mkdir -p src/main/docker
    
    echo "✓ Created Maven source directories"
    
    # Create resource files
    if [ ! -f "src/main/resources/application.properties" ]; then
        cat > src/main/resources/application.properties << EOF
# Application Configuration
app.name=${PROJECT_NAME:-MyApp}
app.version=1.0.0

# Logging Configuration
logging.level.root=INFO
logging.level.com.example=DEBUG
EOF
        echo "✓ Created application.properties"
    fi
    
    # Create logback configuration
    if [ ! -f "src/main/resources/logback.xml" ]; then
        cat > src/main/resources/logback.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>
    
    <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <file>logs/application.log</file>
        <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
            <fileNamePattern>logs/application.%d{yyyy-MM-dd}.log</fileNamePattern>
            <maxHistory>30</maxHistory>
        </rollingPolicy>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>
    
    <root level="INFO">
        <appender-ref ref="CONSOLE" />
        <appender-ref ref="FILE" />
    </root>
</configuration>
EOF
        echo "✓ Created logback.xml"
    fi
    
    # Create test resources
    if [ ! -f "src/test/resources/test.properties" ]; then
        cat > src/test/resources/test.properties << EOF
# Test Configuration
test.environment=true
test.database.url=jdbc:h2:mem:test
test.database.driver=org.h2.Driver
EOF
        echo "✓ Created test.properties"
    fi
    
    # Create Maven wrapper if mvn exists
    if command -v mvn &> /dev/null; then
        echo ""
        echo "Installing Maven wrapper..."
        mvn wrapper:wrapper -Dmaven=3.9.6 2>/dev/null || true
        if [ -f "mvnw" ]; then
            echo "✓ Maven wrapper installed"
        fi
    fi
    
    # Create .gitignore for Maven
    cat > .gitignore << 'EOF'
# Maven
target/
pom.xml.tag
pom.xml.releaseBackup
pom.xml.versionsBackup
pom.xml.next
release.properties
dependency-reduced-pom.xml
buildNumber.properties
.mvn/timing.properties
.mvn/wrapper/maven-wrapper.jar

# IDE
.idea/
*.iml
*.ipr
*.iws
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json
.classpath
.project
.settings/
.factorypath

# OS
.DS_Store
Thumbs.db
desktop.ini

# Logs
logs/
*.log

# Java
*.class
*.jar
*.war
*.ear
*.zip
*.tar.gz
*.rar
hs_err_pid*
replay_pid*

# Gradle remnants (to be safe)
.gradle/
build/
gradle/
gradlew
gradlew.bat
*.gradle
*.gradle.kts
gradle.properties
EOF
    echo "✓ Created .gitignore"
    
    # Create README for Maven
    if [ ! -f "README.md" ]; then
        cat > README.md << EOF
# ${PROJECT_NAME:-Maven Project}

This project has been migrated from Gradle to Maven.

## Building the Project

\`\`\`bash
# Clean and compile
mvn clean compile

# Run tests
mvn test

# Package as JAR
mvn package

# Install to local repository
mvn clean install
\`\`\`

## Maven Wrapper

If Maven wrapper is installed:
\`\`\`bash
./mvnw clean install
\`\`\`

## Project Structure

- \`src/main/java\` - Main Java source files
- \`src/main/resources\` - Main resources
- \`src/test/java\` - Test Java source files
- \`src/test/resources\` - Test resources
- \`target/\` - Build output directory

## IDE Support

- **IntelliJ IDEA**: Import as Maven project
- **Eclipse**: Import as Maven project
- **VS Code**: Open folder, install Java Extension Pack
EOF
        echo "✓ Created README.md"
    fi
    
    echo ""
    echo "Maven directory structure:"
    tree -d -L 3 src/ 2>/dev/null || find src -type d | sort
    echo ""
}

# Main execution
echo "Current directory: $(pwd)"
PROJECT_NAME=$(basename "$(pwd)")
echo ""

# Find all Gradle files
find_all_gradle_files

# Display what was found
display_found_files

# Check if any Gradle files exist
if [ ${#GRADLE_FILES[@]} -eq 0 ] && [ ${#GRADLE_DIRS[@]} -eq 0 ] && [ ${#WRAPPER_FILES[@]} -eq 0 ]; then
    echo "No Gradle files found. Proceeding to create Maven structure..."
    create_maven_structure
    exit 0
fi

# Ask for backup
echo "Do you want to create a backup before removal? (y/n)"
read -r backup_response

if [[ "$backup_response" =~ ^[Yy]$ ]]; then
    create_backup
fi

# Confirmation
echo ""
echo "⚠ WARNING: This will permanently delete all Gradle files!"
echo "Type 'DELETE' to confirm:"
read -r confirm

if [ "$confirm" != "DELETE" ]; then
    echo "Cancelled."
    exit 0
fi

# Remove all Gradle files
echo ""
remove_all_gradle_files

# Stop Gradle daemons
echo "Stopping any Gradle daemons..."
pkill -f gradle 2>/dev/null || true
pkill -f GradleDaemon 2>/dev/null || true

# Create Maven structure
echo ""
create_maven_structure

# Clean user Gradle directory
echo ""
echo "Remove Gradle from user home (~/.gradle)? (y/n)"
read -r clean_home

if [[ "$clean_home" =~ ^[Yy]$ ]]; then
    if [ -d "$HOME/.gradle" ]; then
        SIZE=$(du -sh "$HOME/.gradle" 2>/dev/null | cut -f1)
        rm -rf "$HOME/.gradle"
        echo "✓ Removed ~/.gradle ($SIZE freed)"
    fi
fi

# Final summary
echo ""
echo "======================================="
echo "✓ Complete!"
echo "======================================="
echo ""
echo "Removed:"
echo "- All Gradle build files (*.gradle, *.gradle.kts)"
echo "- Gradle wrapper (gradlew, gradlew.bat)"
echo "- Gradle directories (gradle/, .gradle/, build/, buildSrc/)"
echo "- All Gradle backup files"
echo ""
echo "Created Maven structure:"
echo "- src/main/java - Java source files"
echo "- src/main/resources - Resources and config"
echo "- src/test/java - Test source files"
echo "- src/test/resources - Test resources"
echo "- src/it/* - Integration test directories"
echo "- src/site/* - Site documentation"
echo "- Configuration files (logback.xml, application.properties)"
echo ""

if [ -f "pom.xml" ]; then
    echo "Maven commands:"
    echo "  mvn clean compile     - Clean and compile"
    echo "  mvn test              - Run tests"
    echo "  mvn package           - Create JAR"
    echo "  mvn clean install     - Full build"
    if [ -f "mvnw" ]; then
        echo "  ./mvnw clean install  - Using wrapper"
    fi
else
    echo "⚠ Note: pom.xml not found. Create one to use Maven."
fi

echo ""
echo "Next steps:"
echo "1. Reload VS Code: Ctrl+Shift+P -> 'Developer: Reload Window'"
echo "2. Let Java extensions recognize the Maven project"
echo "3. Run: mvn clean compile"
echo ""
echo "Done!"