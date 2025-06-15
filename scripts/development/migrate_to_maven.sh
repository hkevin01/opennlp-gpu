#!/bin/bash

echo "Migrating project from Gradle to Maven..."
echo "========================================="

PROJECT_DIR=$(pwd)
PROJECT_NAME=$(basename "$PROJECT_DIR")

# Function to extract dependencies from build.gradle
extract_dependencies() {
    echo "Extracting dependencies from build.gradle..."
    
    # Check if build.gradle exists
    if [ ! -f "build.gradle" ]; then
        echo "Warning: build.gradle not found"
        return 1
    fi
    
    # Extract Lombok version if present
    LOMBOK_VERSION=$(grep -oP "lombok:\K[\d.]+" build.gradle | head -1)
    if [ -z "$LOMBOK_VERSION" ]; then
        LOMBOK_VERSION="1.18.32"
    fi
    
    echo "Found Lombok version: $LOMBOK_VERSION"
}

# Function to create pom.xml
create_pom_xml() {
    echo "Creating pom.xml..."
    
    cat > pom.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>${PROJECT_NAME}</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>
    
    <name>${PROJECT_NAME}</name>
    <description>Migrated from Gradle to Maven</description>
    
    <properties>
        <maven.compiler.source>21</maven.compiler.source>
        <maven.compiler.target>21</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <lombok.version>${LOMBOK_VERSION}</lombok.version>
    </properties>
    
    <dependencies>
        <!-- Lombok -->
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <version>\${lombok.version}</version>
            <scope>provided</scope>
        </dependency>
        
        <!-- Logging -->
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-api</artifactId>
            <version>2.0.12</version>
        </dependency>
        <dependency>
            <groupId>ch.qos.logback</groupId>
            <artifactId>logback-classic</artifactId>
            <version>1.5.3</version>
        </dependency>
        
        <!-- Testing -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.10.2</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.mockito</groupId>
            <artifactId>mockito-core</artifactId>
            <version>5.11.0</version>
            <scope>test</scope>
        </dependency>
        
        <!-- Add your other dependencies here -->
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.13.0</version>
                <configuration>
                    <source>21</source>
                    <target>21</target>
                    <annotationProcessorPaths>
                        <path>
                            <groupId>org.projectlombok</groupId>
                            <artifactId>lombok</artifactId>
                            <version>\${lombok.version}</version>
                        </path>
                    </annotationProcessorPaths>
                </configuration>
            </plugin>
            
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.2.5</version>
            </plugin>
            
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>3.4.1</version>
                <configuration>
                    <archive>
                        <manifest>
                            <addClasspath>true</addClasspath>
                            <mainClass>com.example.Main</mainClass>
                        </manifest>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
EOF
    
    echo "✓ pom.xml created"
}

# Function to create Maven directory structure
create_maven_structure() {
    echo "Creating Maven directory structure..."
    
    # Create standard Maven directories
    mkdir -p src/main/java
    mkdir -p src/main/resources
    mkdir -p src/test/java
    mkdir -p src/test/resources
    
    # Move existing source files if they exist in Gradle structure
    if [ -d "src/main/java" ]; then
        echo "✓ Java source directory already exists"
    fi
    
    # Create resources directory files
    if [ ! -f "src/main/resources/application.properties" ]; then
        cat > src/main/resources/application.properties << EOF
# Application Configuration
app.name=${PROJECT_NAME}
app.version=1.0.0

# Logging Configuration
logging.level.root=INFO
logging.level.com.example=DEBUG
EOF
    fi
    
    # Create logback configuration
    cat > src/main/resources/logback.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>
    
    <root level="INFO">
        <appender-ref ref="CONSOLE" />
    </root>
</configuration>
EOF
    
    echo "✓ Maven directory structure created"
}

# Function to create VS Code settings for Maven
create_vscode_maven_settings() {
    echo "Creating VS Code settings for Maven..."
    
    mkdir -p .vscode
    
    # Create settings.json
    cat > .vscode/settings.json << 'EOF'
{
    // Java settings
    "java.configuration.updateBuildConfiguration": "automatic",
    "java.autobuild.enabled": true,
    "java.server.launchMode": "Standard",
    "java.compile.nullAnalysis.mode": "automatic",
    
    // Maven settings
    "maven.executable.preferMavenWrapper": false,
    "maven.terminal.useJavaHome": true,
    "maven.terminal.customEnv": [
        {
            "environmentVariable": "JAVA_HOME",
            "value": "${env:JAVA_HOME}"
        }
    ],
    
    // Performance settings
    "java.jdt.ls.vmargs": "-XX:+UseParallelGC -XX:GCTimeRatio=4 -XX:AdaptiveSizePolicyWeight=90 -Dsun.zip.disableMemoryMapping=true -Xmx2G -Xms100m",
    
    // Editor settings
    "editor.suggestSelection": "first",
    "editor.suggest.snippetsPreventQuickSuggestions": false,
    
    // Files to exclude
    "files.exclude": {
        "**/.classpath": true,
        "**/.project": true,
        "**/.settings": true,
        "**/.factorypath": true,
        "**/target": false
    },
    
    // Search exclusions
    "search.exclude": {
        "**/target": true,
        "**/node_modules": true,
        "**/.gradle": true,
        "**/build": true
    }
}
EOF

    # Create tasks.json for Maven
    cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "mvn: compile",
            "type": "shell",
            "command": "mvn compile",
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": "$maven"
        },
        {
            "label": "mvn: clean",
            "type": "shell",
            "command": "mvn clean",
            "group": "build",
            "problemMatcher": "$maven"
        },
        {
            "label": "mvn: test",
            "type": "shell",
            "command": "mvn test",
            "group": {
                "kind": "test",
                "isDefault": true
            },
            "problemMatcher": "$maven"
        },
        {
            "label": "mvn: package",
            "type": "shell",
            "command": "mvn package",
            "group": "build",
            "problemMatcher": "$maven"
        },
        {
            "label": "mvn: clean install",
            "type": "shell",
            "command": "mvn clean install",
            "group": "build",
            "problemMatcher": "$maven"
        }
    ]
}
EOF

    # Create launch.json
    cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "java",
            "name": "Launch Current File",
            "request": "launch",
            "mainClass": "${file}"
        },
        {
            "type": "java",
            "name": "Launch Main",
            "request": "launch",
            "mainClass": "com.example.Main",
            "projectName": "${workspaceFolderBasename}"
        }
    ]
}
EOF

    # Update extensions.json
    cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "vscjava.vscode-java-pack",
        "vscjava.vscode-maven",
        "gabrielbb.vscode-lombok"
    ]
}
EOF
    
    echo "✓ VS Code Maven settings created"
}

# Function to create .gitignore for Maven
create_maven_gitignore() {
    echo "Creating Maven .gitignore..."
    
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

# OS
.DS_Store
Thumbs.db

# Java
*.class
*.jar
*.war
*.ear
*.zip
*.tar.gz
*.rar

# Logs
*.log

# Gradle (old)
.gradle/
build/
gradle/
gradlew
gradlew.bat
build.gradle
settings.gradle
gradle.properties
EOF
    
    echo "✓ .gitignore created"
}

# Function to install Maven if not present
check_and_install_maven() {
    echo "Checking Maven installation..."
    
    if ! command -v mvn &> /dev/null; then
        echo "Maven not found. Installing Maven..."
        
        # Detect OS
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
        fi
        
        case $OS in
            ubuntu|debian)
                sudo apt-get update
                sudo apt-get install -y maven
                ;;
            fedora|rhel|centos)
                sudo dnf install -y maven
                ;;
            arch|manjaro)
                sudo pacman -S --noconfirm maven
                ;;
            *)
                echo "Please install Maven manually"
                return 1
                ;;
        esac
    else
        echo "✓ Maven is already installed"
        mvn -version
    fi
}

# Main execution
echo ""
echo "This will migrate your Gradle project to Maven"
echo ""

# Check for build.gradle
if [ -f "build.gradle" ]; then
    echo "Found build.gradle - extracting configuration..."
    extract_dependencies
else
    echo "No build.gradle found - creating default Maven configuration"
    LOMBOK_VERSION="1.18.32"
fi

# Backup existing files
if [ -f "pom.xml" ]; then
    cp pom.xml pom.xml.backup.$(date +%Y%m%d%H%M%S)
    echo "Backed up existing pom.xml"
fi

# Check and install Maven
check_and_install_maven

# Create Maven project structure
create_pom_xml
create_maven_structure
create_vscode_maven_settings
create_maven_gitignore

# Copy existing Java files
echo ""
echo "Checking for existing Java files..."
if [ -d "src" ]; then
    echo "✓ Source directory found"
else
    echo "Creating sample Main.java..."
    mkdir -p src/main/java/com/example
    cat > src/main/java/com/example/Main.java << 'EOF'
package com.example;

import lombok.extern.slf4j.Slf4j;
import lombok.Data;

@Slf4j
public class Main {
    public static void main(String[] args) {
        log.info("Hello from Maven!");
        
        Person person = new Person();
        person.setName("Maven User");
        person.setAge(30);
        
        System.out.println("Created: " + person);
    }
    
    @Data
    static class Person {
        private String name;
        private int age;
    }
}
EOF
fi

# Test Maven setup
echo ""
echo "Testing Maven setup..."
mvn clean compile

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Migration to Maven completed successfully!"
    echo ""
    echo "Maven commands you can now use:"
    echo "  mvn clean         - Clean the project"
    echo "  mvn compile       - Compile the project"
    echo "  mvn test          - Run tests"
    echo "  mvn package       - Create JAR file"
    echo "  mvn clean install - Clean, build, and install to local repository"
    echo ""
    echo "VS Code setup:"
    echo "1. Close VS Code if open: killall code"
    echo "2. Reopen with: code ."
    echo "3. VS Code should recognize the Maven project automatically"
    echo "4. Install recommended extensions when prompted"
    echo ""
    echo "Your Gradle files have been preserved but are now in .gitignore"
else
    echo ""
    echo "⚠ Maven compilation failed. Check the error messages above."
fi

# Cleanup Gradle daemon if running
echo ""
echo "Stopping Gradle daemon (if running)..."
pkill -f gradle 2>/dev/null || true

echo ""
echo "Migration complete!"