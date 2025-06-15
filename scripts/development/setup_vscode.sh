#!/bin/bash

# OpenNLP GPU VS Code Setup Script
# Fixes Java version issues and configures VS Code for optimal Maven/Java development

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 OpenNLP GPU - VS Code Setup & Java Configuration${NC}"
echo -e "${BLUE}=================================================${NC}"

# Check if we're in the right directory
if [ ! -f "pom.xml" ]; then
    echo -e "${RED}❌ No pom.xml found. Please run from project root.${NC}"
    exit 1
fi

echo -e "${YELLOW}📁 Project Root: $(pwd)${NC}"

# Function to check and report status
check_status() {
    local check_name="$1"
    local command="$2"
    local success_msg="$3"
    local fail_msg="$4"
    
    echo -e "\n${PURPLE}🔍 Checking: ${check_name}${NC}"
    
    if eval $command > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${success_msg}${NC}"
        return 0
    else
        echo -e "${RED}❌ ${fail_msg}${NC}"
        return 1
    fi
}

# Check Java installations
echo -e "\n${PURPLE}🔍 Checking Java Installations${NC}"

# Check current Java version
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1 | awk -F '"' '{print $2}')
    echo -e "${YELLOW}Current Java: ${JAVA_VERSION}${NC}"
    
    # Extract major version
    if [[ $JAVA_VERSION == 1.8* ]]; then
        JAVA_MAJOR=8
    else
        JAVA_MAJOR=$(echo $JAVA_VERSION | cut -d. -f1)
    fi
    
    if [ "$JAVA_MAJOR" -ge 11 ]; then
        echo -e "${GREEN}✅ Java $JAVA_MAJOR detected (VS Code compatible)${NC}"
        JAVA_11_PLUS=true
    else
        echo -e "${YELLOW}⚠️ Java $JAVA_MAJOR detected (older than 11)${NC}"
        JAVA_11_PLUS=false
    fi
else
    echo -e "${RED}❌ Java not found in PATH${NC}"
    JAVA_11_PLUS=false
fi

# Look for Java 11+ installations
echo -e "\n${PURPLE}🔍 Scanning for Java 11+ installations...${NC}"

JAVA_CANDIDATES=(
    "/usr/lib/jvm/java-11-openjdk-amd64"
    "/usr/lib/jvm/java-17-openjdk-amd64"
    "/usr/lib/jvm/java-21-openjdk-amd64"
    "/usr/lib/jvm/default-java"
    "/opt/java/openjdk"
    "/usr/lib/jvm/adoptopenjdk-11-hotspot-amd64"
    "/usr/lib/jvm/temurin-11-jdk-amd64"
    "/usr/lib/jvm/temurin-17-jdk-amd64"
)

FOUND_JAVA_11=""
for candidate in "${JAVA_CANDIDATES[@]}"; do
    if [ -f "$candidate/bin/java" ]; then
        VERSION=$($candidate/bin/java -version 2>&1 | head -n 1 | awk -F '"' '{print $2}')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        if [[ $VERSION == 1.* ]]; then
            MAJOR=$(echo $VERSION | cut -d. -f2)
        fi
        
        if [ "$MAJOR" -ge 11 ]; then
            echo -e "${GREEN}✅ Found Java $MAJOR at: $candidate${NC}"
            if [ -z "$FOUND_JAVA_11" ]; then
                FOUND_JAVA_11="$candidate"
            fi
        fi
    fi
done

# Install Java 11+ if not found
if [ -z "$FOUND_JAVA_11" ]; then
    echo -e "${YELLOW}⚠️ No Java 11+ found. Installing OpenJDK 11...${NC}"
    
    if command -v apt &> /dev/null; then
        echo -e "${YELLOW}Installing via apt...${NC}"
        sudo apt update
        sudo apt install -y openjdk-11-jdk
        FOUND_JAVA_11="/usr/lib/jvm/java-11-openjdk-amd64"
    elif command -v yum &> /dev/null; then
        echo -e "${YELLOW}Installing via yum...${NC}"
        sudo yum install -y java-11-openjdk-devel
        FOUND_JAVA_11="/usr/lib/jvm/java-11-openjdk"
    else
        echo -e "${RED}❌ Cannot install Java automatically. Please install Java 11+ manually.${NC}"
    fi
fi

# Create VS Code configuration directory
echo -e "\n${PURPLE}🔧 Setting up VS Code configuration...${NC}"
mkdir -p .vscode

# Create comprehensive VS Code settings
echo -e "${YELLOW}Creating VS Code settings.json...${NC}"
cat > .vscode/settings.json << EOF
{
    "java.configuration.updateBuildConfiguration": "automatic",
    "java.compile.nullAnalysis.mode": "automatic",
    "java.maven.downloadSources": true,
    "java.maven.downloadJavadoc": true,
    "java.server.launchMode": "Standard",
    "maven.executable.path": "mvn",
    "java.debug.settings.enableRunDebugCodeLens": true,
    "java.eclipse.downloadSources": true,
    "java.maven.updateSnapshots": true,
    "java.sources.organizeImports.starThreshold": 5,
    "java.sources.organizeImports.staticStarThreshold": 3,
    "java.format.settings.url": "",
    "java.import.maven.enabled": true,
    "java.import.gradle.enabled": false,
    "java.saveActions.organizeImports": true,
    "java.configuration.runtimes": [
        {
            "name": "JavaSE-${MAJOR_11:-11}",
            "path": "$FOUND_JAVA_11",
            "default": true
        }
    ],
    "java.home": "$FOUND_JAVA_11",
    "maven.terminal.useJavaHome": true,
    "maven.terminal.customEnv": [
        {
            "environmentVariable": "JAVA_HOME",
            "value": "$FOUND_JAVA_11"
        }
    ],
    // VS Code XML extension: set xml.java.home to match Java runtime
    "xml.server.preferBinary": true,
    "xml.java.home": "$FOUND_JAVA_11",
    "files.exclude": {
        "**/target": true,
        "**/.settings": true,
        "**/.project": true,
        "**/.classpath": true
    }
}
EOF

echo -e "${GREEN}✅ Created VS Code settings.json with Java $MAJOR_11 configuration (xml.java.home set to $FOUND_JAVA_11)${NC}"

# Create launch configurations for demos
echo -e "${YELLOW}Creating VS Code launch.json...${NC}"
cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "java",
            "name": "🚀 Run GpuDemoApplication",
            "request": "launch",
            "mainClass": "org.apache.opennlp.gpu.demo.GpuDemoApplication",
            "projectName": "opennlp-gpu",
            "args": [],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "JAVA_HOME": "${config:java.home}"
            }
        },
        {
            "type": "java",
            "name": "🧪 Run SimpleGpuDemo",
            "request": "launch",
            "mainClass": "org.apache.opennlp.gpu.demo.SimpleGpuDemo",
            "projectName": "opennlp-gpu",
            "args": [],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "JAVA_HOME": "${config:java.home}"
            }
        },
        {
            "type": "java",
            "name": "📋 Run ComprehensiveDemoTestSuite",
            "request": "launch",
            "mainClass": "org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite",
            "projectName": "opennlp-gpu",
            "args": [],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "JAVA_HOME": "${config:java.home}"
            }
        },
        {
            "type": "java",
            "name": "🏃 Run Current File",
            "request": "launch",
            "mainClass": "${file}",
            "projectName": "opennlp-gpu",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "JAVA_HOME": "${config:java.home}"
            }
        }
    ]
}
EOF

echo -e "${GREEN}✅ Created VS Code launch.json with demo configurations${NC}"

# Create tasks.json for Maven operations
echo -e "${YELLOW}Creating VS Code tasks.json...${NC}"
cat > .vscode/tasks.json << EOF
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "🔨 Maven Clean Compile",
            "type": "shell",
            "command": "mvn",
            "args": ["clean", "compile"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "options": {
                "env": {
                    "JAVA_HOME": "$FOUND_JAVA_11"
                }
            },
            "problemMatcher": ["\$java"]
        },
        {
            "label": "🧪 Run Demo Tests",
            "type": "shell",
            "command": "mvn",
            "args": ["test", "-Dtest=GpuDemoApplication"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "options": {
                "env": {
                    "JAVA_HOME": "$FOUND_JAVA_11"
                }
            },
            "dependsOn": "🔨 Maven Clean Compile"
        },
        {
            "label": "📋 Run All Demos",
            "type": "shell",
            "command": "./scripts/run_all_demos.sh",
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "options": {
                "env": {
                    "JAVA_HOME": "$FOUND_JAVA_11"
                }
            }
        },
        {
            "label": "🔧 Check IDE Setup",
            "type": "shell",
            "command": "./scripts/check_ide_setup.sh",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
EOF

echo -e "${GREEN}✅ Created VS Code tasks.json with Maven tasks${NC}"

# Create extensions.json for recommended extensions
echo -e "${YELLOW}Creating VS Code extensions.json...${NC}"
cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "redhat.java",
        "vscjava.vscode-java-pack",
        "vscjava.vscode-maven",
        "vscjava.vscode-java-test",
        "vscjava.vscode-java-debug",
        "redhat.vscode-xml",
        "ms-vscode.test-adapter-converter"
    ]
}
EOF

echo -e "${GREEN}✅ Created VS Code extensions.json${NC}"

# Test Java setup
echo -e "\n${PURPLE}🧪 Testing Java Configuration${NC}"

if [ -n "$FOUND_JAVA_11" ]; then
    echo -e "${YELLOW}Testing Java $MAJOR_11 at $FOUND_JAVA_11...${NC}"
    
    export JAVA_HOME="$FOUND_JAVA_11"
    export PATH="$JAVA_HOME/bin:$PATH"
    
    JAVA_TEST_VERSION=$($JAVA_HOME/bin/java -version 2>&1 | head -n 1)
    echo -e "${GREEN}✅ Java Test: $JAVA_TEST_VERSION${NC}"
    
    # Test Maven with new Java
    if command -v mvn &> /dev/null; then
        echo -e "${YELLOW}Testing Maven with Java $MAJOR_11...${NC}"
        if JAVA_HOME="$FOUND_JAVA_11" mvn -version > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Maven works with Java $MAJOR_11${NC}"
        else
            echo -e "${RED}❌ Maven failed with Java $MAJOR_11${NC}"
        fi
    fi
fi

# Compile project with new Java setup
echo -e "\n${PURPLE}🔨 Testing Project Compilation${NC}"
if [ -n "$FOUND_JAVA_11" ]; then
    echo -e "${YELLOW}Compiling with Java $MAJOR_11...${NC}"
    
    export JAVA_HOME="$FOUND_JAVA_11"
    if mvn clean compile -q; then
        echo -e "${GREEN}✅ Project compiles successfully with Java $MAJOR_11${NC}"
        COMPILE_SUCCESS=true
    else
        echo -e "${RED}❌ Project compilation failed${NC}"
        COMPILE_SUCCESS=false
    fi
else
    echo -e "${YELLOW}⚠️ Skipping compilation test (no Java 11+ found)${NC}"
    COMPILE_SUCCESS=false
fi

# Generate reload script
echo -e "\n${PURPLE}🔧 Creating VS Code Reload Script${NC}"

cat > scripts/reload_vscode.sh << EOF
#!/bin/bash
# VS Code reload script for OpenNLP GPU project

echo "🔄 Reloading VS Code Java configuration..."

# Set Java environment
export JAVA_HOME="$FOUND_JAVA_11"
export PATH="\$JAVA_HOME/bin:\$PATH"

# Kill VS Code Java processes
echo "Stopping VS Code Java processes..."
pkill -f "java.*jdt.ls" 2>/dev/null || true
sleep 2

# Clean project
echo "Cleaning Maven project..."
mvn clean -q

# Restart with proper Java
echo "VS Code will now use Java $MAJOR_11"
echo "1. Restart VS Code"
echo "2. Press Ctrl+Shift+P"
echo "3. Run: 'Java: Reload Projects'"
echo "4. Run: 'Java: Rebuild Workspace'"
EOF

chmod +x scripts/reload_vscode.sh
echo -e "${GREEN}✅ Created VS Code reload script${NC}"

# Summary and instructions
echo -e "\n${BLUE}📋 VS Code Setup Summary${NC}"
echo -e "${BLUE}========================${NC}"

if [ -n "$FOUND_JAVA_11" ]; then
    echo -e "${GREEN}✅ Java $MAJOR_11 configured at: $FOUND_JAVA_11${NC}"
    echo -e "${GREEN}✅ VS Code settings updated for Java $MAJOR_11${NC}"
    echo -e "${GREEN}✅ Launch configurations created for demos${NC}"
    echo -e "${GREEN}✅ Maven tasks configured${NC}"
    
    if [ "$COMPILE_SUCCESS" = true ]; then
        echo -e "${GREEN}✅ Project compiles successfully${NC}"
    fi
    
    echo -e "\n${YELLOW}🎯 Next Steps:${NC}"
    echo -e "1. Restart VS Code"
    echo -e "2. Install recommended extensions (VS Code will prompt)"
    echo -e "3. Press Ctrl+Shift+P → 'Java: Reload Projects'"
    echo -e "4. Press Ctrl+Shift+P → 'Java: Rebuild Workspace'"
    echo -e "5. Try running demos with F5 or 'Run Java' code lens"
    
    echo -e "\n${PURPLE}🚀 Available VS Code Commands:${NC}"
    echo -e "- F5: Run current file"
    echo -e "- Ctrl+Shift+P → 'Tasks: Run Task' → '🧪 Run Demo Tests'"
    echo -e "- Ctrl+Shift+P → 'Tasks: Run Task' → '📋 Run All Demos'"
    echo -e "- Right-click demo files → Run Java"
    
else
    echo -e "${RED}❌ Could not configure Java 11+${NC}"
    echo -e "${YELLOW}💡 Manual installation required:${NC}"
    echo -e "   sudo apt install openjdk-11-jdk"
    echo -e "   Then re-run this script"
fi

echo -e "\n${PURPLE}🔧 If you still see Java version warnings:${NC}"
echo -e "   Run: ./scripts/reload_vscode.sh"
echo -e "   This will ensure VS Code uses Java $MAJOR_11"

echo -e "\n${BLUE}✅ VS Code setup complete!${NC}"
