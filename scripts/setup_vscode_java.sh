#!/bin/bash

# VSCode Java Configuration Script
# Configures VSCode to use the correct Java runtime

echo "🔧 Setting up VSCode Java configuration..."

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Find the best Java installation
JAVA_17_PATH="/usr/lib/jvm/java-17-openjdk-amd64"
JAVA_21_PATH="/usr/lib/jvm/java-21-openjdk-amd64"

if [ -d "$JAVA_21_PATH" ]; then
    JAVA_HOME_PATH="$JAVA_21_PATH"
    echo -e "${GREEN}✅ Using Java 21: $JAVA_HOME_PATH${NC}"
elif [ -d "$JAVA_17_PATH" ]; then
    JAVA_HOME_PATH="$JAVA_17_PATH"
    echo -e "${GREEN}✅ Using Java 17: $JAVA_HOME_PATH${NC}"
else
    echo -e "${RED}❌ No suitable Java version found${NC}"
    echo "Please install Java 17 or 21:"
    echo "  sudo apt install openjdk-17-jdk"
    exit 1
fi

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Create VSCode settings
cat > .vscode/settings.json << EOF
{
  "java.home": "$JAVA_HOME_PATH",
  "java.configuration.runtimes": [
    {
      "name": "JavaSE-17",
      "path": "$JAVA_HOME_PATH"
    }
  ],
  "java.compile.nullAnalysis.mode": "automatic",
  "java.eclipse.downloadSources": true,
  "java.maven.downloadSources": true,
  "java.debug.settings.onBuildFailureProceed": true,
  "java.debug.settings.console": "integratedTerminal",
  "xml.java.home": "$JAVA_HOME_PATH",
  "xml.server.binary.path": null,
  "xml.server.preferBinary": false,
  "maven.executable.path": "mvn",
  "maven.terminal.useJavaHome": true,
  "java.requirements.JDK11Warning": false,
  "redhat.telemetry.enabled": false,
  "java.server.launchMode": "Standard",
  "java.import.gradle.enabled": false,
  "java.configuration.updateBuildConfiguration": "automatic"
}
EOF

echo -e "${GREEN}✅ VSCode settings configured${NC}"

# Create launch configuration for debugging
cat > .vscode/launch.json << EOF
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "java",
      "name": "Launch GpuDemoApplication",
      "request": "launch",
      "mainClass": "org.apache.opennlp.gpu.demo.GpuDemoApplication",
      "projectName": "opennlp-gpu",
      "env": {
        "JAVA_HOME": "$JAVA_HOME_PATH"
      }
    },
    {
      "type": "java",
      "name": "Debug Tests",
      "request": "launch",
      "mainClass": "",
      "projectName": "opennlp-gpu",
      "env": {
        "JAVA_HOME": "$JAVA_HOME_PATH"
      }
    }
  ]
}
EOF

echo -e "${GREEN}✅ VSCode launch configuration created${NC}"

# Create tasks configuration
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
      "options": {
        "env": {
          "JAVA_HOME": "$JAVA_HOME_PATH"
        }
      },
      "problemMatcher": ["\$tsc"]
    },
    {
      "label": "🧪 Run Demo Tests",
      "type": "shell",
      "command": "mvn",
      "args": ["test", "-Dtest=GpuDemoApplication"],
      "group": "test",
      "options": {
        "env": {
          "JAVA_HOME": "$JAVA_HOME_PATH"
        }
      },
      "dependsOn": "🔨 Maven Clean Compile"
    },
    {
      "label": "📋 Run All Demos",
      "type": "shell",
      "command": "./scripts/run_all_demos.sh",
      "group": "test",
      "options": {
        "env": {
          "JAVA_HOME": "$JAVA_HOME_PATH"
        }
      }
    },
    {
      "label": "🔧 Check IDE Setup",
      "type": "shell",
      "command": "./scripts/check_ide_setup.sh",
      "group": "build"
    },
    {
      "label": "☕ Validate Java Runtime",
      "type": "shell",
      "command": "./scripts/validate_java_runtime.sh",
      "group": "build"
    }
  ]
}
EOF

echo -e "${GREEN}✅ VSCode tasks configuration created${NC}"

# Set environment variables for current session
export JAVA_HOME="$JAVA_HOME_PATH"
export PATH="$JAVA_HOME/bin:$PATH"

echo -e "${YELLOW}⚠️  Environment variables set for current session${NC}"
echo "To make permanent, add to your ~/.bashrc:"
echo "  export JAVA_HOME=$JAVA_HOME_PATH"
echo "  export PATH=\$JAVA_HOME/bin:\$PATH"

# Test the configuration
echo ""
echo "🧪 Testing Java configuration..."
if command -v java &> /dev/null; then
    java -version
    echo -e "${GREEN}✅ Java command working${NC}"
else
    echo -e "${RED}❌ Java command not found${NC}"
fi

if command -v javac &> /dev/null; then
    javac -version
    echo -e "${GREEN}✅ Java compiler working${NC}"
else
    echo -e "${RED}❌ Java compiler not found${NC}"
fi

echo ""
echo -e "${GREEN}🎉 VSCode Java configuration complete!${NC}"
echo "Please reload VSCode for changes to take effect."
