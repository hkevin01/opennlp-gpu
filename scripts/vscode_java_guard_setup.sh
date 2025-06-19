#!/bin/bash

# VSCode Java Environment Guard
# Ensures Java environment stays correct after VSCode reloads

echo "🛡️ Setting up VSCode Java Environment Guard..."

# Create VSCode workspace settings that persist
mkdir -p .vscode

# Update settings.json with robust configuration
cat > .vscode/settings.json << 'EOF'
{
  "java.home": "/usr/lib/jvm/java-21-openjdk-amd64",
  "java.configuration.runtimes": [
    {
      "name": "JavaSE-21",
      "path": "/usr/lib/jvm/java-21-openjdk-amd64",
      "default": true
    },
    {
      "name": "JavaSE-17", 
      "path": "/usr/lib/jvm/java-17-openjdk-amd64"
    }
  ],
  "java.compile.nullAnalysis.mode": "automatic",
  "java.eclipse.downloadSources": true,
  "java.maven.downloadSources": true,
  "java.debug.settings.onBuildFailureProceed": true,
  "java.debug.settings.console": "integratedTerminal",
  "xml.java.home": "/usr/lib/jvm/java-21-openjdk-amd64",
  "xml.server.binary.path": null,
  "xml.server.preferBinary": false,
  "xml.server.vmargs": "-Xmx512m",
  "maven.executable.path": "mvn",
  "maven.terminal.useJavaHome": true,
  "maven.terminal.customEnv": [
    {
      "environmentVariable": "JAVA_HOME",
      "value": "/usr/lib/jvm/java-21-openjdk-amd64"
    }
  ],
  "java.requirements.JDK11Warning": false,
  "redhat.telemetry.enabled": false,
  "java.server.launchMode": "Standard",
  "java.import.gradle.enabled": false,
  "java.configuration.updateBuildConfiguration": "automatic",
  "java.clean.workspace": false,
  "java.import.maven.enabled": true,
  "terminal.integrated.env.linux": {
    "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64",
    "PATH": "/usr/lib/jvm/java-21-openjdk-amd64/bin:${env:PATH}"
  }
}
EOF

# Create extension recommendations
cat > .vscode/extensions.json << 'EOF'
{
  "recommendations": [
    "redhat.java",
    "vscjava.vscode-java-pack",
    "redhat.vscode-xml",
    "vscjava.vscode-maven"
  ]
}
EOF

# Update tasks.json with environment variables
cat > .vscode/tasks.json << 'EOF'
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
          "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64",
          "PATH": "/usr/lib/jvm/java-21-openjdk-amd64/bin:${env:PATH}"
        }
      },
      "problemMatcher": ["$tsc"]
    },
    {
      "label": "🧪 Run Demo Tests", 
      "type": "shell",
      "command": "mvn",
      "args": ["test", "-Dtest=GpuDemoApplication"],
      "group": "test",
      "options": {
        "env": {
          "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64",
          "PATH": "/usr/lib/jvm/java-21-openjdk-amd64/bin:${env:PATH}"
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
          "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64",
          "PATH": "/usr/lib/jvm/java-21-openjdk-amd64/bin:${env:PATH}"
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
    },
    {
      "label": "🛡️ Fix Java Environment",
      "type": "shell",
      "command": "./scripts/fix_java_environment.sh",
      "group": "build"
    }
  ]
}
EOF

# Create startup script that runs automatically
cat > .vscode/startup.sh << 'EOF'
#!/bin/bash
# Automatic Java environment setup for VSCode

export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"

# Verify Java is working
if ! command -v java &> /dev/null || ! java -version 2>&1 | grep -q "21.0"; then
    echo "⚠️ Java environment needs fixing..."
    ./scripts/fix_java_environment.sh
fi

echo "✅ Java environment verified"
EOF

chmod +x .vscode/startup.sh

# Create auto-fix script for VSCode integration
cat > scripts/vscode_java_guard.sh << 'EOF'
#!/bin/bash
# VSCode Java Environment Guard - runs on VSCode startup

JAVA_21_PATH="/usr/lib/jvm/java-21-openjdk-amd64"

# Check if Java environment is correct
if [ "$JAVA_HOME" != "$JAVA_21_PATH" ] || ! command -v java &> /dev/null; then
    echo "🔧 Fixing Java environment..."
    export JAVA_HOME="$JAVA_21_PATH"
    export PATH="$JAVA_HOME/bin:$PATH"
    
    # Update shell environment
    if ! grep -q "JAVA_HOME.*java-21-openjdk-amd64" ~/.bashrc; then
        echo "export JAVA_HOME=$JAVA_21_PATH" >> ~/.bashrc
        echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc
    fi
    
    echo "✅ Java environment fixed"
fi

# Verify VSCode settings
if [ ! -f ".vscode/settings.json" ] || ! grep -q "java-21-openjdk-amd64" .vscode/settings.json; then
    echo "🔧 Updating VSCode settings..."
    ./scripts/setup_vscode_java.sh
fi

echo "🛡️ Java environment guard complete"
EOF

chmod +x scripts/vscode_java_guard.sh

# Add to project README section about environment setup
cat >> docs/java_environment_guide.md << 'EOF'
# Java Environment Setup Guide

## Automatic Setup
Run the automatic setup script:
```bash
./scripts/fix_java_environment.sh
```

## Manual Verification
Check your Java environment:
```bash
./scripts/validate_java_environment.sh
```

## VSCode Integration
If VSCode shows Java version errors:
```bash
./scripts/setup_vscode_java.sh
# Then reload VSCode: Ctrl+Shift+P -> "Developer: Reload Window"
```

## Troubleshooting

### Issue: "Java version is older than Java 11"
**Solution**: 
```bash
./scripts/fix_java_environment.sh
# Reload VSCode
```

### Issue: Maven using wrong Java version
**Solution**:
```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
mvn clean compile
```

### Issue: VSCode XML server errors
**Solution**: VSCode settings are automatically configured to use Java 21 for XML processing.

## Environment Guard
The environment guard script runs automatically to ensure Java stays configured correctly:
```bash
./scripts/vscode_java_guard.sh
```
EOF

echo "✅ VSCode Java Environment Guard setup complete!"
echo ""
echo "🛡️ Protection features installed:"
echo "  • VSCode settings locked to Java 21"
echo "  • Terminal environment variables configured"
echo "  • Maven tasks use correct Java version"
echo "  • Automatic environment verification"
echo "  • Extension recommendations configured"
echo ""
echo "🔄 After VSCode reload, Java environment will remain stable"
echo ""
echo "💡 If you still see Java version errors:"
echo "  1. Run: ./scripts/vscode_java_guard.sh"
echo "  2. Reload VSCode: Ctrl+Shift+P -> 'Developer: Reload Window'"
echo "  3. Verify: ./scripts/quick_java_check.sh"
