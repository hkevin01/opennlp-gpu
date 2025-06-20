#!/bin/bash

# Comprehensive Java Environment Fix
# Fixes all Java-related issues for OpenNLP GPU project including VSCode integration

set -e

echo "🔧 Fixing Java Environment Completely..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Find best Java installation
echo "🔍 Searching for best Java installation..."

JAVA_21_PATH=""
JAVA_17_PATH=""
JAVA_11_PATH=""

# Check common Java installation paths
for java_path in /usr/lib/jvm/java-21-openjdk-amd64 /usr/lib/jvm/temurin-21-jdk-amd64 /usr/lib/jvm/java-17-openjdk-amd64 /usr/lib/jvm/temurin-17-jdk-amd64 /usr/lib/jvm/java-11-openjdk-amd64; do
    if [ -d "$java_path" ] && [ -f "$java_path/bin/java" ]; then
        version=$($java_path/bin/java -version 2>&1 | awk -F '"' '/version/ {print $2}')
        major=$(echo $version | cut -d. -f1)
        
        case "$major" in
            "21")
                JAVA_21_PATH="$java_path"
                print_status "Found Java 21: $java_path"
                ;;
            "17")
                JAVA_17_PATH="$java_path"
                print_status "Found Java 17: $java_path"
                ;;
            "11")
                JAVA_11_PATH="$java_path"
                print_status "Found Java 11: $java_path"
                ;;
        esac
    fi
done

# Determine best Java to use (prefer 21 > 17 > 11)
BEST_JAVA=""
if [ -n "$JAVA_21_PATH" ]; then
    BEST_JAVA="$JAVA_21_PATH"
    print_info "Selected: Java 21 (best performance)"
elif [ -n "$JAVA_17_PATH" ]; then
    BEST_JAVA="$JAVA_17_PATH"
    print_info "Selected: Java 17 (recommended)"
elif [ -n "$JAVA_11_PATH" ]; then
    BEST_JAVA="$JAVA_11_PATH"
    print_warning "Selected: Java 11 (minimum supported)"
else
    print_error "No suitable Java version found!"
    print_info "Please install Java 17 or later:"
    echo "  sudo apt update"
    echo "  sudo apt install openjdk-17-jdk"
    exit 1
fi

echo ""
print_info "Configuring Java environment to use: $BEST_JAVA"

# 1. Update project environment
echo ""
echo "🔧 Updating project environment..."

# Create/update .env file
echo "JAVA_HOME=$BEST_JAVA" > .env
echo "PATH=$BEST_JAVA/bin:\$PATH" >> .env
print_status "Created .env file with Java configuration"

# Update VSCode settings
echo ""
echo "🔧 Updating VSCode settings..."

mkdir -p .vscode

# Update settings.json
cat > .vscode/settings.json << EOF
{
    "java.home": "$BEST_JAVA",
    "java.jdt.ls.java.home": "$BEST_JAVA",
    "java.compile.nullAnalysis.mode": "automatic",
    "java.configuration.runtimes": [
        {
            "name": "JavaSE-17",
            "path": "$BEST_JAVA",
            "default": true
        }
    ],
    "maven.terminal.customEnv": [
        {
            "environmentVariable": "JAVA_HOME",
            "value": "$BEST_JAVA"
        }
    ],
    "redhat.telemetry.enabled": false,
    "xml.java.home": "$BEST_JAVA",
    "xml.server.preferBinary": false
}
EOF

print_status "Updated VSCode Java settings"

# Update tasks.json to ensure JAVA_HOME is set
cat > .vscode/tasks.json << EOF
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "🔨 Maven Clean Compile",
            "type": "shell",
            "command": "mvn",
            "args": [
                "clean",
                "compile"
            ],
            "group": "build",
            "options": {
                "env": {
                    "JAVA_HOME": "$BEST_JAVA"
                }
            }
        },
        {
            "label": "🧪 Run Demo Tests",
            "type": "shell",
            "command": "mvn",
            "args": [
                "test",
                "-Dtest=GpuDemoApplication"
            ],
            "group": "test",
            "options": {
                "env": {
                    "JAVA_HOME": "$BEST_JAVA"
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
                    "JAVA_HOME": "$BEST_JAVA"
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

print_status "Updated VSCode tasks with Java environment"

# 3. Create shell environment setup
echo ""
echo "🔧 Creating environment setup scripts..."

cat > scripts/setup_java_env.sh << 'EOF'
#!/bin/bash
# Source this file to set up Java environment
# Usage: source scripts/setup_java_env.sh

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Java environment loaded from .env"
    echo "   JAVA_HOME: $JAVA_HOME"
    echo "   Java version: $(java -version 2>&1 | head -1)"
else
    echo "❌ .env file not found. Run ./scripts/fix_java_environment.sh first"
fi
EOF

chmod +x scripts/setup_java_env.sh
print_status "Created environment setup script"

# 4. Update Maven wrapper if it exists
if [ -f "mvnw" ]; then
    echo ""
    echo "🔧 Updating Maven wrapper..."
    
    # Set JAVA_HOME for Maven wrapper
    export JAVA_HOME="$BEST_JAVA"
    export PATH="$JAVA_HOME/bin:$PATH"
    
    print_status "Updated Maven wrapper environment"
fi

# 5. Test the configuration
echo ""
echo "🧪 Testing configuration..."

# Source the environment
source scripts/setup_java_env.sh

# Test Java
if command -v java >/dev/null 2>&1; then
    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
    print_status "Java is working: $JAVA_VERSION"
else
    print_error "Java setup failed"
    exit 1
fi

# Test Maven
if command -v mvn >/dev/null 2>&1; then
    MVN_VERSION=$(mvn --version | head -1)
    print_status "Maven is working: $MVN_VERSION"
else
    print_warning "Maven not found"
fi

echo ""
print_status "Java environment auto-fix completed!"
echo ""
print_info "To apply the configuration:"
echo "  1. Restart VSCode"
echo "  2. Or run: source scripts/setup_java_env.sh"
echo "  3. Verify with: ./scripts/validate_java_runtime.sh"
echo ""
print_info "The configuration is now persistent and will work across restarts."
