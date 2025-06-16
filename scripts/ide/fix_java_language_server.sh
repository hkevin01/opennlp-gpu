#!/bin/bash
# Fix Oracle Java SE Language Server Issues
# This script resolves common Java Language Server problems in VS Code

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Java Language Server Fix Script${NC}"
echo -e "${BLUE}===================================${NC}"

# Get project root
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo -e "${YELLOW}📁 Project Root: ${PROJECT_ROOT}${NC}"

# Function to check Java installation
check_java() {
    echo -e "\n${BLUE}☕ Checking Java Installation...${NC}"
    
    if command -v java >/dev/null 2>&1; then
        JAVA_VERSION=$(java -version 2>&1 | head -n1 | cut -d'"' -f2)
        echo -e "${GREEN}✅ Java found: ${JAVA_VERSION}${NC}"
        
        # Extract major version number
        JAVA_MAJOR=$(echo "$JAVA_VERSION" | cut -d'.' -f1)
        if [[ "$JAVA_MAJOR" == "1" ]]; then
            JAVA_MAJOR=$(echo "$JAVA_VERSION" | cut -d'.' -f2)
        fi
        
        echo -e "${BLUE}📊 Java Major Version: ${JAVA_MAJOR}${NC}"
        
        if command -v javac >/dev/null 2>&1; then
            JAVAC_VERSION=$(javac -version 2>&1 | cut -d' ' -f2)
            echo -e "${GREEN}✅ Java Compiler found: ${JAVAC_VERSION}${NC}"
        else
            echo -e "${RED}❌ Java Compiler (javac) not found${NC}"
            echo -e "${YELLOW}💡 Install OpenJDK development package${NC}"
        fi
    else
        echo -e "${RED}❌ Java not found${NC}"
        echo -e "${YELLOW}💡 Please install Java 8 or higher${NC}"
        return 1
    fi
    
    # Check JAVA_HOME
    if [ -n "$JAVA_HOME" ]; then
        echo -e "${GREEN}✅ JAVA_HOME set: ${JAVA_HOME}${NC}"
    else
        echo -e "${YELLOW}⚠️ JAVA_HOME not set${NC}"
        # Try to detect Java home
        if command -v readlink >/dev/null 2>&1 && command -v dirname >/dev/null 2>&1; then
            DETECTED_JAVA_HOME=$(readlink -f $(which java) | sed "s/bin\/java//")
            echo -e "${BLUE}🔍 Detected Java Home: ${DETECTED_JAVA_HOME}${NC}"
            export JAVA_HOME="$DETECTED_JAVA_HOME"
        fi
    fi
}

# Function to fix Oracle Java extension conflicts
fix_oracle_java_extension() {
    echo -e "\n${BLUE}🔧 Fixing Oracle Java Extension Conflicts...${NC}"
    
    # Check if Oracle Java extension is installed
    if command -v code >/dev/null 2>&1; then
        if code --list-extensions | grep -q "Oracle.oracle-java"; then
            echo -e "${YELLOW}⚠️ Oracle Java extension detected${NC}"
            echo -e "${BLUE}💡 Oracle Java extension requires Java 17+, but project uses Java 8${NC}"
            
            echo -e "${YELLOW}🔄 Disabling Oracle Java extension...${NC}"
            code --disable-extension Oracle.oracle-java
            
            echo -e "${GREEN}✅ Oracle Java extension disabled${NC}"
        else
            echo -e "${GREEN}✅ Oracle Java extension not installed${NC}"
        fi
        
        # Ensure Red Hat Java extension is installed and enabled
        if ! code --list-extensions | grep -q "redhat.java"; then
            echo -e "${BLUE}📦 Installing Red Hat Java extension...${NC}"
            code --install-extension redhat.java --force
        else
            echo -e "${GREEN}✅ Red Hat Java extension already installed${NC}"
        fi
        
        # Enable Red Hat Java extension
        echo -e "${BLUE}🔄 Ensuring Red Hat Java extension is enabled...${NC}"
        code --enable-extension redhat.java
        
    else
        echo -e "${YELLOW}⚠️ VS Code CLI not found${NC}"
        echo -e "${BLUE}💡 Manual steps:${NC}"
        echo -e "  1. Open VS Code"
        echo -e "  2. Go to Extensions (Ctrl+Shift+X)"
        echo -e "  3. Disable: Oracle Java extension"
        echo -e "  4. Enable: Red Hat Java Language Support"
    fi
}

# Function to clean VS Code workspace
clean_vscode_workspace() {
    echo -e "\n${BLUE}🧹 Cleaning VS Code Workspace...${NC}"
    
    # Remove VS Code workspace metadata
    WORKSPACE_DIRS=(
        "${PROJECT_ROOT}/.vscode/settings.json.bak"
        "${PROJECT_ROOT}/.metadata"
        "${PROJECT_ROOT}/.project"
        "${PROJECT_ROOT}/.classpath"
        "${PROJECT_ROOT}/.settings"
        "${PROJECT_ROOT}/bin"
        "${HOME}/.config/Code/User/workspaceStorage/*/Oracle.oracle-java"
    )
    
    for dir in "${WORKSPACE_DIRS[@]}"; do
        if [ -e "$dir" ]; then
            echo -e "${YELLOW}🗑️ Removing: $dir${NC}"
            rm -rf "$dir"
        fi
    done
    
    # Clean target directory
    if [ -d "${PROJECT_ROOT}/target" ]; then
        echo -e "${YELLOW}🗑️ Cleaning Maven target directory${NC}"
        rm -rf "${PROJECT_ROOT}/target"
    fi
    
    # Clean Oracle Java user directories
    ORACLE_DIRS=(
        "${HOME}/.config/Code/User/workspaceStorage/*/Oracle.oracle-java/userdir"
        "${HOME}/.cache/vscode-java"
        "${HOME}/.vscode/extensions/oracle.oracle-java-*/userdir"
    )
    
    for dir in "${ORACLE_DIRS[@]}"; do
        if [ -e "$dir" ]; then
            echo -e "${YELLOW}🗑️ Removing Oracle Java cache: $dir${NC}"
            rm -rf "$dir"
        fi
    done
}

# Function to update VS Code settings
update_vscode_settings() {
    echo -e "\n${BLUE}⚙️ Updating VS Code Settings...${NC}"
    
    VSCODE_DIR="${PROJECT_ROOT}/.vscode"
    mkdir -p "$VSCODE_DIR"
    
    # Create or update settings.json with Java 8 specific settings
    cat > "${VSCODE_DIR}/settings.json" << 'EOF'
{
    "java.home": null,
    "java.jdt.ls.vmargs": "-noverify -Xmx1G -XX:+UseG1GC -XX:+UseStringDeduplication",
    "java.configuration.detectJdksAtStart": false,
    "java.configuration.checkProjectSettingsExclusions": false,
    "java.import.maven.enabled": true,
    "java.import.gradle.enabled": false,
    "java.maven.downloadSources": true,
    "java.maven.downloadJavadoc": false,
    "java.compile.nullAnalysis.mode": "disabled",
    "java.signatureHelp.enabled": true,
    "java.contentProvider.preferred": "fernflower",
    "java.debug.settings.enableRunDebugCodeLens": true,
    "java.test.editor.enableShortcuts": true,
    "java.completion.enabled": true,
    "java.completion.overwrite": true,
    "java.completion.guessMethodArguments": true,
    "java.completion.favoriteStaticMembers": [
        "org.junit.jupiter.api.Assertions.*",
        "org.junit.Assert.*",
        "org.mockito.Mockito.*"
    ],
    "java.sources.organizeImports.starThreshold": 99,
    "java.sources.organizeImports.staticStarThreshold": 99,
    "java.eclipse.downloadSources": true,
    "java.maven.updateSnapshots": false,
    "java.format.enabled": true,
    "java.format.settings.url": null,
    "java.format.settings.profile": null,
    "java.cleanup.actionsOnSave": [],
    "maven.executable.path": "",
    "maven.terminal.useJavaHome": true,
    "maven.terminal.customEnv": [
        {
            "environmentVariable": "JAVA_HOME",
            "value": "${java.home}"
        }
    ],
    "java.configuration.runtimes": [
        {
            "name": "JavaSE-1.8",
            "path": "${env:JAVA_HOME}",
            "default": true
        }
    ],
    "java.compile.nullAnalysis.mode": "automatic",
    "extensions.ignoreRecommendations": false,
    "java.server.launchMode": "Standard"
}
EOF
    
    echo -e "${GREEN}✅ Updated VS Code settings for Java 8${NC}"
    
    # Create extensions.json to recommend correct extensions
    cat > "${VSCODE_DIR}/extensions.json" << 'EOF'
{
    "recommendations": [
        "redhat.java",
        "vscjava.vscode-java-debug",
        "vscjava.vscode-java-test",
        "vscjava.vscode-maven",
        "vscjava.vscode-java-dependency"
    ],
    "unwantedRecommendations": [
        "oracle.oracle-java"
    ]
}
EOF
    
    echo -e "${GREEN}✅ Created extension recommendations${NC}"
    
    # ...existing launch.json creation code...
}

# Function to check VS Code extensions
check_vscode_extensions() {
    echo -e "\n${BLUE}🔌 Checking VS Code Extensions...${NC}"
    
    REQUIRED_EXTENSIONS=(
        "redhat.java"
        "vscjava.vscode-java-debug"
        "vscjava.vscode-java-test"
        "vscjava.vscode-maven"
        "vscjava.vscode-java-dependency"
    )
    
    UNWANTED_EXTENSIONS=(
        "oracle.oracle-java"
    )
    
    if command -v code >/dev/null 2>&1; then
        echo -e "${GREEN}✅ VS Code CLI found${NC}"
        
        # Remove unwanted extensions
        for ext in "${UNWANTED_EXTENSIONS[@]}"; do
            if code --list-extensions | grep -q "$ext"; then
                echo -e "${YELLOW}🗑️ Removing conflicting extension: $ext${NC}"
                code --uninstall-extension "$ext"
            else
                echo -e "${GREEN}✅ Conflicting extension not installed: $ext${NC}"
            fi
        done
        
        # Install required extensions
        for ext in "${REQUIRED_EXTENSIONS[@]}"; do
            if code --list-extensions | grep -q "$ext"; then
                echo -e "${GREEN}✅ Extension installed: $ext${NC}"
            else
                echo -e "${YELLOW}⚠️ Extension missing: $ext${NC}"
                echo -e "${BLUE}💡 Installing: $ext${NC}"
                code --install-extension "$ext" --force
            fi
        done
    else
        echo -e "${YELLOW}⚠️ VS Code CLI not found${NC}"
        echo -e "${BLUE}💡 Manually install these extensions:${NC}"
        for ext in "${REQUIRED_EXTENSIONS[@]}"; do
            echo -e "  ✅ $ext"
        done
        echo -e "${BLUE}💡 Manually remove these extensions:${NC}"
        for ext in "${UNWANTED_EXTENSIONS[@]}"; do
            echo -e "  ❌ $ext"
        done
    fi
}

# Function to fix Maven configuration
fix_maven_config() {
    echo -e "\n${BLUE}📦 Checking Maven Configuration...${NC}"
    
    if [ -f "${PROJECT_ROOT}/pom.xml" ]; then
        echo -e "${GREEN}✅ pom.xml found${NC}"
        
        # Validate Maven project
        cd "$PROJECT_ROOT"
        if mvn validate -q; then
            echo -e "${GREEN}✅ Maven project is valid${NC}"
        else
            echo -e "${RED}❌ Maven validation failed${NC}"
            echo -e "${YELLOW}💡 Running Maven clean compile...${NC}"
            mvn clean compile -q
        fi
    else
        echo -e "${RED}❌ pom.xml not found${NC}"
        return 1
    fi
}

# Function to restart Java Language Server
restart_language_server() {
    echo -e "\n${BLUE}🔄 Instructions to Restart Java Language Server...${NC}"
    echo -e "${YELLOW}In VS Code:${NC}"
    echo -e "  1. Press ${BLUE}Ctrl+Shift+P${NC} (or ${BLUE}Cmd+Shift+P${NC} on Mac)"
    echo -e "  2. Type: ${BLUE}Java: Clean Workspace${NC}"
    echo -e "  3. Press Enter and restart VS Code"
    echo -e "  4. Alternatively, use: ${BLUE}Java: Restart Projects${NC}"
    echo -e "  5. If still failing, use: ${BLUE}Developer: Restart Extension Host${NC}"
    echo -e "  6. ${YELLOW}IMPORTANT${NC}: Ensure Oracle Java extension is disabled"
}

# Function to create Java environment script
create_java_env_script() {
    echo -e "\n${BLUE}📝 Creating Java Environment Script...${NC}"
    
    cat > "${PROJECT_ROOT}/scripts/ide/java_env.sh" << 'EOF'
#!/bin/bash
# Java Environment Setup for OpenNLP GPU

# Detect Java installation
if command -v java >/dev/null 2>&1; then
    JAVA_PATH=$(which java)
    JAVA_HOME_DETECTED=$(readlink -f "$JAVA_PATH" | sed 's/\/bin\/java$//')
    
    export JAVA_HOME="$JAVA_HOME_DETECTED"
    export PATH="$JAVA_HOME/bin:$PATH"
    
    # Get Java version
    JAVA_VERSION=$(java -version 2>&1 | head -n1 | cut -d'"' -f2)
    JAVA_MAJOR=$(echo "$JAVA_VERSION" | cut -d'.' -f1)
    if [[ "$JAVA_MAJOR" == "1" ]]; then
        JAVA_MAJOR=$(echo "$JAVA_VERSION" | cut -d'.' -f2)
    fi
    
    echo "Java Environment:"
    echo "  JAVA_HOME: $JAVA_HOME"
    echo "  Java Version: $JAVA_VERSION"
    echo "  Java Major: $JAVA_MAJOR"
    echo "  PATH: $PATH"
    
    # Warn if Java 17+ (Oracle extension conflict)
    if [[ "$JAVA_MAJOR" -ge 17 ]]; then
        echo "⚠️ Java $JAVA_MAJOR detected - Oracle Java extension may interfere"
        echo "💡 Use Red Hat Java extension for VS Code"
    fi
    
else
    echo "❌ Java not found. Please install Java 8 or higher."
    exit 1
fi

# Maven configuration
export MAVEN_OPTS="-Xmx1024m -XX:MaxPermSize=256m"
export M2_HOME="${M2_HOME:-/usr/share/maven}"

echo "Maven Environment:"
echo "  MAVEN_OPTS: $MAVEN_OPTS"
echo "  M2_HOME: $M2_HOME"

# VS Code Java configuration
export VSCODE_JAVA_HOME="$JAVA_HOME"
export VSCODE_MAVEN_HOME="$M2_HOME"

echo "VS Code Environment:"
echo "  VSCODE_JAVA_HOME: $VSCODE_JAVA_HOME"
echo "  VSCODE_MAVEN_HOME: $VSCODE_MAVEN_HOME"
EOF
    
    chmod +x "${PROJECT_ROOT}/scripts/ide/java_env.sh"
    echo -e "${GREEN}✅ Created Java environment script${NC}"
}

# Function to display troubleshooting info
show_troubleshooting() {
    echo -e "\n${BLUE}🔧 Oracle Java Extension Troubleshooting${NC}"
    echo -e "${BLUE}==========================================${NC}"
    
    echo -e "\n${YELLOW}Oracle Java Extension Issues:${NC}"
    echo -e "1. ${RED}Requires Java 17+${NC}: Oracle extension won't work with Java 8"
    echo -e "2. ${RED}Conflicts with Red Hat${NC}: Cannot run both extensions simultaneously"
    echo -e "3. ${RED}NetBeans LSP server${NC}: Oracle extension uses NetBeans, not Eclipse JDT"
    
    echo -e "\n${YELLOW}Solutions Applied:${NC}"
    echo -e "1. ${GREEN}Disabled Oracle Java extension${NC}"
    echo -e "2. ${GREEN}Enabled Red Hat Java Language Support${NC}"
    echo -e "3. ${GREEN}Configured Java 8 runtime settings${NC}"
    echo -e "4. ${GREEN}Cleaned workspace cache and metadata${NC}"
    
    echo -e "\n${YELLOW}Manual Extension Management:${NC}"
    echo -e "  Disable: ${RED}Oracle.oracle-java${NC}"
    echo -e "  Enable: ${GREEN}redhat.java${NC}"
    echo -e "  Enable: ${GREEN}vscjava.vscode-java-debug${NC}"
    echo -e "  Enable: ${GREEN}vscjava.vscode-maven${NC}"
    
    echo -e "\n${YELLOW}VS Code Commands (Ctrl+Shift+P):${NC}"
    echo -e "  - Extensions: Disable Oracle Java"
    echo -e "  - Extensions: Enable Red Hat Java Language Support"
    echo -e "  - Java: Clean Workspace"
    echo -e "  - Java: Restart Projects"
    echo -e "  - Developer: Restart Extension Host"
    echo -e "  - Developer: Reload Window"
    
    echo -e "\n${BLUE}💡 Key Points:${NC}"
    echo -e "  - This project uses Java 8 (OpenJDK 1.8.0_452)"
    echo -e "  - Oracle Java extension requires Java 17+"
    echo -e "  - Red Hat Java extension supports Java 8+"
    echo -e "  - Use Red Hat extension for Java 8 projects"
}

# Main execution
main() {
    echo -e "${BLUE}Starting Oracle Java Language Server fix...${NC}"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Run fix steps
    check_java
    fix_oracle_java_extension
    clean_vscode_workspace
    update_vscode_settings
    fix_maven_config
    check_vscode_extensions
    create_java_env_script
    restart_language_server
    show_troubleshooting
    
    echo -e "\n${GREEN}🎉 Oracle Java extension conflict resolved!${NC}"
    echo -e "${YELLOW}💡 Please restart VS Code completely and reopen the project.${NC}"
    echo -e "${BLUE}📋 Ensure Red Hat Java extension is active, Oracle Java disabled.${NC}"
}

# Run main function
main "$@"
