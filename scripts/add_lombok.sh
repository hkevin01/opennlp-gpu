#!/bin/bash

echo "Adding Lombok to the OpenNLP GPU project using Maven..."

# Define the Lombok version
LOMBOK_VERSION="1.18.30"

# Check if pom.xml exists
if [ ! -f "$(pwd)/pom.xml" ]; then
    echo "ERROR: pom.xml not found. Make sure you're in the project root directory."
    exit 1
fi

echo "Checking and updating pom.xml for Lombok..."

# Create a backup of the original file
cp "$(pwd)/pom.xml" "$(pwd)/pom.xml.bak"
echo "Created backup at $(pwd)/pom.xml.bak"

# Add Lombok dependency if it doesn't exist
if ! grep -q "<artifactId>lombok</artifactId>" "$(pwd)/pom.xml"; then
    sed -i '/<dependencies>/a\
        <dependency>\
            <groupId>org.projectlombok</groupId>\
            <artifactId>lombok</artifactId>\
            <version>'"$LOMBOK_VERSION"'</version>\
            <scope>provided</scope>\
        </dependency>' "$(pwd)/pom.xml"
    echo "Added Lombok dependency to pom.xml"
else
    echo "Lombok dependency already exists in pom.xml"
fi

# Ensure annotation processing is enabled in Maven compiler plugin
if ! grep -q "<annotationProcessorPaths>" "$(pwd)/pom.xml"; then
    sed -i '/<plugins>/a\
        <plugin>\
            <groupId>org.apache.maven.plugins</groupId>\
            <artifactId>maven-compiler-plugin</artifactId>\
            <version>3.8.1</version>\
            <configuration>\
                <annotationProcessorPaths>\
                    <path>\
                        <groupId>org.projectlombok</groupId>\
                        <artifactId>lombok</artifactId>\
                        <version>'"$LOMBOK_VERSION"'</version>\
                    </path>\
                </annotationProcessorPaths>\
            </configuration>\
        </plugin>' "$(pwd)/pom.xml"
    echo "Enabled annotation processing in Maven compiler plugin"
else
    echo "Annotation processing is already enabled in Maven compiler plugin"
fi

echo "Creating lombok.config file..."
cat << EOF > "$(pwd)/lombok.config"
# This file is generated for Lombok configuration
config.stopBubbling = true
lombok.addLombokGeneratedAnnotation = true
lombok.anyConstructor.addConstructorProperties = true
lombok.accessors.chain = true
lombok.fieldDefaults.defaultPrivate = true
lombok.fieldDefaults.defaultFinal = false
lombok.toString.doNotUseGetters = true
lombok.equalsAndHashCode.callSuper = call
lombok.log.fieldName = logger
EOF

echo "Uncomment Lombok imports in Java files..."
# Find all Java files with commented Lombok imports
for file in $(grep -l "// import lombok\." --include="*.java" -r $(pwd)/src); do
    echo "Uncommenting Lombok imports in $file"
    sed -i 's|// import lombok\.|import lombok.|g' "$file"
done

# Detect and install VS Code extension if VS Code is being used
if [ -d "$(pwd)/.vscode" ] || [ -f "$(pwd)/.vscode.code-workspace" ]; then
    echo "VS Code project detected. Checking for Lombok extension..."
    
    if command -v code >/dev/null 2>&1; then
        # Try multiple possible extension IDs - case sensitivity matters
        if code --list-extensions | grep -qi "lombok"; then
            echo "Lombok extension is already installed in VS Code."
        else
            echo "Attempting to install Lombok extension for VS Code..."
            # Try different extension IDs that might work
            if code --install-extension GabrielBB.vscode-lombok || code --install-extension gabrielbb.vscode-lombok; then
                echo "Lombok extension installed successfully!"
            else
                echo "Standard installation methods failed."
                echo "However, the extension might still be installed correctly."
                echo "Please check your VS Code Extensions panel (Ctrl+Shift+X) to verify."
                echo ""
                echo "If the extension is not installed, you can install it manually:"
                echo "1. Open VS Code."
                echo "2. Press Ctrl+P and type: ext install GabrielBB.vscode-lombok"
                echo "3. Press Enter to install the extension."
                echo "4. Or visit: https://marketplace.visualstudio.com/items?itemName=GabrielBB.vscode-lombok"
            fi
        fi
    else
        echo "VS Code CLI not found. To install the Lombok extension manually:"
        echo "1. Open VS Code."
        echo "2. Press Ctrl+P and type: ext install GabrielBB.vscode-lombok"
        echo "3. Press Enter to install the extension."
    fi
    
    # Create/update VS Code settings for Lombok
    mkdir -p "$(pwd)/.vscode"
    SETTINGS_FILE="$(pwd)/.vscode/settings.json"
    
    if [ -f "$SETTINGS_FILE" ]; then
        # Add Lombok settings if they don't exist
        if ! grep -q "java.jdt.ls.lombokSupport" "$SETTINGS_FILE"; then
            # If the file exists but doesn't have Lombok settings
            sed -i '$ s/}/,\n  "java.jdt.ls.lombokSupport": true\n}/' "$SETTINGS_FILE"
        fi
    else
        # Create settings file with Lombok support
        cat > "$SETTINGS_FILE" << EOF
{
  "java.jdt.ls.lombokSupport": true,
  "java.format.enabled": true,
  "java.configuration.updateBuildConfiguration": "automatic"
}
EOF
    fi
    echo "VS Code settings updated for Lombok support."
    
    # Add CMake configuration guidance
    echo ""
    echo "NOTE: If VS Code is asking you to select a CMakeLists.txt file, this is related to C++ components,"
    echo "not the Lombok Java configuration we just completed."
    echo ""
    echo "For C++ components with GPU support, select the appropriate CMakeLists.txt based on your target:"
    echo "- Main CMakeLists.txt: For general project configuration"
    echo "- cuda/CMakeLists.txt: If you're developing with NVIDIA CUDA support"
    echo "- rocm/CMakeLists.txt: If you're developing with AMD ROCm support"
    echo "- cpp/CMakeLists.txt: For standard C++ components without specific GPU optimizations"
    echo ""
    echo "You can change this selection later in VS Code by clicking on the CMake status bar item."
fi

echo "Lombok has been successfully added to the project using Maven!"
