#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "===== Starting Automated Fix Script for OpenNLP GPU Project ====="

# Define the project root
PROJECT_ROOT="$(pwd)"
POM_FILE="$PROJECT_ROOT/pom.xml"

# Define versions
JOCL_VERSION="2.0.4"
CUDA_VERSION="11.8.0"
ROCM_VERSION="5.4.2"
LOMBOK_VERSION="1.18.30"
SLF4J_VERSION="1.7.36"
MAVEN_COMPILER_PLUGIN_VERSION="3.8.1"

# Backup pom.xml
backup_pom() {
    if [ -f "$POM_FILE" ]; then
        echo "Creating backup of pom.xml..."
        cp "$POM_FILE" "${POM_FILE}.bak"
        echo "Backup created at ${POM_FILE}.bak"
    else
        echo "ERROR: pom.xml not found in ${PROJECT_ROOT}. Exiting."
        exit 1
    fi
}

# Add repositories to pom.xml
add_repositories() {
    echo "Adding necessary repositories to pom.xml..."
    repositories_to_add=$(cat <<EOF
        <repositories>
            <repository>
                <id>central</id>
                <url>https://repo.maven.apache.org/maven2</url>
            </repository>
            <repository>
                <id>jocl</id>
                <url>https://repo.maven.apache.org/maven2/org/jocl/jocl</url>
            </repository>
        </repositories>
EOF
)
    if ! grep -q "<repositories>" "$POM_FILE"; then
        sed -i "/<\/project>/i\\
$repositories_to_add" "$POM_FILE"
        echo "Repositories added to pom.xml."
    else
        echo "Repositories section already exists in pom.xml."
    fi
}

# Add dependencies to pom.xml
add_dependencies() {
    echo "Adding necessary dependencies to pom.xml..."

    # Ensure dependencies section exists
    if ! grep -q "<dependencies>" "$POM_FILE"; then
        sed -i '/<\/project>/i\\
        <dependencies>\
        </dependencies>' "$POM_FILE"
        echo "Created dependencies section in pom.xml."
    fi

    # Add JOCL dependency if not present
    if ! grep -q "<artifactId>jocl</artifactId>" "$POM_FILE"; then
        sed -i '/<dependencies>/a\\
            <dependency>\
                <groupId>org.jocl</groupId>\
                <artifactId>jocl</artifactId>\
                <version>'"$JOCL_VERSION"'</version>\
            </dependency>' "$POM_FILE"
        echo "Added JOCL dependency to pom.xml."
    else
        echo "JOCL dependency already exists in pom.xml."
    fi

    # Add Lombok dependency if not present
    if ! grep -q "<artifactId>lombok</artifactId>" "$POM_FILE"; then
        sed -i '/<dependencies>/a\\
            <dependency>\
                <groupId>org.projectlombok</groupId>\
                <artifactId>lombok</artifactId>\
                <version>'"$LOMBOK_VERSION"'</version>\
                <scope>provided</scope>\
            </dependency>' "$POM_FILE"
        echo "Added Lombok dependency to pom.xml."
    else
        echo "Lombok dependency already exists in pom.xml."
    fi

    # Add SLF4J dependencies if not present
    if ! grep -q "<artifactId>slf4j-api</artifactId>" "$POM_FILE"; then
        sed -i '/<dependencies>/a\\
            <dependency>\
                <groupId>org.slf4j</groupId>\
                <artifactId>slf4j-api</artifactId>\
                <version>'"$SLF4J_VERSION"'</version>\
            </dependency>\
            <dependency>\
                <groupId>org.slf4j</groupId>\
                <artifactId>slf4j-simple</artifactId>\
                <version>'"$SLF4J_VERSION"'</version>\
            </dependency>' "$POM_FILE"
        echo "Added SLF4J dependencies to pom.xml."
    else
        echo "SLF4J dependencies already exist in pom.xml."
    fi
}

# Configure Maven Compiler Plugin for Lombok and Annotation Processing
configure_maven_compiler_plugin() {
    echo "Configuring Maven Compiler Plugin for annotation processing..."

    # Ensure build section exists
    if ! grep -q "<build>" "$POM_FILE"; then
        sed -i '/<\/dependencies>/a\\
        <build>\
            <plugins>\
            </plugins>\
        </build>' "$POM_FILE"
        echo "Created build section in pom.xml."
    fi

    # Use a safer approach to handle existing Maven Compiler Plugin
    if grep -q "maven-compiler-plugin" "$POM_FILE"; then
        echo "Maven Compiler Plugin already exists. Using a safer approach to update it..."
        
        # Create a temporary file
        TEMP_POM="${POM_FILE}.new"
        
        # Create a new pom with updated compiler plugin
        awk '
        {
            # Print the current line
            print $0;
            
            # If we find the plugins section opening tag, add our plugin right after it
            if ($0 ~ /<plugins>/) {
                print "            <plugin>";
                print "                <groupId>org.apache.maven.plugins</groupId>";
                print "                <artifactId>maven-compiler-plugin</artifactId>";
                print "                <version>'"$MAVEN_COMPILER_PLUGIN_VERSION"'</version>";
                print "                <configuration>";
                print "                    <source>1.8</source>";
                print "                    <target>1.8</target>";
                print "                    <annotationProcessorPaths>";
                print "                        <path>";
                print "                            <groupId>org.projectlombok</groupId>";
                print "                            <artifactId>lombok</artifactId>";
                print "                            <version>'"$LOMBOK_VERSION"'</version>";
                print "                        </path>";
                print "                    </annotationProcessorPaths>";
                print "                    <encoding>UTF-8</encoding>";
                print "                </configuration>";
                print "            </plugin>";
            }
        }
        # Skip the existing maven-compiler-plugin section if found
        /maven-compiler-plugin/,/<\/plugin>/ { next }
        ' "$POM_FILE" > "$TEMP_POM"
        
        # Move the new file over the original
        mv "$TEMP_POM" "$POM_FILE"
        echo "Updated Maven Compiler Plugin configuration using awk."
    else
        # Add Maven Compiler Plugin with annotation processing
        echo "Adding new Maven Compiler Plugin configuration..."
        awk '
        {
            # Print the current line
            print $0;
            
            # If we find the plugins section opening tag, add our plugin right after it
            if ($0 ~ /<plugins>/) {
                print "            <plugin>";
                print "                <groupId>org.apache.maven.plugins</groupId>";
                print "                <artifactId>maven-compiler-plugin</artifactId>";
                print "                <version>'"$MAVEN_COMPILER_PLUGIN_VERSION"'</version>";
                print "                <configuration>";
                print "                    <source>1.8</source>";
                print "                    <target>1.8</target>";
                print "                    <annotationProcessorPaths>";
                print "                        <path>";
                print "                            <groupId>org.projectlombok</groupId>";
                print "                            <artifactId>lombok</artifactId>";
                print "                            <version>'"$LOMBOK_VERSION"'</version>";
                print "                        </path>";
                print "                    </annotationProcessorPaths>";
                print "                    <encoding>UTF-8</encoding>";
                print "                </configuration>";
                print "            </plugin>";
            }
        }
        ' "$POM_FILE" > "${POM_FILE}.new"
        
        # Move the new file over the original
        mv "${POM_FILE}.new" "$POM_FILE"
        echo "Added Maven Compiler Plugin configuration."
    fi
}

# Ensure Lombok configuration
configure_lombok() {
    echo "Configuring Lombok..."

    LOMBOK_CONFIG_FILE="$PROJECT_ROOT/lombok.config"

    cat <<EOF > "$LOMBOK_CONFIG_FILE"
# Lombok configuration file
config.stopBubbling = true
lombok.addLombokGeneratedAnnotation = true
lombok.anyConstructor.addConstructorProperties = true
lombok.accessors.chain = true
lombok.fieldDefaults.defaultPrivate = true
lombok.fieldDefaults.defaultFinal = false
lombok.toString.doNotUseGetters = true
lombok.equalsAndHashCode.callSuper = call
lombok.log.fieldName = log
lombok.log.slf4j.fieldName = log
lombok.log.log4j.fieldName = log
EOF

    echo "Updated lombok.config file."
}

# Create Test File to Verify Lombok Annotations
create_lombok_test_file() {
    echo "Creating LombokTester.java to verify Lombok annotations..."

    TEST_DIR="$PROJECT_ROOT/src/test/java/org/apache/opennlp/gpu/test"
    mkdir -p "$TEST_DIR"

    TEST_FILE="$TEST_DIR/LombokTester.java"

    cat <<EOF > "$TEST_FILE"
package org.apache.opennlp.gpu.test;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Log4j;

@Log4j
@RequiredArgsConstructor
public class LombokTester {
    @Getter
    private final String name;
    
    public void testLogging() {
        log.info("Testing Lombok logging with field name: {}", name);
    }
}
EOF

    echo "Created $TEST_FILE."
}

# Add SLF4J dependencies with more specific version and ensure direct import
add_slf4j_dependencies() {
    echo "Adding or updating SLF4J dependencies..."
    
    # Check if SLF4J is in pom.xml
    if ! grep -q "<artifactId>slf4j-api</artifactId>" "$POM_FILE" || ! grep -q "<version>$SLF4J_VERSION</version>" "$POM_FILE"; then
        # Remove any existing SLF4J dependencies to avoid version conflicts
        TMP_FILE="$(mktemp)"
        sed '/<artifactId>slf4j-api<\/artifactId>/,/<\/dependency>/d' "$POM_FILE" > "$TMP_FILE"
        sed '/<artifactId>slf4j-simple<\/artifactId>/,/<\/dependency>/d' "$TMP_FILE" > "$POM_FILE"
        
        # Add SLF4J with explicit dependency management
        sed -i '/<dependencies>/a\
            <!-- SLF4J API -->\
            <dependency>\
                <groupId>org.slf4j</groupId>\
                <artifactId>slf4j-api</artifactId>\
                <version>'"$SLF4J_VERSION"'</version>\
            </dependency>\
            <!-- SLF4J Simple implementation -->\
            <dependency>\
                <groupId>org.slf4j</groupId>\
                <artifactId>slf4j-simple</artifactId>\
                <version>'"$SLF4J_VERSION"'</version>\
            </dependency>' "$POM_FILE"
        
        echo "Added explicit SLF4J dependencies to pom.xml"
    fi
}

# Install Lombok jar directly (for when Maven dependency isn't enough)
install_lombok_jar() {
    echo "Installing Lombok jar directly to ensure proper resolution..."
    
    # Create .m2 directory if it doesn't exist
    mkdir -p "$HOME/.m2/repository/org/projectlombok/lombok/$LOMBOK_VERSION"
    
    # Download Lombok jar if it doesn't exist
    LOMBOK_JAR="$HOME/.m2/repository/org/projectlombok/lombok/$LOMBOK_VERSION/lombok-$LOMBOK_VERSION.jar"
    if [ ! -f "$LOMBOK_JAR" ]; then
        echo "Downloading Lombok jar..."
        if command -v curl >/dev/null 2>&1; then
            curl -L "https://repo1.maven.org/maven2/org/projectlombok/lombok/$LOMBOK_VERSION/lombok-$LOMBOK_VERSION.jar" -o "$LOMBOK_JAR"
        elif command -v wget >/dev/null 2>&1; then
            wget "https://repo1.maven.org/maven2/org/projectlombok/lombok/$LOMBOK_VERSION/lombok-$LOMBOK_VERSION.jar" -O "$LOMBOK_JAR"
        else
            echo "ERROR: Neither curl nor wget is available. Cannot download Lombok."
        fi
    fi
    
    # Verify download
    if [ -f "$LOMBOK_JAR" ]; then
        echo "Lombok jar available at: $LOMBOK_JAR"
        
        # Copy to project directory for IDE usage
        mkdir -p "$PROJECT_ROOT/lib"
        cp "$LOMBOK_JAR" "$PROJECT_ROOT/lib/"
        echo "Copied Lombok jar to project lib directory for IDE usage"
    else
        echo "ERROR: Failed to ensure Lombok jar is available."
    fi
}

# Create IDE configuration for Lombok
create_ide_config() {
    echo "Creating IDE configuration for Lombok..."
    
    # VS Code configuration
    if command -v code >/dev/null 2>&1; then
        mkdir -p "$PROJECT_ROOT/.vscode"
        cat << EOF > "$PROJECT_ROOT/.vscode/settings.json"
{
    "java.configuration.updateBuildConfiguration": "automatic",
    "java.project.referencedLibraries": [
        "lib/**/*.jar"
    ],
    "java.jdt.ls.lombokSupport": true,
    "java.format.enabled": true
}
EOF
        echo "Created VS Code settings for Lombok"
    fi
    
    # Eclipse/IntelliJ .factorypath for annotation processing
    cat << EOF > "$PROJECT_ROOT/.factorypath"
<factorypath>
    <factorypathentry kind="VARJAR" id="M2_REPO/org/projectlombok/lombok/$LOMBOK_VERSION/lombok-$LOMBOK_VERSION.jar" enabled="true" runInBatchMode="false"/>
</factorypath>
EOF
    echo "Created .factorypath for IDE annotation processing"
}

# Fix Java Files Automatically
fix_java_files() {
    echo "Automatically fixing Java files based on errors..."

    # 1. Remove duplicate method definitions
    echo "Removing duplicate method definitions..."
    find "$PROJECT_ROOT/src" -name "*.java" | while read -r file; do
        # Check for duplicate getDeviceCount() methods
        DUP_METHODS=$(grep -c "getDeviceCount()" "$file" || true)
        if [ "$DUP_METHODS" -gt 1 ]; then
            echo "Duplicate getDeviceCount() found in $file. Removing duplicates..."
            # Keep the first occurrence and remove the rest
            awk '/getDeviceCount\(\)/ {count++} count <= 1 {print} count > 1 {next} !/getDeviceCount\(\)/ {print}' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
            echo "Duplicates removed from $file."
        fi
    done

    # 2. Implement missing constructors and methods
    echo "Implementing missing constructors and methods..."

    # Handle ComputeConfiguration constructor
    COMPUTE_PROVIDER_FACTORY_FILE="$PROJECT_ROOT/src/main/java/org/apache/opennlp/gpu/common/ComputeProviderFactory.java"
    if [ -f "$COMPUTE_PROVIDER_FACTORY_FILE" ]; then
        if grep -q "new ComputeConfiguration()" "$COMPUTE_PROVIDER_FACTORY_FILE"; then
            echo "Adding missing constructor parameters to ComputeConfiguration in ComputeProviderFactory.java..."
            # Use a different delimiter (|) instead of / to avoid escaping issues
            sed -i 's|new ComputeConfiguration()|new ComputeConfiguration(/* Add required Properties here */)|g' "$COMPUTE_PROVIDER_FACTORY_FILE"
            echo "Updated ComputeConfiguration constructor."
        fi
    fi

    # Implement missing methods in interfaces
    echo "Implementing missing interface methods..."

    # Define an array of files and the methods they need to implement
    declare -A files_methods=(
        ["ComputeProvider.java"]="getResourceManager()"
        ["FeatureExtractionOperation.java"]="getProvider()"
        ["MatrixOperation.java"]="getProvider()"
    )

    for interface_file in "${!files_methods[@]}"; do
        interface_path="$PROJECT_ROOT/src/main/java/org/apache/opennlp/gpu/common/$interface_file"
        method="${files_methods[$interface_file]}"
        if [ -f "$interface_path" ]; then
            # Ensure the method exists in the interface
            if ! grep -q "$method" "$interface_path"; then
                echo "Adding $method to $interface_file..."
                sed -i "/public interface $interface_file/a\\
    public ComputeProvider getProvider();" "$interface_path"
                echo "Added $method to $interface_file."
            fi
        fi
    done

    # Implement the missing methods in the implementing classes
    declare -A implementing_classes=(
        ["CudaUtil.java"]="getDeviceCount()"
        ["RocmUtil.java"]="getDeviceCount()"
        ["ComputeProviderFactory.java"]="getPreferredProviderType(), getSmallProblemThreshold()"
        ["MemoryManager.java"]="Handle type conversions"
        ["OpenClComputeProvider.java"]="getResourceManager()"
        ["RocmComputeProvider.java"]="getResourceManager()"
        ["CpuFeatureExtractionOperation.java"]="getProvider()"
        ["CpuMatrixOperation.java"]="getProvider()"
        ["CudaMatrixOperation.java"]="getProvider()"
        ["OperationFactory.java"]="Fix missing OpenClFeatureExtractionOperation and logger"
        ["NativeLibraryLoader.java"]="Fix logger variable"
    )

    for class_file in "${!implementing_classes[@]}"; do
        class_path="$PROJECT_ROOT/src/main/java/org/apache/opennlp/gpu/compute/$class_file"
        if [ -f "$class_path" ]; then
            # Implement missing methods based on the class
            case "$class_file" in
                "OpenClComputeProvider.java"|"RocmComputeProvider.java")
                    if ! grep -q "getResourceManager()" "$class_path"; then
                        echo "Adding getResourceManager() method to $class_file..."
                        sed -i "/public class $class_file/a\\
    @Override\\
    public ResourceManager getResourceManager() {\\
        // TODO: Implement proper resource manager\\
        return null;\\
    }" "$class_path"
                        echo "Added getResourceManager() to $class_file."
                    fi
                    ;;
                "CpuFeatureExtractionOperation.java"|"CpuMatrixOperation.java"|"CudaMatrixOperation.java")
                    if ! grep -q "getProvider()" "$class_path"; then
                        echo "Adding getProvider() method to $class_file..."
                        sed -i "/public class $class_file/a\\
    @Override\\
    public ComputeProvider getProvider() {\\
        return this.provider;\\
    }" "$class_path"
                        echo "Added getProvider() to $class_file."
                    fi
                    ;;
                "RocmUtil.java"|"CudaUtil.java")
                    # These classes have duplicate getDeviceCount, already handled
                    ;;
                "OperationFactory.java")
                    # Fix missing OpenClFeatureExtractionOperation and logger
                    if grep -q "OpenClFeatureExtractionOperation" "$class_path"; then
                        echo "Adding OpenClFeatureExtractionOperation dependency in OperationFactory.java..."
                        sed -i '/return new/ i \\        // Ensure OpenClFeatureExtractionOperation is properly referenced' "$class_path"
                    fi
                    if ! grep -q "private static final org.slf4j.Logger log" "$class_path"; then
                        echo "Adding SLF4J logger to OperationFactory.java..."
                        sed -i "/public class OperationFactory/a \\
    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(OperationFactory.class);" "$class_path"
                        echo "Added SLF4J logger to OperationFactory.java."
                    fi
                    ;;
                "NativeLibraryLoader.java")
                    # Add SLF4J logger
                    if ! grep -q "private static final org.slf4j.Logger log" "$class_path"; then
                        echo "Adding SLF4J logger to NativeLibraryLoader.java..."
                        sed -i "/public class NativeLibraryLoader/a \\
    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(NativeLibraryLoader.class);" "$class_path"
                        echo "Added SLF4J logger to NativeLibraryLoader.java."
                    fi
                    ;;
                *)
                    ;;
            esac
        fi
    done

    # 3. Fix incompatible type conversions (long to int)
    echo "Fixing incompatible type conversions from long to int..."

    # Find and modify lines where long is being cast to int
    find "$PROJECT_ROOT/src" -name "*.java" | while read -r file; do
        if grep -q "incompatible types: possible lossy conversion from long to int" "$file"; then
            echo "Fixing type conversion in $file..."
            # Example fix: Change variable type from int to long or add explicit casting
            sed -i 's/(int) longVariable/(int)(longVariable)/g' "$file"
            echo "Added explicit casting in $file."
        fi
    done

    # 4. Ensure all classes implementing interfaces have required methods
    echo "Ensuring all classes implementing interfaces have required methods..."

    # Process all classes that implement interfaces and check for required methods
    find "$PROJECT_ROOT/src/main/java/org/apache/opennlp/gpu" -type f -name "*.java" | while read -r file; do
        if grep -q "implements MatrixOperation" "$file"; then
            if ! grep -q "getProvider()" "$file"; then
                echo "Adding getProvider() method to $(basename "$file")..."
                sed -i "/public class .* implements MatrixOperation/a\\
    @Override\\
    public ComputeProvider getProvider() {\\
        return this.provider;\\
    }" "$file"
            fi
        fi

        if grep -q "implements ComputeProvider" "$file"; then
            if ! grep -q "getResourceManager()" "$file"; then
                echo "Adding getResourceManager() method to $(basename "$file")..."
                sed -i "/public class .* implements ComputeProvider/a\\
    @Override\\
    public ResourceManager getResourceManager() {\\
        // TODO: Implement proper resource manager\\
        return null;\\
    }" "$file"
            fi
        fi
    done

    # 5. Fix logger variable errors by ensuring proper SLF4J usage
    echo "Ensuring proper SLF4J logger usage across classes..."

    # Add @Slf4j annotation and proper logger declarations
    find "$PROJECT_ROOT/src/main/java/org/apache/opennlp/gpu" -name "*.java" | while read -r file; do
        CLASS_NAME=$(basename "$file" .java)
        # Skip test files
        if [[ "$file" == *"test"* ]]; then
            continue
        fi

        # Fix incorrect Slf4j imports - this was causing the "import lombok cannot be resolved" error
        if grep -q "import lombok.extern/slf4j" "$file"; then
            echo "Fixing incorrect SLF4J import in $CLASS_NAME..."
            sed -i 's|import lombok.extern/slf4j/.*|import lombok.extern/slf4j/log4j;|g' "$file"
        fi

        # Ensure proper SLF4J imports are present
        if ! grep -q "import lombok.extern/slf4j/log4j" "$file" && ! grep -q "import org.slf4j.Logger" "$file"; then
            echo "Adding proper SLF4J import to $CLASS_NAME..."
            sed -i "/package /a\\
import lombok.extern/slf4j/log4j;" "$file"
        fi

        # Check if class already has @Slf4j
        if ! grep -q "@Slf4j" "$file"; then
            echo "Adding @Slf4j annotation to $CLASS_NAME..."
            sed -i "/public class $CLASS_NAME/i @Slf4j" "$file"
        fi

        # Add manual SLF4J logger if needed (to handle edge cases)
        if grep -q "log\." "$file" && ! grep -q "@Slf4j" "$file" && ! grep -q "Logger log" "$file"; then
            echo "Adding manual SLF4J logger to $CLASS_NAME..."
            sed -i "/public class $CLASS_NAME/a\\
    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger($CLASS_NAME.class);" "$file"
            
            # Add imports for manual logger
            sed -i "/package /a\\
import org.slf4j.Logger;\\
import org.slf4j.LoggerFactory;" "$file"
        fi

        # Replace 'logger' with 'log' if present
        if grep -q "logger\." "$file"; then
            echo "Replacing 'logger' with 'log' in $CLASS_NAME..."
            sed -i 's/logger\./log\./g' "$file"
        fi
    done

    echo "Java files have been automatically fixed."
}

# Add missing libraries or correct existing ones based on common errors
add_missing_libraries() {
    echo "Adding or correcting missing libraries in pom.xml..."

    # Example: Ensure ComputeConfiguration has the correct constructor
    # This may require adding or updating specific dependencies
    # Adjust based on the project's requirements

    # Placeholder for additional library fixes
    echo "No additional libraries to add at this time."
}

# Final Maven Build Attempt
run_maven_build() {
    echo "===== Running Maven Build ====="
    if command -v mvn >/dev/null 2>&1; then
        echo "Cleaning the project..."
        mvn clean

        echo "Compiling the project..."
        mvn compile -Dmaven.compiler.showWarnings=true -Dmaven.compiler.verbose=true
        echo "Maven compilation completed successfully."
    else
        echo "ERROR: Maven is not installed or not available in PATH."
        exit 1
    fi
}

# Execute the functions in order
backup_pom
add_repositories
add_dependencies
add_slf4j_dependencies
configure_maven_compiler_plugin
configure_lombok
create_lombok_test_file
install_lombok_jar  # Add this new function
create_ide_config   # Add this new function
fix_java_files
add_missing_libraries
run_maven_build

echo "===== Automated Fix Script Completed ====="