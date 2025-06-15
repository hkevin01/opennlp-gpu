#!/bin/bash

# Function to add manual logging to any Java class
fix_class_logging() {
    local class_file=$1
    
    if [ -f "$class_file" ]; then
        local class_name=$(basename "$class_file" .java)
        echo "Fixing logging in $class_name..."
        
        # Make a backup
        cp "$class_file" "${class_file}.bak"
        
        # Add manual SLF4J logger declaration at the top of the class
        if ! grep -q "private static final org.slf4j.Logger log" "$class_file"; then
            sed -i "/public class $class_name/a \\
    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger($class_name.class);" "$class_file"
            echo "Added manual logger declaration to $class_name class"
        fi
        
        # Add required imports
        if ! grep -q "import org.slf4j.Logger;" "$class_file"; then
            sed -i "/package /a \\
import org.slf4j.Logger;\\
import org.slf4j.LoggerFactory;" "$class_file"
            echo "Added SLF4J imports to $class_name"
        fi
        
        echo "$class_name class has been fixed with manual SLF4J logger"
    else
        echo "ERROR: Could not find $class_file"
    fi
}

# Fix specific problematic classes
fix_rocm_util() {
    fix_class_logging "$(pwd)/src/main/java/org/apache/opennlp/gpu/rocm/RocmUtil.java"
}

fix_native_library_loader() {
    fix_class_logging "$(pwd)/src/main/java/org/apache/opennlp/gpu/util/NativeLibraryLoader.java"
}

# Run fixes for all problematic classes
fix_rocm_util
fix_native_library_loader

# Fix any other class with log symbol errors
echo "Scanning for any other classes with log symbol errors..."
for java_file in $(find "$(pwd)/src" -name "*.java" | xargs grep -l "log\." 2>/dev/null); do
    # Skip classes that already have proper logging
    if ! grep -q "private static final org.slf4j.Logger log" "$java_file" && ! grep -q "@Slf4j" "$java_file"; then
        fix_class_logging "$java_file"
    fi
done
