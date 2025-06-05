#!/bin/bash

echo "Applying comprehensive fixes for Lombok and logging issues..."

# Process specific files with known issues
process_file() {
    local file=$1
    local class_name=$(basename "$file" .java)
    
    echo "Processing $file"
    
    # Add Slf4j annotation if missing but uses logging
    if grep -q "log\." "$file" && ! grep -q "@Slf4j" "$file"; then
        echo "  Adding @Slf4j annotation to $class_name"
        sed -i "/^public class $class_name/i @Slf4j" "$file"
        sed -i "/^package/a\import lombok.extern.slf4j.log4j2.Slf4j;" "$file"
    fi
    
    # Add Getter annotation if missing but implements interface
    if (grep -q "implements MatrixOperation" "$file" || grep -q "implements FeatureExtractionOperation" "$file") && ! grep -q "@Getter" "$file" && grep -q "provider" "$file"; then
        echo "  Adding @Getter for provider in $class_name"
        sed -i "/private.*ComputeProvider provider/i\    @Getter" "$file"
        
        # Add import if missing
        if ! grep -q "import lombok.Getter;" "$file"; then
            sed -i "/^package/a\import lombok.Getter;" "$file"
        fi
    fi
    
    # Fix missing implementation of getProvider method
    if (grep -q "implements MatrixOperation" "$file" || grep -q "implements FeatureExtractionOperation" "$file") && ! grep -q "getProvider()" "$file"; then
        echo "  Adding getProvider() method to $class_name"
        sed -i "/public class $class_name/a\\
    @Override\\
    public ComputeProvider getProvider() {\\
        return this.provider;\\
    }" "$file"
    fi
    
    # Fix missing implementation of getResourceManager method
    if grep -q "implements ComputeProvider" "$file" && ! grep -q "getResourceManager()" "$file"; then
        local resource_manager_class="${class_name}ResourceManager"
        echo "  Adding getResourceManager() method to $class_name"
        if grep -q "$resource_manager_class" "$file"; then
            # If there's a resource manager inner class
            sed -i "/public class $class_name/a\\
    @Override\\
    public ResourceManager getResourceManager() {\\
        return new ${resource_manager_class}();\\
    }" "$file"
        else
            # Add a basic implementation
            sed -i "/public class $class_name/a\\
    @Override\\
    public ResourceManager getResourceManager() {\\
        // TODO: Implement proper resource manager\\
        return null;\\
    }" "$file"
        fi
        
        # Add import for ResourceManager if missing
        if ! grep -q "import org.apache.opennlp.gpu.common.ResourceManager;" "$file"; then
            sed -i "/^package/a\import org.apache.opennlp.gpu.common.ResourceManager;" "$file"
        fi
    fi
}

# Fix CudaUtil class (which has special logging needs)
fix_cuda_util() {
    local file="$(pwd)/src/main/java/org/apache/opennlp/gpu/cuda/CudaUtil.java"
    if [ -f "$file" ]; then
        echo "Fixing CudaUtil logging..."
        
        # Add Slf4j annotation if missing
        if ! grep -q "@Slf4j" "$file"; then
            sed -i "/^public class CudaUtil/i @Slf4j" "$file"
            sed -i "/^package/a\import lombok.extern.slf4j.log4j2.Slf4j;" "$file"
        fi
    fi
}

# Process all classes mentioned in error logs
process_specific_files() {
    # Process RocmMatrixOperation
    if [ -f "$(pwd)/src/main/java/org/apache/opennlp/gpu/compute/RocmMatrixOperation.java" ]; then
        process_file "$(pwd)/src/main/java/org/apache/opennlp/gpu/compute/RocmMatrixOperation.java"
    fi
    
    # Process CudaUtil
    fix_cuda_util
    
    # Process other problem files
    for file in $(find "$(pwd)/src" -name "*.java" | xargs grep -l "cannot find symbol.*log" 2>/dev/null); do
        process_file "$file"
    done
    
    # Process files with missing method implementations
    for file in $(find "$(pwd)/src" -name "*.java" | xargs grep -l "is not abstract and does not override" 2>/dev/null); do
        process_file "$file"
    done
}

process_specific_files

echo "Lombok and logging fixes applied!"
