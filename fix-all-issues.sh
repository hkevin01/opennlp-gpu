#!/bin/bash
# Comprehensive fix script for all known compilation issues

echo "Starting comprehensive fixes..."

# Fix RocmFeatureExtractionOperation.java
if [ -f "src/main/java/org/apache/opennlp/gpu/compute/RocmFeatureExtractionOperation.java" ]; then
    echo "Fixing RocmFeatureExtractionOperation.java..."
    # Ensure proper class structure
    if ! grep -q "^}[[:space:]]*$" src/main/java/org/apache/opennlp/gpu/compute/RocmFeatureExtractionOperation.java; then
        echo "}" >> src/main/java/org/apache/opennlp/gpu/compute/RocmFeatureExtractionOperation.java
    fi
fi

# Fix OpenClMatrixOperation.java
if [ -f "src/main/java/org/apache/opennlp/gpu/compute/OpenClMatrixOperation.java" ]; then
    echo "Fixing OpenClMatrixOperation.java..."
    # Ensure proper class structure
    if ! grep -q "^}[[:space:]]*$" src/main/java/org/apache/opennlp/gpu/compute/OpenClMatrixOperation.java; then
        echo "}" >> src/main/java/org/apache/opennlp/gpu/compute/OpenClMatrixOperation.java
    fi
fi

# Additional fixes for other common issues
for java_file in $(find src/main/java -name "*.java"); do
    if [ -f "$java_file" ]; then
        # Remove any orphaned method signatures or incomplete code blocks
        sed -i '/public static getFinal()/d' "$java_file"
        sed -i '/public final getComputeProvider()/d' "$java_file"
        sed -i '/public native getLong()/d' "$java_file"
        sed -i '/public native getVoid()/d' "$java_file"
        sed -i '/public native getInt()/d' "$java_file"
    fi
done

echo "Comprehensive fixes completed."
