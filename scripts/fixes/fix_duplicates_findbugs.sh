#!/bin/bash

# Exit on error
set -e

echo "===== Fixing duplicate constructors and FindBugs imports ====="

PROJECT_ROOT="$(pwd)"

# Fix duplicate constructors
fix_duplicate_constructors() {
    echo "Fixing duplicate constructors..."
    
    # Fix CpuMatrixOperation
    CPU_MATRIX_OP="$PROJECT_ROOT/src/main/java/org/apache/opennlp/gpu/compute/CpuMatrixOperation.java"
    if [ -f "$CPU_MATRIX_OP" ]; then
        echo "  Fixing CpuMatrixOperation..."
        # Create a temporary file
        TMP_FILE=$(mktemp)
        # Use awk to remove duplicate constructor but preserve the rest
        awk '
        BEGIN { skip=0; constructorCount=0; }
        /public CpuMatrixOperation\(ComputeProvider provider\)/ { 
            constructorCount++;
            if (constructorCount > 1) {
                skip=1;
                next;
            }
        }
        /^\s*}/ {
            if (skip) {
                skip=0;
                next;
            }
        }
        !skip { print; }
        ' "$CPU_MATRIX_OP" > "$TMP_FILE"
        mv "$TMP_FILE" "$CPU_MATRIX_OP"
    fi
    
    # Fix CpuFeatureExtractionOperation
    CPU_FEATURE_OP="$PROJECT_ROOT/src/main/java/org/apache/opennlp/gpu/compute/CpuFeatureExtractionOperation.java"
    if [ -f "$CPU_FEATURE_OP" ]; then
        echo "  Fixing CpuFeatureExtractionOperation..."
        # Create a temporary file
        TMP_FILE=$(mktemp)
        # Use awk to remove duplicate constructor but preserve the rest
        awk '
        BEGIN { skip=0; constructorCount=0; }
        /public CpuFeatureExtractionOperation\(ComputeProvider provider\)/ { 
            constructorCount++;
            if (constructorCount > 1) {
                skip=1;
                next;
            }
        }
        /^\s*}/ {
            if (skip) {
                skip=0;
                next;
            }
        }
        !skip { print; }
        ' "$CPU_FEATURE_OP" > "$TMP_FILE"
        mv "$TMP_FILE" "$CPU_FEATURE_OP"
    fi
    
    echo "Duplicate constructors fixed."
}

# Update FindBugs imports
fix_findbugs_imports() {
    echo "Updating FindBugs imports..."
    
    # Find all Java files with edu.umd.cs.findbugs.annotations imports
    find "$PROJECT_ROOT/src" -name "*.java" -type f -exec grep -l "edu.umd.cs.findbugs.annotations" {} \; | while read -r file; do
        echo "  Updating imports in $(basename "$file")"
        
        # Replace edu.umd.cs.findbugs.annotations with com.google.code.findbugs.annotations
        sed -i 's/edu\.umd\.cs\.findbugs\.annotations/com.google.code.findbugs.annotations/g' "$file"
        
        # If there are any @SuppressFBWarnings annotations, replace them with @SuppressWarnings
        sed -i 's/@SuppressFBWarnings/@SuppressWarnings/g' "$file"
    done
    
    echo "FindBugs imports updated."
}

# Execute fixes
fix_duplicate_constructors
fix_findbugs_imports

echo "===== Duplicate constructors and FindBugs imports fixed ====="
echo "Try compiling the project again with 'mvn clean compile'"
chmod +x "$0"
