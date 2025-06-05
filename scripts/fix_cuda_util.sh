#!/bin/bash

echo "Applying automatic fix for CudaUtil logging issues..."

CUDA_UTIL_FILE="$(pwd)/src/main/java/org/apache/opennlp/gpu/cuda/CudaUtil.java"

if [ -f "$CUDA_UTIL_FILE" ]; then
    echo "Found CudaUtil.java, applying fixes..."
    
    # Make a backup
    cp "$CUDA_UTIL_FILE" "${CUDA_UTIL_FILE}.bak"
    
    # Add manual SLF4J logger declaration at the top of the class
    if ! grep -q "private static final org.slf4j.Logger log" "$CUDA_UTIL_FILE"; then
        sed -i '/public class CudaUtil/a \
    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(CudaUtil.class);' "$CUDA_UTIL_FILE"
        echo "Added manual logger declaration to CudaUtil class"
    fi
    
    # Add required imports
    if ! grep -q "import org.slf4j.Logger;" "$CUDA_UTIL_FILE"; then
        sed -i '/package org.apache.opennlp.gpu.cuda;/a \
import org.slf4j.Logger;\
import org.slf4j.LoggerFactory;' "$CUDA_UTIL_FILE"
        echo "Added SLF4J imports to CudaUtil"
    fi
    
    echo "CudaUtil class has been fixed with manual SLF4J logger"
else
    echo "ERROR: Could not find CudaUtil.java at expected location"
fi
