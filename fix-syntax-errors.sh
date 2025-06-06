#!/bin/bash
# Fix syntax errors - missing closing braces

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}Fixing syntax errors in Java files...${NC}"

# Fix RocmFeatureExtractionOperation.java
if [ -f "src/main/java/org/apache/opennlp/gpu/compute/RocmFeatureExtractionOperation.java" ]; then
    echo -e "${YELLOW}Fixing RocmFeatureExtractionOperation.java...${NC}"
    
    # Count opening and closing braces
    opening_braces=$(grep -o '{' src/main/java/org/apache/opennlp/gpu/compute/RocmFeatureExtractionOperation.java | wc -l)
    closing_braces=$(grep -o '}' src/main/java/org/apache/opennlp/gpu/compute/RocmFeatureExtractionOperation.java | wc -l)
    
    echo "Opening braces: $opening_braces, Closing braces: $closing_braces"
    
    # Add missing closing braces
    missing_braces=$((opening_braces - closing_braces))
    if [ $missing_braces -gt 0 ]; then
        echo -e "${YELLOW}Adding $missing_braces missing closing brace(s)...${NC}"
        for ((i=1; i<=missing_braces; i++)); do
            echo "}" >> src/main/java/org/apache/opennlp/gpu/compute/RocmFeatureExtractionOperation.java
        done
    fi
    
    echo -e "${GREEN}Fixed RocmFeatureExtractionOperation.java${NC}"
fi

# Fix OpenClMatrixOperation.java
if [ -f "src/main/java/org/apache/opennlp/gpu/compute/OpenClMatrixOperation.java" ]; then
    echo -e "${YELLOW}Fixing OpenClMatrixOperation.java...${NC}"
    
    # Count opening and closing braces
    opening_braces=$(grep -o '{' src/main/java/org/apache/opennlp/gpu/compute/OpenClMatrixOperation.java | wc -l)
    closing_braces=$(grep -o '}' src/main/java/org/apache/opennlp/gpu/compute/OpenClMatrixOperation.java | wc -l)
    
    echo "Opening braces: $opening_braces, Closing braces: $closing_braces"
    
    # Add missing closing braces
    missing_braces=$((opening_braces - closing_braces))
    if [ $missing_braces -gt 0 ]; then
        echo -e "${YELLOW}Adding $missing_braces missing closing brace(s)...${NC}"
        for ((i=1; i<=missing_braces; i++)); do
            echo "}" >> src/main/java/org/apache/opennlp/gpu/compute/OpenClMatrixOperation.java
        done
    fi
    
    echo -e "${GREEN}Fixed OpenClMatrixOperation.java${NC}"
fi

echo -e "${GREEN}Syntax errors fixed!${NC}"
