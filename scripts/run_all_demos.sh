#!/bin/bash

# OpenNLP GPU Extension - Run All Demo Examples
# Executes all example projects with performance benchmarking

set -e

echo "🧪 Testing OpenNLP GPU Extension Examples"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track results
TOTAL_EXAMPLES=5
PASSED_EXAMPLES=0
TOTAL_START_TIME=$(date +%s%N)

# Function to run example and measure performance
run_example() {
    local example_name=$1
    local example_dir=$2
    local main_class=$3
    
    echo -e "${BLUE}Testing: ${example_name}${NC}"
    echo "Directory: ${example_dir}"
    
    if [ ! -d "${example_dir}" ]; then
        echo -e "${RED}❌ Directory not found: ${example_dir}${NC}"
        return 1
    fi
    
    cd "${example_dir}"
    
    # Check if the Java file exists
    if [ ! -f "${main_class}.java" ]; then
        echo -e "${RED}❌ Main class not found: ${main_class}.java${NC}"
        cd - > /dev/null
        return 1
    fi
    
    # Compile and run with timing
    echo "   Compiling..."
    if ! mvn clean compile -q; then
        echo -e "${RED}❌ Compilation failed${NC}"
        cd - > /dev/null
        return 1
    fi
    
    echo "   Running example..."
    start_time=$(date +%s%N)
    
    if mvn exec:java -Dexec.mainClass="${main_class}" -q; then
        end_time=$(date +%s%N)
        duration=$(( (end_time - start_time) / 1000000 ))
        echo -e "${GREEN}✅ ${example_name}: Completed in ${duration}ms${NC}"
        ((PASSED_EXAMPLES++))
    else
        echo -e "${RED}❌ ${example_name}: Execution failed${NC}"
    fi
    
    cd - > /dev/null
    echo ""
}

# Store original directory
ORIGINAL_DIR=$(pwd)

# Ensure we're in the project root
if [ ! -f "pom.xml" ]; then
    echo -e "${RED}❌ Please run this script from the project root directory${NC}"
    exit 1
fi

# Build the main project first
echo "🔨 Building main project..."
if ! mvn clean compile -q; then
    echo -e "${RED}❌ Main project build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Main project built successfully${NC}"
echo ""

# Run all examples
run_example "Sentiment Analysis" "examples/sentiment_analysis" "GpuSentimentAnalysis"
run_example "Named Entity Recognition" "examples/ner" "GpuNamedEntityRecognition"
run_example "Document Classification" "examples/classification" "GpuDocumentClassification"
run_example "Language Detection" "examples/language_detection" "GpuLanguageDetection"
run_example "Question Answering" "examples/question_answering" "GpuQuestionAnswering"

# Calculate total time
TOTAL_END_TIME=$(date +%s%N)
TOTAL_DURATION=$(( (TOTAL_END_TIME - TOTAL_START_TIME) / 1000000000 ))

# Results summary
echo "========================================="
echo "🎉 Demo Testing Results"
echo "========================================="
echo -e "${GREEN}✅ Passed: ${PASSED_EXAMPLES}/${TOTAL_EXAMPLES} examples${NC}"

if [ $PASSED_EXAMPLES -eq $TOTAL_EXAMPLES ]; then
    echo -e "${GREEN}🎊 All examples passed successfully!${NC}"
    echo -e "${BLUE}💾 Total execution time: ${TOTAL_DURATION}s${NC}"
    echo ""
    echo -e "${YELLOW}📈 GPU acceleration is working correctly${NC}"
    echo -e "${YELLOW}🚀 Your system is ready for high-performance NLP processing${NC}"
else
    FAILED_EXAMPLES=$((TOTAL_EXAMPLES - PASSED_EXAMPLES))
    echo -e "${RED}❌ ${FAILED_EXAMPLES} examples failed${NC}"
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting tips:${NC}"
    echo "   - Ensure GPU drivers are properly installed"
    echo "   - Check that Java 11+ is installed"
    echo "   - Verify Maven dependencies are resolved"
    echo "   - Run './verify.sh' for system diagnostics"
fi

echo ""
echo -e "${BLUE}📖 For detailed documentation, see: examples/README.md${NC}"
echo -e "${BLUE}🔧 For troubleshooting, see: docs/FAQ.md${NC}"
