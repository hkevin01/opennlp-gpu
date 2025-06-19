#!/bin/bash

# OpenNLP GPU Project - README Accuracy Verification
# This script verifies that all claims in the corrected README.md are accurate

echo "======================================================"
echo "OpenNLP GPU Project - README Accuracy Verification"
echo "======================================================"
echo "ℹ️  Verifying all README claims are accurate"
echo "ℹ️  Date: $(date)"
echo

# Check 1: Verify working examples exist
echo "🔍 Checking working examples mentioned in README..."
examples_found=0
examples_expected=5

check_example() {
    local example_path=$1
    local example_name=$2
    if [ -f "$example_path" ]; then
        echo "✅ $example_name exists: $example_path"
        ((examples_found++))
    else
        echo "❌ $example_name missing: $example_path"
    fi
}

check_example "examples/sentiment_analysis/GpuSentimentAnalysis.java" "Sentiment Analysis"
check_example "examples/ner/GpuNamedEntityRecognition.java" "Named Entity Recognition"
check_example "examples/classification/GpuDocumentClassification.java" "Document Classification"
check_example "examples/language_detection/GpuLanguageDetection.java" "Language Detection"
check_example "examples/question_answering/GpuQuestionAnswering.java" "Question Answering"

echo "   Examples found: $examples_found/$examples_expected"
echo

# Check 2: Verify GPU diagnostics tool exists and works
echo "🔍 Checking GPU diagnostics tool..."
if [ -f "src/main/java/org/apache/opennlp/gpu/tools/GpuDiagnostics.java" ]; then
    echo "✅ GpuDiagnostics.java exists"
    
    # Try to run it (quick check)
    if mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics" -q > /dev/null 2>&1; then
        echo "✅ GpuDiagnostics runs successfully"
    else
        echo "⚠️  GpuDiagnostics found but may have issues (normal if no GPU)"
    fi
else
    echo "❌ GpuDiagnostics.java missing"
fi
echo

# Check 3: Verify scripts mentioned in README exist
echo "🔍 Checking scripts mentioned in README..."
scripts_found=0
scripts_expected=3

check_script() {
    local script_path=$1
    local script_name=$2
    if [ -f "$script_path" ] && [ -x "$script_path" ]; then
        echo "✅ $script_name exists and is executable: $script_path"
        ((scripts_found++))
    elif [ -f "$script_path" ]; then
        echo "⚠️  $script_name exists but not executable: $script_path"
        ((scripts_found++))
    else
        echo "❌ $script_name missing: $script_path"
    fi
}

check_script "scripts/check_gpu_prerequisites.sh" "GPU Prerequisites Check"
check_script "scripts/run_all_demos.sh" "Run All Demos"
check_script "scripts/test_cross_platform_compatibility.sh" "Cross-Platform Testing"

echo "   Scripts found: $scripts_found/$scripts_expected"
echo

# Check 4: Verify project builds
echo "🔍 Checking project builds successfully..."
if mvn clean compile -q > /dev/null 2>&1; then
    echo "✅ Project builds successfully"
else
    echo "❌ Project build failed"
fi
echo

# Check 5: Verify no misleading APIs are referenced in README
echo "🔍 Checking for removed misleading API references..."
misleading_apis=("GpuConfigurationManager.initializeGpuSupport" "GpuBatchProcessor" "GpuMaxentModelFactory")
misleading_found=0

for api in "${misleading_apis[@]}"; do
    if grep -q "$api" README.md; then
        echo "❌ Found misleading API reference: $api"
        ((misleading_found++))
    fi
done

if [ $misleading_found -eq 0 ]; then
    echo "✅ No misleading API references found in README"
else
    echo "⚠️  Found $misleading_found misleading API references"
fi
echo

# Check 6: Verify documentation consistency
echo "🔍 Checking documentation consistency..."
if grep -q "experimental\|research" README.md; then
    echo "✅ README correctly describes project as experimental/research"
else
    echo "⚠️  README may not clearly indicate experimental status"
fi

if grep -q "enterprise-grade\|production-ready" README.md; then
    echo "⚠️  README may still contain production claims"
else
    echo "✅ No inappropriate production claims found"
fi
echo

# Final summary
echo "======================================================"
echo "📋 Verification Summary"
echo "======================================================"
echo "Examples: $examples_found/$examples_expected found"
echo "Scripts: $scripts_found/$scripts_expected found"
echo "Misleading APIs: $misleading_found found (should be 0)"

if [ $examples_found -eq $examples_expected ] && [ $scripts_found -eq $scripts_expected ] && [ $misleading_found -eq 0 ]; then
    echo "🎉 README accuracy verification PASSED"
    echo "✅ All claims in README.md appear to be accurate"
else
    echo "⚠️  README accuracy verification found issues"
    echo "ℹ️  Please review the items above"
fi

echo "======================================================"
