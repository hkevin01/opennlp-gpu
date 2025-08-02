#!/bin/bash
# Deploy to Maven Central
# Usage: ./deploy.sh [version]

set -euo pipefail

VERSION=${1:-$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout)}

echo "🚀 Deploying OpenNLP GPU Extension v$VERSION to Maven Central"

# Ensure we have GPG key for signing
if ! gpg --list-secret-keys | grep -q "sec"; then
    echo "❌ No GPG secret key found. Please set up GPG signing first:"
    echo "   gpg --gen-key"
    echo "   gpg --keyserver keyserver.ubuntu.com --send-keys YOUR_KEY_ID"
    exit 1
fi

# Clean and test
echo "🧹 Cleaning and testing..."
mvn clean test

# Build documentation
echo "📚 Building documentation..."
mvn javadoc:javadoc

# Deploy to staging repository
echo "📦 Deploying to staging repository..."
mvn deploy -P release

echo "✅ Successfully deployed to staging repository!"
echo ""
echo "Next steps:"
echo "1. Go to https://s01.oss.sonatype.org/"
echo "2. Login with your Sonatype credentials"
echo "3. Navigate to 'Staging Repositories'"
echo "4. Find 'orgapacheopennlp-XXXX'"
echo "5. Click 'Close' to validate the staging repository"
echo "6. If validation passes, click 'Release' to publish to Maven Central"
echo ""
echo "The artifact will be available at:"
echo "  https://repo1.maven.org/maven2/org/apache/opennlp/opennlp-gpu/$VERSION/"
echo "  (Usually available within 2-4 hours)"
