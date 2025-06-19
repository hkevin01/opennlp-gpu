#!/bin/bash
# Source this file to set up Java environment
# Usage: source scripts/setup_java_env.sh

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Java environment loaded from .env"
    echo "   JAVA_HOME: $JAVA_HOME"
    echo "   Java version: $(java -version 2>&1 | head -1)"
else
    echo "❌ .env file not found. Run ./scripts/fix_java_environment.sh first"
fi
