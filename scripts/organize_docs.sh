#!/bin/bash
set -e

DOCS_DIR="$(dirname "$0")/../docs"

# Create subfolders
mkdir -p "$DOCS_DIR/overview"
mkdir -p "$DOCS_DIR/testing"
mkdir -p "$DOCS_DIR/usage"
mkdir -p "$DOCS_DIR/lombok"
mkdir -p "$DOCS_DIR/build"
mkdir -p "$DOCS_DIR/api"

# Move overview/project files
mv "$DOCS_DIR/project_plan.md" "$DOCS_DIR/overview/" 2>/dev/null || true
mv "$DOCS_DIR/project_progress.md" "$DOCS_DIR/overview/" 2>/dev/null || true
mv "$DOCS_DIR/technologies_overview.md" "$DOCS_DIR/overview/" 2>/dev/null || true

# Move testing files
mv "$DOCS_DIR/test_plan.md" "$DOCS_DIR/testing/" 2>/dev/null || true
mv "$DOCS_DIR/test_plan_progress.md" "$DOCS_DIR/testing/" 2>/dev/null || true

# Move usage/user guide files
mv "$DOCS_DIR/user_guide.md" "$DOCS_DIR/usage/" 2>/dev/null || true
mv "$DOCS_DIR/getting_started.md" "$DOCS_DIR/usage/" 2>/dev/null || true
mv "$DOCS_DIR/logging.md" "$DOCS_DIR/usage/" 2>/dev/null || true

# Move Lombok-related docs
mv "$DOCS_DIR/lombok-removal.md" "$DOCS_DIR/lombok/" 2>/dev/null || true
mv "$DOCS_DIR/lombok-removal-guide.md" "$DOCS_DIR/lombok/" 2>/dev/null || true
mv "$DOCS_DIR/lombok-vscode-issues.md" "$DOCS_DIR/lombok/" 2>/dev/null || true
mv "$DOCS_DIR/lombok-slf4j-guide.md" "$DOCS_DIR/lombok/" 2>/dev/null || true
mv "$DOCS_DIR/lombok-and-h2.md" "$DOCS_DIR/lombok/" 2>/dev/null || true

# Move build tool docs
mv "$DOCS_DIR/gradle-vs-maven.md" "$DOCS_DIR/build/" 2>/dev/null || true
mv "$DOCS_DIR/gradle-commands.md" "$DOCS_DIR/build/" 2>/dev/null || true

# Move API docs if present
if compgen -G "$DOCS_DIR/api*" > /dev/null; then
  mv "$DOCS_DIR/api"* "$DOCS_DIR/api/" 2>/dev/null || true
fi

echo "✅ Documentation organized into subfolders."
