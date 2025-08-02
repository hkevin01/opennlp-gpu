#!/bin/bash

# Claude AI Prompt Update Script
# Updates all prompt templates to the latest versions

PROMPTS_DIR=".github/prompts"
BACKUP_DIR=".github/prompts-backup-$(date +%Y%m%d-%H%M%S)"

echo "🔄 Updating Claude AI prompts..."

# Create backup
if [[ -d "$PROMPTS_DIR" ]]; then
    echo "📦 Creating backup at $BACKUP_DIR"
    cp -r "$PROMPTS_DIR" "$BACKUP_DIR"
fi

# Update prompt library metadata
echo "📝 Updating prompt library metadata"
sed -i "s/created: .*/created: \"$(date)\"/" "$PROMPTS_DIR/prompt-library.yml"

echo "✅ Claude AI prompts updated successfully!"
echo "📦 Backup available at: $BACKUP_DIR"
