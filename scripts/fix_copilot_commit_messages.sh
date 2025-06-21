#!/bin/bash

# GitHub Copilot Commit Message Fix Script
# This script helps troubleshoot and fix Copilot commit message generation in VS Code

echo "🔧 GitHub Copilot Commit Message Fix Script"
echo "==========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Check if VS Code is running
echo "🔍 Checking VS Code status..."
if pgrep -f "code" > /dev/null; then
    echo "✅ VS Code is running"
    VSCODE_RUNNING=true
else
    echo "❌ VS Code is not running - we'll start it at the end"
    VSCODE_RUNNING=false
fi

# Check Copilot extensions
echo ""
echo "📦 Checking Copilot Extensions:"
COPILOT_MAIN=$(code --list-extensions | grep "github.copilot$")
COPILOT_CHAT=$(code --list-extensions | grep "github.copilot-chat")

if [ -n "$COPILOT_MAIN" ]; then
    echo "✅ GitHub Copilot extension: $COPILOT_MAIN"
    COPILOT_INSTALLED=true
else
    echo "❌ GitHub Copilot extension not found"
    echo "   Installing GitHub Copilot..."
    code --install-extension github.copilot
    COPILOT_INSTALLED=false
fi

if [ -n "$COPILOT_CHAT" ]; then
    echo "✅ GitHub Copilot Chat extension: $COPILOT_CHAT"
    COPILOT_CHAT_INSTALLED=true
else
    echo "❌ GitHub Copilot Chat extension not found"
    echo "   Installing GitHub Copilot Chat..."
    code --install-extension github.copilot-chat
    COPILOT_CHAT_INSTALLED=false
fi

# Check git status
echo ""
echo "📋 Git Repository Status:"
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "✅ Git repository detected"
    if git status --porcelain | grep -q .; then
        echo "✅ Git changes detected:"
        git status --porcelain | head -10
        if [ $(git status --porcelain | wc -l) -gt 10 ]; then
            echo "   ... and $(( $(git status --porcelain | wc -l) - 10 )) more files"
        fi
        HAS_CHANGES=true
    else
        echo "ℹ️  No git changes found - creating a test change..."
        echo "# Copilot commit message test" >> test_commit_message.tmp
        git add test_commit_message.tmp
        echo "✅ Created test change for commit message testing"
        HAS_CHANGES=true
    fi
else
    echo "❌ No git repository found - initializing..."
    git init
    git add .
    echo "✅ Git repository initialized"
    HAS_CHANGES=true
fi

# Check VS Code settings
echo ""
echo "⚙️  Checking VS Code Settings:"
SETTINGS_FILE=".vscode/settings.json"
if [ -f "$SETTINGS_FILE" ]; then
    echo "✅ VS Code settings file found"
    
    # Check for key Copilot settings
    if grep -q '"scminput": true' "$SETTINGS_FILE"; then
        echo "✅ Copilot SCM input enabled"
    else
        echo "❌ Copilot SCM input not enabled"
    fi
    
    if grep -q 'commitMessage.*true' "$SETTINGS_FILE"; then
        echo "✅ Experimental commit message features enabled"
    else
        echo "❌ Experimental commit message features not enabled"
    fi
else
    echo "❌ VS Code settings file not found"
fi

echo ""
echo "🛠️  Step-by-Step Fix Process:"
echo "================================"

echo ""
echo "Step 1: Reload VS Code window"
if [ "$VSCODE_RUNNING" = true ]; then
    echo "   Executing: Developer: Reload Window command..."
    code --command "workbench.action.reloadWindow"
    sleep 3
    echo "✅ VS Code window reloaded"
else
    echo "   Will open VS Code at the end"
fi

echo ""
echo "Step 2: Verify Copilot authentication"
echo "   Checking Copilot status..."
# Try to get Copilot status (this will work if authenticated)
COPILOT_STATUS=$(code --command "github.copilot.status" 2>/dev/null || echo "unknown")
echo "   Copilot status: $COPILOT_STATUS"

echo ""
echo "Step 3: Test commit message functionality"
if [ "$HAS_CHANGES" = true ]; then
    echo "✅ Ready to test commit messages with current changes"
else
    echo "❌ No changes available for testing"
fi

echo ""
echo "🚀 Manual Testing Steps:"
echo "========================"
echo "1. Open VS Code in this project:"
echo "   code '$PROJECT_DIR'"
echo ""
echo "2. Ensure you're signed in to GitHub Copilot:"
echo "   • Press Ctrl+Shift+P"
echo "   • Run 'GitHub Copilot: Sign In'"
echo "   • Follow the authentication flow"
echo ""
echo "3. Test commit message generation:"
echo "   • Press Ctrl+Shift+G (Source Control)"
echo "   • Stage some files (click + next to files)"
echo "   • Click in the commit message text box"
echo "   • Look for the ✨ sparkle icon or 🤖 robot icon"
echo "   • Type a few words and wait for suggestions"
echo "   • Press Tab to accept suggestions"
echo ""
echo "4. Alternative methods to trigger commit messages:"
echo "   • Press Ctrl+Shift+P and run 'GitHub Copilot Chat: Generate Commit Message'"
echo "   • Right-click in the commit message box for context menu"
echo "   • Use Ctrl+I (inline chat) in the commit message box"

echo ""
echo "🔧 Advanced Troubleshooting:"
echo "============================"
echo "If commit messages still don't work:"
echo ""
echo "1. Check Copilot subscription:"
echo "   • Go to https://github.com/settings/copilot"
echo "   • Verify your subscription is active"
echo ""
echo "2. Reset Copilot extensions:"
echo "   • Press Ctrl+Shift+P"
echo "   • Run 'Extensions: Disable' for GitHub Copilot"
echo "   • Run 'Extensions: Enable' for GitHub Copilot"
echo "   • Restart VS Code"
echo ""
echo "3. Clear VS Code workspace state:"
echo "   • Close VS Code"
echo "   • Delete .vscode/settings.json temporarily"
echo "   • Restart VS Code and reconfigure"
echo ""
echo "4. Check VS Code output logs:"
echo "   • Press Ctrl+Shift+P"
echo "   • Run 'Developer: Toggle Developer Tools'"
echo "   • Check Console for Copilot-related errors"

echo ""
echo "🎯 Expected Behavior:"
echo "===================="
echo "When working correctly, you should see:"
echo "• ✨ or 🤖 icon next to the commit message input"
echo "• Automatic suggestions as you type"
echo "• Context-aware commit messages based on your changes"
echo "• Tab completion for accepting suggestions"

echo ""
if [ "$VSCODE_RUNNING" = false ]; then
    echo "🚀 Starting VS Code..."
    code "$PROJECT_DIR"
    echo "✅ VS Code started with project"
fi

echo ""
echo "📝 Quick Test Summary:"
echo "======================"
echo "Project: $PROJECT_DIR"
echo "Git changes: $([ "$HAS_CHANGES" = true ] && echo "✅ Available" || echo "❌ None")"
echo "Copilot extensions: $([ "$COPILOT_INSTALLED" = true ] && echo "✅ Installed" || echo "❌ Missing")"
echo "VS Code running: $([ "$VSCODE_RUNNING" = true ] && echo "✅ Yes" || echo "🚀 Starting")"

echo ""
echo "✨ Ready to test GitHub Copilot commit message generation!"
echo "   Open Source Control (Ctrl+Shift+G) and try typing in the commit message box."

# Cleanup test file if we created one
if [ -f "test_commit_message.tmp" ]; then
    rm -f test_commit_message.tmp
    git reset HEAD test_commit_message.tmp 2>/dev/null || true
fi
