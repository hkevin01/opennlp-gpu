#!/bin/bash

# GitHub Copilot Agent Auto-Continue Helper Script
# This script helps with automated continuation in VS Code Copilot Agent mode

echo "🤖 GitHub Copilot Agent Auto-Continue Helper"
echo "=============================================="

# Function to send continue command to active VS Code instance
send_continue() {
    echo "Sending 'continue' command to VS Code..."
    # Try to send continue command via VS Code CLI
    code --command "github.copilot.chat.continue" 2>/dev/null || {
        # Alternative: use xdotool if available (Linux)
        if command -v xdotool &> /dev/null; then
            echo "Using xdotool to send keystrokes..."
            # Focus VS Code window and send Ctrl+Shift+Enter
            xdotool search --name "Visual Studio Code" windowactivate
            sleep 0.2
            xdotool key ctrl+shift+Return
        else
            echo "⚠️  Could not send continue command automatically"
            echo "💡 Tip: Use Ctrl+Shift+Enter in VS Code to continue"
        fi
    }
}

# Function to setup auto-continue watcher
setup_watcher() {
    echo "Setting up auto-continue watcher..."
    echo "This will monitor for agent completion and auto-send continue..."
    
    # Monitor VS Code log for agent completion (if available)
    if [ -f "$HOME/.config/Code/logs/renderer1.log" ]; then
        echo "Monitoring VS Code logs for agent completion..."
        tail -f "$HOME/.config/Code/logs/renderer1.log" | while read line; do
            if echo "$line" | grep -q "agent.*complete\|task.*finished\|waiting.*input"; then
                echo "🔄 Agent task completed, sending continue..."
                send_continue
                sleep 2
            fi
        done &
        echo "Watcher started in background (PID: $!)"
        echo "Stop with: kill $!"
    else
        echo "⚠️  VS Code log file not found"
        echo "💡 Manual continue required"
    fi
}

# Check arguments
case "$1" in
    "continue")
        send_continue
        ;;
    "watch")
        setup_watcher
        ;;
    "setup")
        echo "Setting up VS Code for auto-continue mode..."
        echo "✅ Settings configured in .vscode/settings.json"
        echo "✅ Keybindings configured for quick continue"
        echo "✅ Use Ctrl+Shift+Space for quick 'continue' in chat"
        echo "✅ Use Ctrl+Alt+C for continue conversation"
        echo ""
        echo "🎯 Usage:"
        echo "  $0 continue  - Send continue command once"
        echo "  $0 watch     - Monitor and auto-continue"
        echo "  $0 setup     - Show setup information"
        ;;
    *)
        echo "Usage: $0 {continue|watch|setup}"
        echo ""
        echo "Commands:"
        echo "  continue  - Send continue command to active VS Code"
        echo "  watch     - Setup auto-continue watcher"
        echo "  setup     - Show setup information"
        echo ""
        echo "Quick Tips:"
        echo "  • Ctrl+Shift+Enter - Continue in chat"
        echo "  • Ctrl+Shift+Space - Quick 'continue' text"
        echo "  • Ctrl+Alt+C - Continue conversation"
        exit 1
        ;;
esac
