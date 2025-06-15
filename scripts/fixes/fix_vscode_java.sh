#!/bin/bash

echo "Automating VS Code Java setup and fixes..."
echo "================================================"

PROJECT_DIR=$(pwd)
echo "Working in project: $PROJECT_DIR"
echo ""

# Function to wait for user confirmation
wait_for_user() {
    echo ""
    echo "Press Enter when ready to continue..."
    read -r
}

# Function to check if VS Code is running
check_vscode_running() {
    if pgrep -f "code" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Step 1: Close VS Code completely
echo "Step 1: Closing VS Code completely..."
echo "--------------------------------------"
if check_vscode_running; then
    echo "VS Code is running. Closing all instances..."
    killall code 2>/dev/null || true
    killall code-insiders 2>/dev/null || true
    sleep 2
    
    # Force kill if still running
    if check_vscode_running; then
        echo "Force killing remaining VS Code processes..."
        pkill -9 -f "code" 2>/dev/null || true
        sleep 1
    fi
    echo "✓ VS Code closed"
else
    echo "✓ VS Code is not running"
fi

# Step 2: Clear VS Code extension host cache
echo ""
echo "Step 2: Clearing VS Code extension host cache..."
echo "------------------------------------------------"
if [ -d "$HOME/.vscode/extensions" ]; then
    echo "Removing extension cache files..."
    rm -rf ~/.vscode/extensions/*-cache* 2>/dev/null || true
    rm -rf ~/.vscode/extensions/.obsolete 2>/dev/null || true
    echo "✓ Extension cache cleared"
else
    echo "No VS Code extensions directory found"
fi

# Step 3: Start VS Code fresh
echo ""
echo "Step 3: Starting VS Code fresh..."
echo "----------------------------------"
echo "Opening VS Code in current project directory..."
code . &
VSCODE_PID=$!
sleep 5  # Give VS Code time to start

echo "✓ VS Code started with PID: $VSCODE_PID"
echo ""
echo "Waiting for VS Code to fully load..."
sleep 10  # Additional time for extensions to start loading

# Step 4: Guide through Java setup
echo ""
echo "Step 4: Java Extension Setup"
echo "----------------------------"
cat << 'INSTRUCTIONS'
Please check VS Code now and:

1. Look at the bottom status bar for Java extension initialization
   - You should see "Java: Loading..." or similar
   - Wait until it shows "Java: Ready" or the Java version

2. If prompted to select Java runtime:
   - Choose Java 21 from the list
   - If Java 21 is not listed, click "Download" to get it

3. After Java is loaded, reload the window:
   - Press Ctrl+Shift+P to open command palette
   - Type "Developer: Reload Window" and press Enter
INSTRUCTIONS

echo ""
echo "Have you completed the above steps? (y/n)"
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Please complete the steps before continuing."
    wait_for_user
fi

# Step 5: Troubleshooting steps
echo ""
echo "Step 5: Checking if additional troubleshooting is needed..."
echo "-----------------------------------------------------------"
echo "Is VS Code still freezing or having issues? (y/n)"
read -r has_issues

if [[ "$has_issues" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Applying troubleshooting steps..."
    
    # Create a script to run VS Code commands
    cat > /tmp/vscode_commands.sh << 'EOF'
#!/bin/bash
echo "Running VS Code troubleshooting commands..."

# Clean workspace
echo "Cleaning Java workspace..."
code --command "java.clean.workspace" 2>/dev/null || true
sleep 2

# Rebuild projects
echo "Rebuilding Java projects..."
code --command "java.rebuild.projects" 2>/dev/null || true
sleep 2

echo "Commands sent to VS Code"
EOF
    
    chmod +x /tmp/vscode_commands.sh
    /tmp/vscode_commands.sh
    rm -f /tmp/vscode_commands.sh
    
    echo ""
    echo "Additional manual steps:"
    echo "1. In VS Code, press Ctrl+Shift+P"
    echo "2. Run: 'Java: Clean Workspace'"
    echo "3. Run: 'Java: Rebuild Projects'"
    echo "4. Disable all extensions except Java Extension Pack:"
    echo "   - Click Extensions icon in sidebar"
    echo "   - Click ... menu > Disable All Installed Extensions"
    echo "   - Find 'Java Extension Pack' and re-enable it"
    
    wait_for_user
fi

# Step 6: Full reset if needed
echo ""
echo "Step 6: Full Reset Option"
echo "-------------------------"
echo "Do you want to perform a complete VS Code Java reset? This will:"
echo "- Remove all workspace storage"
echo "- Remove all Java extension data"
echo "- Require re-downloading Java extensions"
echo ""
echo "Perform full reset? (y/n)"
read -r do_reset

if [[ "$do_reset" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Closing VS Code for full reset..."
    killall code 2>/dev/null || true
    sleep 2
    
    echo "Removing VS Code Java data..."
    rm -rf ~/.config/Code/User/workspaceStorage 2>/dev/null || true
    rm -rf ~/.config/Code/User/globalStorage/redhat.java 2>/dev/null || true
    rm -rf ~/.config/Code/User/globalStorage/vscjava.* 2>/dev/null || true
    
    # Also clear some additional caches
    rm -rf ~/.config/Code/Backups 2>/dev/null || true
    rm -rf ~/.config/Code/blob_storage 2>/dev/null || true
    rm -rf ~/.config/Code/GPUCache 2>/dev/null || true
    
    echo "✓ VS Code Java data reset complete"
    echo ""
    echo "Starting VS Code fresh after reset..."
    code . &
    sleep 5
    
    echo ""
    echo "VS Code has been reset. You will need to:"
    echo "1. Wait for Java extensions to download and install"
    echo "2. Select Java 21 when prompted"
    echo "3. Reload the window after setup"
fi

# Alternative VS Code Insiders option
echo ""
echo "Alternative: VS Code Insiders"
echo "-----------------------------"
echo "If regular VS Code continues to have issues, would you like to install and use VS Code Insiders? (y/n)"
read -r use_insiders

if [[ "$use_insiders" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing VS Code Insiders..."
    
    # Check if snap is available
    if command -v snap &> /dev/null; then
        sudo snap install code-insiders --classic
        echo "✓ VS Code Insiders installed"
        
        echo ""
        echo "Closing regular VS Code..."
        killall code 2>/dev/null || true
        sleep 2
        
        echo "Opening project in VS Code Insiders..."
        code-insiders . &
        
        echo ""
        echo "✓ VS Code Insiders is now running"
        echo "Note: You'll need to install Java extensions in Insiders as well"
    else
        echo "Snap is not available. Please install VS Code Insiders manually from:"
        echo "https://code.visualstudio.com/insiders/"
    fi
fi

# Final status check
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Current status:"
echo "- Project directory: $PROJECT_DIR"
echo "- VS Code process: $(pgrep -f "code" > /dev/null && echo "Running" || echo "Not running")"
echo ""
echo "If you're still experiencing issues:"
echo "1. Try restarting your computer"
echo "2. Reinstall Java extensions"
echo "3. Check for VS Code updates"
echo "4. Review the Java extension logs:"
echo "   - View > Output > Select 'Java' from dropdown"
echo ""
echo "Project structure:"
ls -la .vscode/ 2>/dev/null || echo "No .vscode directory found"
