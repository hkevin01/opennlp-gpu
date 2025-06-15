#!/bin/bash

# Function to check if VS Code is installed
check_vscode_installed() {
    if ! command -v code &> /dev/null; then
        echo "Visual Studio Code is not installed. Please install it first."
        exit 1
    fi
}

# Function to update GitHub Copilot extension
update_copilot_extension() {
    echo "Checking for GitHub Copilot extension..."
    EXTENSION_ID="GitHub.copilot"

    # Check if the extension is installed
    if code --list-extensions | grep -q "$EXTENSION_ID"; then
        echo "GitHub Copilot is installed. Updating..."
        code --uninstall-extension "$EXTENSION_ID"
    else
        echo "GitHub Copilot is not installed. Installing..."
    fi

    # Install the latest version of the extension
    code --install-extension "$EXTENSION_ID"
    echo "GitHub Copilot has been updated successfully."
}

# Function to enable editor suggestions in VS Code settings
enable_editor_suggestions() {
    SETTINGS_FILE="$HOME/.config/Code/User/settings.json"

    # Ensure the settings file exists
    if [ ! -f "$SETTINGS_FILE" ]; then
        mkdir -p "$(dirname "$SETTINGS_FILE")"
        echo "{}" > "$SETTINGS_FILE"
    fi

    # Update the settings file to enable editor suggestions
    if grep -q '"github.copilot.editorSuggestions.enabled"' "$SETTINGS_FILE"; then
        sed -i 's/"github.copilot.editorSuggestions.enabled":.*/"github.copilot.editorSuggestions.enabled": true,/' "$SETTINGS_FILE"
    else
        sed -i '1s/^/{\n  "github.copilot.editorSuggestions.enabled": true,\n/' "$SETTINGS_FILE"
    fi

    echo "Editor suggestions for GitHub Copilot have been enabled in VS Code settings."
}

# Main script execution
check_vscode_installed
update_copilot_extension
enable_editor_suggestions
