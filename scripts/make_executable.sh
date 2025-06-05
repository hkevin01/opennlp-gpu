#!/bin/bash

# Make the scripts executable
chmod +x "$(pwd)/scripts/fix_plugin_conflict.sh"
chmod +x "$(pwd)/scripts/upgrade_gradle.sh"

echo "Scripts are now executable!"
echo "To fix the plugin conflict: ./scripts/fix_plugin_conflict.sh"
echo "To upgrade Gradle for Java 21 support: ./scripts/upgrade_gradle.sh"
