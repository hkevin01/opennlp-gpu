#!/bin/bash

# Make all scripts executable
chmod +x "$(pwd)/scripts/fix_plugin_conflict.sh"
chmod +x "$(pwd)/scripts/upgrade_gradle.sh"
chmod +x "$(pwd)/scripts/add_lombok.sh"
chmod +x "$(pwd)/scripts/fix_gradle_structure.sh"

echo "Scripts are now executable!"
echo "To fix the plugin conflict: ./scripts/fix_plugin_conflict.sh"
echo "To upgrade Gradle for Java 21 support: ./scripts/upgrade_gradle.sh"
echo "To add Lombok to the project: ./scripts/add_lombok.sh"
echo "To fix Gradle build file structure: ./scripts/fix_gradle_structure.sh"
