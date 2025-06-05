#!/bin/bash

echo "Upgrading Gradle to version 8.10 to support Java 21..."

# Set Gradle version
GRADLE_VERSION="8.10"
TEMP_DIR="$(pwd)/gradle-temp"

# Create temp directory for downloads
mkdir -p "$TEMP_DIR"

echo "Downloading Gradle ${GRADLE_VERSION} distribution..."
GRADLE_ZIP="$TEMP_DIR/gradle-${GRADLE_VERSION}-bin.zip"
DOWNLOAD_URL="https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip"

# Download the full Gradle distribution
curl -L -o "$GRADLE_ZIP" "$DOWNLOAD_URL"

if [ ! -f "$GRADLE_ZIP" ] || [ ! -s "$GRADLE_ZIP" ]; then
    echo "Failed to download Gradle distribution. Please check your internet connection."
    exit 1
fi

echo "Extracting Gradle distribution..."
unzip -q "$GRADLE_ZIP" -d "$TEMP_DIR"

# Now use the downloaded Gradle to generate a proper wrapper
echo "Generating Gradle wrapper using downloaded distribution..."
"$TEMP_DIR/gradle-${GRADLE_VERSION}/bin/gradle" wrapper --gradle-version "$GRADLE_VERSION" --distribution-type bin

# Verify the wrapper is working
if [ -f "$(pwd)/gradlew" ]; then
    echo "Testing Gradle wrapper..."
    chmod +x "$(pwd)/gradlew"
    ./gradlew --version
    
    if [ $? -eq 0 ]; then
        echo "Gradle wrapper is working correctly."
    else
        echo "WARNING: Gradle wrapper test failed."
    fi
else
    echo "ERROR: gradlew script could not be created."
    exit 1
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Gradle upgraded to version ${GRADLE_VERSION}!"
echo "This version supports Java 21."
echo "To run the build, use: ./gradlew build"
