#!/bin/bash

echo "Fixing plugin conflict in build.gradle..."

# Path to the build.gradle file
BUILD_GRADLE="$(pwd)/build.gradle"

# Create a backup of the original file
cp "$BUILD_GRADLE" "${BUILD_GRADLE}.bak"

# Replace the conflicting plugin setup with a custom native build approach
sed -i '/id .cpp-library./d' "$BUILD_GRADLE"
sed -i '/id .cpp-unit-test./d' "$BUILD_GRADLE"

# Add custom native build support without plugin conflict
cat << 'EOF' > temp_file
plugins {
    id 'java'
    id 'io.freefair.lombok' version '8.4'
    // Remove cpp plugins to avoid configuration conflicts
}

group = 'org.apache.opennlp'
version = '0.1.0-SNAPSHOT'

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(11)
    }
}

// Custom native build configuration without cpp-library plugin
configurations {
    nativeBuild
}

repositories {
    mavenCentral()
}

// The rest of your build.gradle file...
EOF

# Merge the temporary file with the rest of the build.gradle
sed -n '/The rest of your build.gradle file/,$p' temp_file > header.tmp
sed '1,/repositories {/d' "$BUILD_GRADLE" > body.tmp
cat header.tmp body.tmp > "$BUILD_GRADLE"
rm temp_file header.tmp body.tmp

# Add custom native build task
cat << 'EOF' >> "$BUILD_GRADLE"

// Define a custom task to build CUDA native code with CMake
task buildCudaNative(type: Exec) {
    description = 'Build CUDA native library using CMake'
    group = 'Build'
    
    // Create build directory if it doesn't exist
    doFirst {
        mkdir("${buildDir}/cmake")
    }
    
    // Configure CMake
    workingDir "${buildDir}/cmake"
    commandLine 'cmake', "${projectDir}/src/main/cpp", 
                '-DCMAKE_BUILD_TYPE=Release',
                "-DCMAKE_INSTALL_PREFIX=${buildDir}/natives"
    
    // Build and install
    doLast {
        exec {
            workingDir "${buildDir}/cmake"
            commandLine 'cmake', '--build', '.', '--target', 'install'
        }
    }
}

// Make Java compilation depend on native library
compileJava.dependsOn buildCudaNative

// Add native library directory to Java library path
tasks.withType(Test) {
    systemProperty 'java.library.path', "${buildDir}/natives/lib"
}

// Task to copy native libraries to the JAR
task copyNativesToJar(type: Copy) {
    from "${buildDir}/natives/lib"
    into "${buildDir}/resources/main/natives"
    include '*.so', '*.dll', '*.dylib', '*.jnilib'
}

// Make processResources depend on copying natives
processResources.dependsOn copyNativesToJar

EOF

echo "Plugin conflict fixed in build.gradle!"
echo "The original file has been backed up to ${BUILD_GRADLE}.bak"
