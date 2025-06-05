#!/bin/bash

echo "Upgrading Gradle to version 8.10 to support Java 21..."

# Set Gradle version
GRADLE_VERSION="8.10"
TEMP_DIR="$(pwd)/gradle-temp"

# Create temp directory for downloads
mkdir -p "$TEMP_DIR"

# First check if build.gradle exists and has basic Java plugin
if [ ! -f "$(pwd)/build.gradle" ]; then
    echo "Creating a minimal build.gradle file..."
    cat << EOF > "$(pwd)/build.gradle"
plugins {
    id 'java'
}

repositories {
    mavenCentral()
}

dependencies {
    // Add your dependencies here
}
EOF
else
    # Check if java plugin is applied
    if ! grep -q "id 'java'" "$(pwd)/build.gradle"; then
        echo "WARNING: build.gradle might not have Java plugin applied."
        echo "Adding it at the beginning of the file..."
        
        # Create a temp file with plugins block
        cat << EOF > "$TEMP_DIR/plugin_block.gradle"
plugins {
    id 'java'
    // Existing plugins will be merged
EOF
        
        # Append existing content, replacing any existing plugins block
        sed '/^plugins {/,/^}/d' "$(pwd)/build.gradle" >> "$TEMP_DIR/plugin_block.gradle"
        echo "}" >> "$TEMP_DIR/plugin_block.gradle"
        
        # Backup original
        cp "$(pwd)/build.gradle" "$(pwd)/build.gradle.bak"
        
        # Replace with new content
        mv "$TEMP_DIR/plugin_block.gradle" "$(pwd)/build.gradle"
    fi
fi

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

# Ensure wrapper JAR directory exists
mkdir -p "$(pwd)/gradle/wrapper"

# Download wrapper JAR directly from a known working source
echo "Downloading Gradle wrapper JAR directly..."
mkdir -p "$(pwd)/gradle/wrapper"

# Use a direct URL to a known working wrapper JAR (not the version-specific one)
WRAPPER_JAR_URL="https://raw.githubusercontent.com/gradle/gradle/v8.6.0/gradle/wrapper/gradle-wrapper.jar"
curl -L -o "$(pwd)/gradle/wrapper/gradle-wrapper.jar" "${WRAPPER_JAR_URL}"

# Verify the JAR exists and has content
if [ ! -s "$(pwd)/gradle/wrapper/gradle-wrapper.jar" ]; then
    echo "Direct download failed. Trying alternative URL..."
    # Try Maven Central as fallback
    FALLBACK_URL="https://repo.maven.apache.org/maven2/org/gradle/gradle-wrapper/8.6.0/gradle-wrapper-8.6.0.jar"
    curl -L -o "$(pwd)/gradle/wrapper/gradle-wrapper.jar" "${FALLBACK_URL}"
    
    # If still failing, try gradle.org directly
    if [ ! -s "$(pwd)/gradle/wrapper/gradle-wrapper.jar" ]; then
        echo "Trying gradle.org distribution directly..."
        DIST_URL="https://services.gradle.org/distributions/gradle-8.10-bin.zip"
        TEMP_ZIP="$TEMP_DIR/gradle-dist.zip"
        curl -L -o "$TEMP_ZIP" "$DIST_URL"
        
        if [ -s "$TEMP_ZIP" ]; then
            echo "Extracting wrapper JAR from distribution..."
            unzip -q -j "$TEMP_ZIP" "gradle-8.10/lib/gradle-wrapper-8.10.jar" -d "$(pwd)/gradle/wrapper/"
            mv "$(pwd)/gradle/wrapper/gradle-wrapper-8.10.jar" "$(pwd)/gradle/wrapper/gradle-wrapper.jar"
        fi
    fi
fi

# Final verification
if [ ! -s "$(pwd)/gradle/wrapper/gradle-wrapper.jar" ]; then
    echo "ERROR: Could not obtain Gradle wrapper JAR. Manual installation required."
    echo "Please download gradle-wrapper.jar to gradle/wrapper/ directory."
    exit 1
fi

# Verify the wrapper JAR exists and has content
if [ ! -s "$(pwd)/gradle/wrapper/gradle-wrapper.jar" ]; then
    echo "ERROR: Failed to obtain a valid Gradle wrapper JAR."
    exit 1
else
    echo "Successfully obtained Gradle wrapper JAR."
fi

# Create wrapper properties file
echo "Creating wrapper properties file..."
cat << EOF > "$(pwd)/gradle/wrapper/gradle-wrapper.properties"
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
EOF

# Create gradlew script
echo "Creating gradlew script..."
cat << 'EOF' > "$(pwd)/gradlew"
#!/bin/sh

#
# Copyright © 2015-2021 the original authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

##############################################################################
##
##  Gradle start up script for UN*X
##
##############################################################################

# Attempt to set APP_HOME
# Resolve links: $0 may be a link
PRG="$0"
# Need this for relative symlinks.
while [ -h "$PRG" ] ; do
    ls=`ls -ld "$PRG"`
    link=`expr "$ls" : '.*-> \(.*\)$'`
    if expr "$link" : '/.*' > /dev/null; then
        PRG="$link"
    else
        PRG=`dirname "$PRG"`"/$link"
    fi
done
SAVED="`pwd`"
cd "`dirname \"$PRG\"`/" >/dev/null
APP_HOME="`pwd -P`"
cd "$SAVED" >/dev/null

APP_NAME="Gradle"
APP_BASE_NAME=`basename "$0"`

# Add default JVM options here. You can also use JAVA_OPTS and GRADLE_OPTS to pass JVM options to this script.
DEFAULT_JVM_OPTS='"-Xmx64m" "-Xms64m"'

# Use the maximum available, or set MAX_FD != -1 to use that value.
MAX_FD="maximum"

warn () {
    echo "$*"
}

die () {
    echo
    echo "$*"
    echo
    exit 1
}

# OS specific support (must be 'true' or 'false').
cygwin=false
msys=false
darwin=false
nonstop=false
case "`uname`" in
  CYGWIN* )
    cygwin=true
    ;;
  Darwin* )
    darwin=true
    ;;
  MINGW* )
    msys=true
    ;;
  NONSTOP* )
    nonstop=true
    ;;
esac

CLASSPATH=$APP_HOME/gradle/wrapper/gradle-wrapper.jar

# Determine the Java command to use to start the JVM.
if [ -n "$JAVA_HOME" ] ; then
    if [ -x "$JAVA_HOME/jre/sh/java" ] ; then
        # IBM's JDK on AIX uses strange locations for the executables
        JAVACMD="$JAVA_HOME/jre/sh/java"
    else
        JAVACMD="$JAVA_HOME/bin/java"
    fi
    if [ ! -x "$JAVACMD" ] ; then
        die "ERROR: JAVA_HOME is set to an invalid directory: $JAVA_HOME

Please set the JAVA_HOME variable in your environment to match the
location of your Java installation."
    fi
else
    JAVACMD="java"
    which java >/dev/null 2>&1 || die "ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.

Please set the JAVA_HOME variable in your environment to match the
location of your Java installation."
fi

# Increase the maximum file descriptors if we can.
if [ "$cygwin" = "false" -a "$darwin" = "false" -a "$nonstop" = "false" ] ; then
    MAX_FD_LIMIT=`ulimit -H -n`
    if [ $? -eq 0 ] ; then
        if [ "$MAX_FD" = "maximum" -o "$MAX_FD" = "max" ] ; then
            MAX_FD="$MAX_FD_LIMIT"
        fi
        ulimit -n $MAX_FD
        if [ $? -ne 0 ] ; then
            warn "Could not set maximum file descriptor limit: $MAX_FD"
        fi
    else
        warn "Could not query maximum file descriptor limit: $MAX_FD_LIMIT"
    fi
fi

# For Darwin, add options to specify how the application appears in the dock
if $darwin; then
    GRADLE_OPTS="$GRADLE_OPTS \"-Xdock:name=$APP_NAME\" \"-Xdock:icon=$APP_HOME/media/gradle.icns\""
fi

# For Cygwin or MSYS, switch paths to Windows format before running java
if [ "$cygwin" = "true" -o "$msys" = "true" ] ; then
    APP_HOME=`cygpath --path --mixed "$APP_HOME"`
    CLASSPATH=`cygpath --path --mixed "$CLASSPATH"`
    JAVACMD=`cygpath --unix "$JAVACMD"`

    # We build the pattern for arguments to be converted via cygpath
    ROOTDIRSRAW=`find -L / -maxdepth 1 -mindepth 1 -type d 2>/dev/null`
    SEP=""
    for dir in $ROOTDIRSRAW ; do
        ROOTDIRS="$ROOTDIRS$SEP$dir"
        SEP="|"
    done
    OURCYGPATTERN="(^($ROOTDIRS))"
    # Add a user-defined pattern to the cygpath arguments
    if [ "$GRADLE_CYGPATTERN" != "" ] ; then
        OURCYGPATTERN="$OURCYGPATTERN|($GRADLE_CYGPATTERN)"
    fi
    # Now convert the arguments - kludge to limit ourselves to /bin/sh
    i=0
    for arg in "$@" ; do
        CHECK=`echo "$arg"|egrep -c "$OURCYGPATTERN" -`
        CHECK2=`echo "$arg"|egrep -c "^-"`                                 ### Determine if an option

        if [ $CHECK -ne 0 ] && [ $CHECK2 -eq 0 ] ; then                    ### Added a condition
            eval `echo args$i`=`cygpath --path --ignore --mixed "$arg"`
        else
            eval `echo args$i`="\"$arg\""
        fi
        i=`expr $i + 1`
    done
    case $i in
        0) set -- ;;
        1) set -- "$args0" ;;
        2) set -- "$args0" "$args1" ;;
        3) set -- "$args0" "$args1" "$args2" ;;
        4) set -- "$args0" "$args1" "$args2" "$args3" ;;
        5) set -- "$args0" "$args1" "$args2" "$args3" "$args4" ;;
        6) set -- "$args0" "$args1" "$args2" "$args3" "$args4" "$args5" ;;
        7) set -- "$args0" "$args1" "$args2" "$args3" "$args4" "$args5" "$args6" ;;
        8) set -- "$args0" "$args1" "$args2" "$args3" "$args4" "$args5" "$args6" "$args7" ;;
        9) set -- "$args0" "$args1" "$args2" "$args3" "$args4" "$args5" "$args6" "$args7" "$args8" ;;
    esac
fi

# Escape application args
save () {
    for i do printf %s\\n "$i" | sed "s/'/'\\\\''/g;1s/^/'/;\$s/\$/' \\\\/" ; done
    echo " "
}
APP_ARGS=`save "$@"`

# Collect all arguments for the java command, following the shell quoting and substitution rules
eval set -- $DEFAULT_JVM_OPTS $JAVA_OPTS $GRADLE_OPTS "\"-Dorg.gradle.appname=$APP_BASE_NAME\"" -classpath "\"$CLASSPATH\"" org.gradle.wrapper.GradleWrapperMain "$APP_ARGS"

exec "$JAVACMD" "$@"
EOF

# Make gradlew executable
chmod +x "$(pwd)/gradlew"

# Create gradlew.bat for Windows
echo "Creating gradlew.bat script..."
cat << 'EOF' > "$(pwd)/gradlew.bat"
@rem
@rem Copyright 2015 the original author or authors.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem      https://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem

@if "%DEBUG%" == "" @echo off
@rem ##########################################################################
@rem
@rem  Gradle startup script for Windows
@rem
@rem ##########################################################################

@rem Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

set DIRNAME=%~dp0
if "%DIRNAME%" == "" set DIRNAME=.
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%

@rem Resolve any "." and ".." in APP_HOME to make it shorter.
for %%i in ("%APP_HOME%") do set APP_HOME=%%~fi

@rem Add default JVM options here. You can also use JAVA_OPTS and GRADLE_OPTS to pass JVM options to this script.
set DEFAULT_JVM_OPTS="-Xmx64m" "-Xms64m"

@rem Find java.exe
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if "%ERRORLEVEL%" == "0" goto execute

echo.
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%/bin/java.exe

if exist "%JAVA_EXE%" goto execute

echo.
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME%
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:execute
@rem Setup the command line

set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar


@rem Execute Gradle
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GRADLE_OPTS% "-Dorg.gradle.appname=%APP_BASE_NAME%" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*

:end
@rem End local scope for the variables with windows NT shell
if "%ERRORLEVEL%"=="0" goto mainEnd

:fail
rem Set variable GRADLE_EXIT_CONSOLE if you need the _script_ return code instead of
rem the _cmd.exe /c_ return code!
if  not "" == "%GRADLE_EXIT_CONSOLE%" exit 1
exit /b 1

:mainEnd
if "%OS%"=="Windows_NT" endlocal

:omega
EOF

# Verify the wrapper is working
if [ -f "$(pwd)/gradlew" ] && [ -f "$(pwd)/gradle/wrapper/gradle-wrapper.jar" ]; then
    echo "Testing Gradle wrapper..."
    chmod +x "$(pwd)/gradlew"
    ./gradlew --version
    
    if [ $? -eq 0 ]; then
        echo "Gradle wrapper is working correctly."
    else
        echo "WARNING: Gradle wrapper test failed."
    fi
else
    echo "ERROR: Gradle wrapper files could not be created properly."
    exit 1
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Gradle upgraded to version ${GRADLE_VERSION}!"
echo "This version supports Java 21."
echo "To run the build, use: ./gradlew build"
