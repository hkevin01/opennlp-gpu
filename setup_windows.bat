@echo off
REM OpenNLP GPU Extension - Windows Setup Script
REM Supports: Windows 10/11 with WSL, native Windows with Visual Studio
REM Compatible with: Local machines, Azure VMs, AWS EC2 Windows instances

setlocal enabledelayedexpansion

REM Colors for output (Windows 10+ with ANSI support)
set "RED=[31m"
set "GREEN=[32m"
set "YELLOW=[33m"
set "BLUE=[34m"
set "NC=[0m"

REM Script configuration
set "SCRIPT_DIR=%~dp0"
set "LOG_FILE=%SCRIPT_DIR%logs\setup.log"
set "ERROR_LOG=%SCRIPT_DIR%logs\setup-errors.log"
set "JAVA_VERSION=21"
set "CMAKE_MIN_VERSION=3.16"

REM Create logs directory
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

REM System detection
set "OS_TYPE=windows"
set "GPU_TYPE="
set "CLOUD_PROVIDER="

REM Installation flags
set "INSTALL_JAVA=false"
set "INSTALL_CMAKE=false"
set "INSTALL_MAVEN=false"
set "INSTALL_GPU_DRIVERS=false"
set "BUILD_NATIVE=false"
set "BUILD_JAVA=false"

echo %BLUE%[INFO]%NC% Starting OpenNLP GPU Extension Setup for Windows...
echo %BLUE%[INFO]%NC% Log file: %LOG_FILE%
echo.

REM Function to print status messages
:print_status
echo %BLUE%[INFO]%NC% %~1
echo [%date% %time%] INFO: %~1 >> "%LOG_FILE%"
goto :eof

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
echo [%date% %time%] SUCCESS: %~1 >> "%LOG_FILE%"
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
echo [%date% %time%] WARNING: %~1 >> "%LOG_FILE%"
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
echo [%date% %time%] ERROR: %~1 >> "%ERROR_LOG%"
goto :eof

:print_step
echo.
echo %BLUE%=== %~1 ===%NC%
echo [%date% %time%] STEP: %~1 >> "%LOG_FILE%"
goto :eof

REM Detect system information
:detect_system
call :print_step "Detecting System Configuration"

REM Detect Windows version
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
call :print_status "Detected Windows version: %VERSION%"

REM Check for WSL
wsl --status >nul 2>&1
if %errorlevel% equ 0 (
    set "WSL_AVAILABLE=true"
    call :print_status "WSL detected and available"
) else (
    set "WSL_AVAILABLE=false"
    call :print_status "WSL not available, using native Windows build"
)

REM Detect Visual Studio
where cl >nul 2>&1
if %errorlevel% equ 0 (
    set "VISUAL_STUDIO=true"
    call :print_status "Visual Studio compiler detected"
) else (
    set "VISUAL_STUDIO=false"
    call :print_warning "Visual Studio compiler not found in PATH"
)

REM Detect GPU
call :detect_gpu
goto :eof

REM GPU Detection
:detect_gpu
call :print_step "Detecting GPU Hardware"

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set "GPU_TYPE=nvidia"
    call :print_success "NVIDIA GPU detected"
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader,nounits') do (
        call :print_status "GPU: %%i"
    )
) else (
    REM Check for AMD GPU (if AMD software is installed)
    where rocm-smi >nul 2>&1
    if %errorlevel% equ 0 (
        set "GPU_TYPE=amd"
        call :print_success "AMD GPU with ROCm detected"
    ) else (
        REM Check Windows GPU info
        wmic path win32_VideoController get name /value 2>nul | findstr /i "radeon" >nul
        if %errorlevel% equ 0 (
            set "GPU_TYPE=amd_no_rocm"
            call :print_warning "AMD GPU detected but ROCm not installed"
        ) else (
            set "GPU_TYPE=cpu_only"
            call :print_warning "No supported GPU detected, will use CPU-only mode"
        )
    )
)
goto :eof

REM Check Java installation
:check_java
call :print_step "Checking Java Installation"

where java >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%i in ('java -version 2^>^&1 ^| findstr /i version') do (
        set "JAVA_VER=%%i"
        set "JAVA_VER=!JAVA_VER:"=!"
    )
    call :print_status "Java found: !JAVA_VER!"
    
    REM Check if version is adequate (simplified check)
    echo !JAVA_VER! | findstr /r "^1[1-9]\|^[2-9]" >nul
    if %errorlevel% equ 0 (
        call :print_success "Java version is adequate"
    ) else (
        call :print_warning "Java version may be too old, recommend Java 11+"
        set "INSTALL_JAVA=true"
    )
) else (
    call :print_warning "Java not found, will install"
    set "INSTALL_JAVA=true"
)
goto :eof

REM Check Maven installation
:check_maven
call :print_step "Checking Maven Installation"

where mvn >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%i in ('mvn -version 2^>^&1 ^| findstr /i "Apache Maven"') do (
        call :print_success "Maven found: %%i"
    )
) else (
    call :print_warning "Maven not found, will install"
    set "INSTALL_MAVEN=true"
)
goto :eof

REM Check CMake installation
:check_cmake
call :print_step "Checking CMake Installation"

where cmake >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%i in ('cmake --version 2^>^&1') do (
        call :print_success "CMake found: %%i"
    )
) else (
    call :print_warning "CMake not found, will install"
    set "INSTALL_CMAKE=true"
)
goto :eof

REM Install dependencies
:install_dependencies
call :print_step "Installing Dependencies"

if "%INSTALL_JAVA%"=="true" (
    call :print_status "Installing Java..."
    call :print_warning "Please install Java 11+ from https://adoptium.net/ or use Chocolatey:"
    call :print_status "choco install openjdk"
)

if "%INSTALL_MAVEN%"=="true" (
    call :print_status "Installing Maven..."
    call :print_warning "Please install Maven from https://maven.apache.org/ or use Chocolatey:"
    call :print_status "choco install maven"
)

if "%INSTALL_CMAKE%"=="true" (
    call :print_status "Installing CMake..."
    call :print_warning "Please install CMake from https://cmake.org/ or use Chocolatey:"
    call :print_status "choco install cmake"
)

REM Check if we can proceed
if "%INSTALL_JAVA%"=="true" (
    call :print_error "Please install Java and run this script again"
    goto :error_exit
)

if "%INSTALL_CMAKE%"=="true" (
    call :print_error "Please install CMake and run this script again"
    goto :error_exit
)
goto :eof

REM Build native library
:build_native_library
call :print_step "Building Native C++ Library"

cd /d "%SCRIPT_DIR%src\main\cpp"
if %errorlevel% neq 0 (
    call :print_error "Cannot find C++ source directory"
    goto :error_exit
)

REM Clean previous builds
if exist CMakeCache.txt del /q CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles
if exist Makefile del /q Makefile
if exist build rmdir /s /q build
if exist *.dll del /q *.dll
if exist *.lib del /q *.lib

REM Configure with CMake
call :print_status "Configuring build with CMake..."

set "CMAKE_ARGS="
if "%GPU_TYPE%"=="nvidia" (
    set "CMAKE_ARGS=-DUSE_CUDA=ON"
) else if "%GPU_TYPE%"=="amd" (
    set "CMAKE_ARGS=-DUSE_ROCM=ON"
)

REM Use Visual Studio generator if available
if "%VISUAL_STUDIO%"=="true" (
    set "CMAKE_GENERATOR=-G "Visual Studio 17 2022""
) else (
    set "CMAKE_GENERATOR=-G "MinGW Makefiles""
)

call :print_status "Running: cmake . %CMAKE_GENERATOR% %CMAKE_ARGS%"
cmake . %CMAKE_GENERATOR% %CMAKE_ARGS% >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    call :print_warning "CMake configuration failed, trying CPU-only build..."
    cmake . %CMAKE_GENERATOR% -DUSE_CPU_ONLY=ON >> "%LOG_FILE%" 2>&1
    if %errorlevel% neq 0 (
        call :print_error "CMake configuration failed completely"
        call :print_error "Check %LOG_FILE% for detailed error information"
        goto :error_exit
    )
)

REM Build
call :print_status "Building native library..."
if "%VISUAL_STUDIO%"=="true" (
    cmake --build . --config Release >> "%LOG_FILE%" 2>&1
) else (
    make >> "%LOG_FILE%" 2>&1
)

if %errorlevel% neq 0 (
    call :print_error "Native library build failed"
    call :print_error "Check %LOG_FILE% for detailed error information"
    goto :error_exit
)

REM Verify build
if exist "opennlp_gpu.dll" (
    call :print_success "Native library built successfully"
    
    REM Copy to Java resources
    if not exist "%SCRIPT_DIR%src\main\resources\native\windows\x86_64" mkdir "%SCRIPT_DIR%src\main\resources\native\windows\x86_64"
    copy opennlp_gpu.dll "%SCRIPT_DIR%src\main\resources\native\windows\x86_64\" >nul 2>&1
    if %errorlevel% equ 0 (
        call :print_success "Native library copied to resources"
    ) else (
        call :print_warning "Could not copy native library to resources directory"
    )
) else (
    call :print_error "Native library build verification failed"
    goto :error_exit
)

cd /d "%SCRIPT_DIR%"
goto :eof

REM Build Java components
:build_java_components
call :print_step "Building Java Components"

call :print_status "Running Maven build..."
mvn clean compile >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    call :print_error "Maven build failed"
    call :print_error "Check %LOG_FILE% for detailed error information"
    goto :error_exit
)

call :print_success "Java components built successfully"
goto :eof

REM Run verification tests
:run_verification
call :print_step "Running Verification Tests"

call :print_status "Testing GPU detection..."
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    call :print_warning "GPU diagnostics failed, but CPU fallback should work"
)

call :print_status "Running basic functionality test..."
mvn test -Dtest=GpuBasicTest >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    call :print_warning "Some tests failed, check %LOG_FILE% for details"
) else (
    call :print_success "All basic tests passed"
)
goto :eof

REM Main execution
:main
call :detect_system
call :check_java
call :check_maven
call :check_cmake

call :install_dependencies
call :build_native_library
call :build_java_components
call :run_verification

call :print_step "Setup Complete"
call :print_success "OpenNLP GPU Extension setup completed successfully!"
call :print_status "You can now run: mvn exec:java -Dexec.mainClass=\"org.apache.opennlp.gpu.examples.GpuDemo\""

if "%WSL_AVAILABLE%"=="true" (
    echo.
    call :print_status "💡 Tip: For better GPU support, consider using WSL with Linux GPU drivers"
    call :print_status "    See: https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute"
)

goto :eof

REM Error handling
:error_exit
call :print_error "Setup failed. Check logs:"
call :print_error "  Main log: %LOG_FILE%"
call :print_error "  Error log: %ERROR_LOG%"
echo.
call :print_status "For support, please check the troubleshooting guide:"
call :print_status "  docs/setup/SETUP_GUIDE.md"
exit /b 1

REM Entry point
call :main
