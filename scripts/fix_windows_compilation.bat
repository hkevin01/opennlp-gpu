@echo off
REM Quick Fix Script for Windows Compilation Issues
REM This script fixes common C++ compilation problems on Windows

echo 🔧 Fixing OpenNLP GPU Extension Windows Compilation Issues...

REM Fix 1: Clean build environment
echo 1. Cleaning build environment...
cd /d "%~dp0\src\main\cpp"
if exist CMakeCache.txt del /q CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles >nul 2>&1
if exist Makefile del /q Makefile >nul 2>&1
if exist build rmdir /s /q build >nul 2>&1
if exist *.dll del /q *.dll >nul 2>&1
if exist *.lib del /q *.lib >nul 2>&1

REM Fix 2: Verify math includes
echo 2. Checking math library includes...
findstr /m "include.*cmath" jni\GpuOperationsJNI.cpp >nul
if %errorlevel% neq 0 (
    echo    Adding missing math includes...
    powershell -Command "(Get-Content jni\GpuOperationsJNI.cpp) -replace '#include <vector>', '#include <vector>\n#include <cmath>\n#include <algorithm>' | Set-Content jni\GpuOperationsJNI.cpp"
)

REM Fix 3: Set Windows compilation environment
echo 3. Setting Windows compilation environment...
set CXXFLAGS=/O2 /D_USE_MATH_DEFINES /bigobj
set CFLAGS=/O2 /D_USE_MATH_DEFINES

REM Fix 4: Try Visual Studio build
echo 4. Attempting Visual Studio build...
cmake . -G "Visual Studio 17 2022" -DUSE_CPU_ONLY=ON
if %errorlevel% neq 0 (
    echo    Visual Studio 2022 not found, trying 2019...
    cmake . -G "Visual Studio 16 2019" -DUSE_CPU_ONLY=ON
    if %errorlevel% neq 0 (
        echo    Visual Studio not found, trying MinGW...
        cmake . -G "MinGW Makefiles" -DUSE_CPU_ONLY=ON
    )
)

if %errorlevel% neq 0 (
    echo ❌ CMake configuration failed
    echo Please ensure you have Visual Studio or MinGW installed
    pause
    exit /b 1
)

echo 5. Building with CMake...
cmake --build . --config Release

if exist "Release\opennlp_gpu.dll" (
    echo ✅ Build successful! DLL created in Release\opennlp_gpu.dll
    
    REM Copy to resources
    if not exist "%~dp0src\main\resources\native\windows\x86_64" mkdir "%~dp0src\main\resources\native\windows\x86_64" >nul 2>&1
    copy "Release\opennlp_gpu.dll" "%~dp0src\main\resources\native\windows\x86_64\" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ DLL copied to Java resources
    )
    
    echo ✅ Windows fix completed successfully!
) else if exist "opennlp_gpu.dll" (
    echo ✅ Build successful! DLL created as opennlp_gpu.dll
    
    REM Copy to resources
    if not exist "%~dp0src\main\resources\native\windows\x86_64" mkdir "%~dp0src\main\resources\native\windows\x86_64" >nul 2>&1
    copy "opennlp_gpu.dll" "%~dp0src\main\resources\native\windows\x86_64\" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ DLL copied to Java resources
    )
    
    echo ✅ Windows fix completed successfully!
) else (
    echo ❌ Build failed. Check output above for errors.
    echo Common issues:
    echo   - Missing Visual Studio or MinGW
    echo   - Missing Windows SDK
    echo   - Antivirus blocking compilation
    pause
    exit /b 1
)

cd /d "%~dp0"
echo.
echo 🎉 You can now run: mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.GpuDemo"
pause
