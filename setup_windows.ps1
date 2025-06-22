# OpenNLP GPU Extension - Windows PowerShell Setup Script
# Supports: Windows 10/11, PowerShell 5.1+, PowerShell Core 7+
# Compatible with: Local machines, Azure VMs, AWS EC2 Windows instances

param(
    [switch]$ForceInstall,
    [switch]$SkipGpuDrivers,
    [switch]$Verbose,
    [string]$JavaVersion = "21",
    [string]$CmakeVersion = "3.16"
)

# Set error action preference
$ErrorActionPreference = "Stop"
$VerbosePreference = if ($Verbose) { "Continue" } else { "SilentlyContinue" }

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"
$LogFile = Join-Path $LogDir "setup.log"
$ErrorLogFile = Join-Path $LogDir "setup-errors.log"

# Create logs directory
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Colors for output
$Colors = @{
    Red    = "Red"
    Green  = "Green"
    Yellow = "Yellow"
    Blue   = "Blue"
    Cyan   = "Cyan"
}

# Logging functions
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
    Add-Content -Path $LogFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') INFO: $Message"
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
    Add-Content -Path $LogFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') SUCCESS: $Message"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
    Add-Content -Path $LogFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') WARNING: $Message"
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
    Add-Content -Path $ErrorLogFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ERROR: $Message"
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor $Colors.Blue
    Add-Content -Path $LogFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') STEP: $Message"
}

# System detection
function Get-SystemInfo {
    Write-Step "Detecting System Configuration"
    
    $systemInfo = @{
        OS                = (Get-CimInstance Win32_OperatingSystem).Caption
        Version           = [System.Environment]::OSVersion.Version
        Architecture      = [System.Environment]::OSVersion.Platform
        WSLAvailable      = $false
        VisualStudio      = $false
        PowerShellVersion = $PSVersionTable.PSVersion
    }
    
    Write-Status "Detected OS: $($systemInfo.OS)"
    Write-Status "PowerShell Version: $($systemInfo.PowerShellVersion)"
    
    # Check for WSL
    try {
        $wslStatus = wsl --status 2>$null
        if ($LASTEXITCODE -eq 0) {
            $systemInfo.WSLAvailable = $true
            Write-Status "WSL detected and available"
        }
    }
    catch {
        Write-Status "WSL not available"
    }
    
    # Check for Visual Studio
    try {
        $clPath = Get-Command cl -ErrorAction SilentlyContinue
        if ($clPath) {
            $systemInfo.VisualStudio = $true
            Write-Status "Visual Studio compiler detected"
        }
        else {
            Write-Warning "Visual Studio compiler not found in PATH"
        }
    }
    catch {
        Write-Warning "Visual Studio compiler not found"
    }
    
    return $systemInfo
}

# GPU detection
function Get-GpuInfo {
    Write-Step "Detecting GPU Hardware"
    
    $gpuInfo = @{
        Type    = "cpu_only"
        Devices = @()
        HasCuda = $false
        HasRocm = $false
    }
    
    # Check for NVIDIA GPU
    try {
        $nvidiaOutput = nvidia-smi --query-gpu=name --format=csv, noheader, nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvidiaOutput) {
            $gpuInfo.Type = "nvidia"
            $gpuInfo.HasCuda = $true
            $gpuInfo.Devices = $nvidiaOutput -split "`n" | Where-Object { $_.Trim() -ne "" }
            Write-Success "NVIDIA GPU detected"
            foreach ($gpu in $gpuInfo.Devices) {
                Write-Status "GPU: $($gpu.Trim())"
            }
        }
    }
    catch {
        # Continue to check for AMD
    }
    
    # Check for AMD GPU if no NVIDIA found
    if ($gpuInfo.Type -eq "cpu_only") {
        try {
            $rocmOutput = rocm-smi 2>$null
            if ($LASTEXITCODE -eq 0) {
                $gpuInfo.Type = "amd"
                $gpuInfo.HasRocm = $true
                Write-Success "AMD GPU with ROCm detected"
            }
            else {
                # Check for AMD GPU without ROCm
                $amdGpus = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*Radeon*" -or $_.Name -like "*AMD*" }
                if ($amdGpus) {
                    $gpuInfo.Type = "amd_no_rocm"
                    $gpuInfo.Devices = $amdGpus | ForEach-Object { $_.Name }
                    Write-Warning "AMD GPU detected but ROCm not installed"
                    foreach ($gpu in $gpuInfo.Devices) {
                        Write-Status "GPU: $gpu"
                    }
                }
            }
        }
        catch {
            # No AMD GPU or ROCm
        }
    }
    
    if ($gpuInfo.Type -eq "cpu_only") {
        Write-Warning "No supported GPU detected, will use CPU-only mode"
    }
    
    return $gpuInfo
}

# Check software requirements
function Test-JavaInstallation {
    Write-Step "Checking Java Installation"
    
    try {
        $javaVersion = java -version 2>&1 | Select-String "version" | ForEach-Object { $_.ToString() }
        if ($javaVersion -match '"([^"]+)"') {
            $version = $matches[1]
            Write-Status "Java found: $version"
            
            # Parse major version
            if ($version -match '^(\d+)\.') {
                $majorVersion = [int]$matches[1]
            }
            elseif ($version -match '^(\d+)') {
                $majorVersion = [int]$matches[1]
            }
            
            if ($majorVersion -ge 11) {
                Write-Success "Java version is adequate"
                return $true
            }
            else {
                Write-Warning "Java version is too old, need Java 11+"
                return $false
            }
        }
    }
    catch {
        Write-Warning "Java not found"
        return $false
    }
}

function Test-MavenInstallation {
    Write-Step "Checking Maven Installation"
    
    try {
        $mavenVersion = mvn -version 2>&1 | Select-String "Apache Maven" | ForEach-Object { $_.ToString() }
        if ($mavenVersion) {
            Write-Success "Maven found: $($mavenVersion.Split()[2])"
            return $true
        }
    }
    catch {
        Write-Warning "Maven not found"
        return $false
    }
}

function Test-CmakeInstallation {
    Write-Step "Checking CMake Installation"
    
    try {
        $cmakeVersion = cmake --version 2>&1 | Select-String "cmake version" | ForEach-Object { $_.ToString() }
        if ($cmakeVersion) {
            Write-Success "CMake found: $($cmakeVersion.Split()[2])"
            return $true
        }
    }
    catch {
        Write-Warning "CMake not found"
        return $false
    }
}

# Installation functions
function Install-Prerequisites {
    Write-Step "Installing Prerequisites"
    
    $hasJava = Test-JavaInstallation
    $hasMaven = Test-MavenInstallation
    $hasCmake = Test-CmakeInstallation
    
    $missingTools = @()
    if (-not $hasJava) { $missingTools += "Java" }
    if (-not $hasMaven) { $missingTools += "Maven" }
    if (-not $hasCmake) { $missingTools += "CMake" }
    
    if ($missingTools.Count -gt 0) {
        Write-Warning "Missing required tools: $($missingTools -join ', ')"
        
        # Check for Chocolatey
        try {
            $chocoVersion = choco --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Status "Chocolatey found, can install missing tools automatically"
                
                if ($ForceInstall) {
                    if (-not $hasJava) {
                        Write-Status "Installing Java via Chocolatey..."
                        choco install openjdk -y
                    }
                    if (-not $hasMaven) {
                        Write-Status "Installing Maven via Chocolatey..."
                        choco install maven -y
                    }
                    if (-not $hasCmake) {
                        Write-Status "Installing CMake via Chocolatey..."
                        choco install cmake -y
                    }
                }
                else {
                    Write-Status "To auto-install missing tools, run with -ForceInstall flag"
                    Write-Status "Or install manually:"
                    if (-not $hasJava) { Write-Status "  choco install openjdk" }
                    if (-not $hasMaven) { Write-Status "  choco install maven" }
                    if (-not $hasCmake) { Write-Status "  choco install cmake" }
                }
            }
            else {
                Write-Status "Install Chocolatey for easy package management:"
                Write-Status "  Set-ExecutionPolicy Bypass -Scope Process -Force"
                Write-Status "  [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072"
                Write-Status "  iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
            }
        }
        catch {
            Write-Status "Manual installation required:"
            Write-Status "  Java: https://adoptium.net/"
            Write-Status "  Maven: https://maven.apache.org/"
            Write-Status "  CMake: https://cmake.org/"
        }
        
        if (-not $ForceInstall) {
            throw "Required tools missing. Install them and run this script again, or use -ForceInstall flag."
        }
    }
}

# Build functions
function Build-NativeLibrary {
    param($GpuInfo)
    
    Write-Step "Building Native C++ Library"
    
    $cppDir = Join-Path $ScriptDir "src\main\cpp"
    if (-not (Test-Path $cppDir)) {
        throw "Cannot find C++ source directory: $cppDir"
    }
    
    Push-Location $cppDir
    try {
        # Clean previous builds
        $cleanTargets = @("CMakeCache.txt", "CMakeFiles", "Makefile", "build", "*.dll", "*.lib", "*.pdb")
        foreach ($target in $cleanTargets) {
            if (Test-Path $target) {
                Remove-Item $target -Recurse -Force
            }
        }
        
        # Configure with CMake
        Write-Status "Configuring build with CMake..."
        
        $cmakeArgs = @()
        if ($GpuInfo.Type -eq "nvidia") {
            $cmakeArgs += "-DUSE_CUDA=ON"
        }
        elseif ($GpuInfo.Type -eq "amd") {
            $cmakeArgs += "-DUSE_ROCM=ON"
        }
        
        # Detect Visual Studio version
        $vsGenerator = ""
        if (Get-Command "cl" -ErrorAction SilentlyContinue) {
            $vsVersion = cl 2>&1 | Select-String "Version (\d+)\." | ForEach-Object { $_.Matches.Groups[1].Value }
            switch ($vsVersion) {
                "19" { $vsGenerator = "Visual Studio 16 2019" }
                "17" { $vsGenerator = "Visual Studio 17 2022" }
                default { $vsGenerator = "Visual Studio 17 2022" }
            }
            $cmakeArgs += "-G", $vsGenerator
        }
        
        Write-Status "Running: cmake . $($cmakeArgs -join ' ')"
        & cmake . @cmakeArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "CMake configuration failed, trying CPU-only build..."
            & cmake . -G $vsGenerator -DUSE_CPU_ONLY=ON 2>&1 | Tee-Object -FilePath $LogFile -Append
            
            if ($LASTEXITCODE -ne 0) {
                throw "CMake configuration failed completely"
            }
        }
        
        # Build
        Write-Status "Building native library..."
        & cmake --build . --config Release 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            throw "Native library build failed"
        }
        
        # Verify build
        $dllPath = "Release\opennlp_gpu.dll"
        if (-not (Test-Path $dllPath)) {
            $dllPath = "opennlp_gpu.dll"
        }
        
        if (Test-Path $dllPath) {
            Write-Success "Native library built successfully"
            
            # Copy to Java resources
            $resourceDir = Join-Path $ScriptDir "src\main\resources\native\windows\x86_64"
            if (-not (Test-Path $resourceDir)) {
                New-Item -ItemType Directory -Path $resourceDir -Force | Out-Null
            }
            
            Copy-Item $dllPath $resourceDir -Force
            Write-Success "Native library copied to resources"
        }
        else {
            throw "Native library build verification failed"
        }
    }
    finally {
        Pop-Location
    }
}

function Build-JavaComponents {
    Write-Step "Building Java Components"
    
    Push-Location $ScriptDir
    try {
        Write-Status "Running Maven build..."
        & mvn clean compile 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            throw "Maven build failed"
        }
        
        Write-Success "Java components built successfully"
    }
    finally {
        Pop-Location
    }
}

function Test-Installation {
    Write-Step "Running Verification Tests"
    
    Push-Location $ScriptDir
    try {
        Write-Status "Testing GPU detection..."
        & mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics" 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "GPU diagnostics failed, but CPU fallback should work"
        }
        
        Write-Status "Running basic functionality test..."
        & mvn test -Dtest=GpuBasicTest 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Some tests failed, check $LogFile for details"
        }
        else {
            Write-Success "All basic tests passed"
        }
    }
    finally {
        Pop-Location
    }
}

# Main execution
function Main {
    try {
        Write-Status "Starting OpenNLP GPU Extension Setup for Windows..."
        Write-Status "Log file: $LogFile"
        
        $systemInfo = Get-SystemInfo
        $gpuInfo = Get-GpuInfo
        
        Install-Prerequisites
        Build-NativeLibrary -GpuInfo $gpuInfo
        Build-JavaComponents
        Test-Installation
        
        Write-Step "Setup Complete"
        Write-Success "OpenNLP GPU Extension setup completed successfully!"
        Write-Status "You can now run: mvn exec:java -Dexec.mainClass=`"org.apache.opennlp.gpu.examples.GpuDemo`""
        
        if ($systemInfo.WSLAvailable) {
            Write-Host "`n💡 Tip: For better GPU support, consider using WSL with Linux GPU drivers" -ForegroundColor $Colors.Cyan
            Write-Status "    See: https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute"
        }
        
    }
    catch {
        Write-Error "Setup failed: $($_.Exception.Message)"
        Write-Error "Check logs:"
        Write-Error "  Main log: $LogFile"
        Write-Error "  Error log: $ErrorLogFile"
        Write-Host "`nFor support, please check the troubleshooting guide:" -ForegroundColor $Colors.Yellow
        Write-Status "  docs\setup\SETUP_GUIDE.md"
        exit 1
    }
}

# Entry point
Main
