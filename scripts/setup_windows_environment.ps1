# OpenNLP GPU Windows Environment Setup Script
# PowerShell script for Windows-specific environment configuration

param(
    [switch]$InstallJava,
    [switch]$InstallMaven,
    [switch]$InstallGit,
    [switch]$Verbose
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Enable verbose output if requested
if ($Verbose) {
    $VerbosePreference = "Continue"
}

Write-Host "🌐 OpenNLP GPU - Windows Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Function to check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Function to check if Chocolatey is installed
function Test-Chocolatey {
    try {
        $null = Get-Command choco -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Function to install Chocolatey
function Install-Chocolatey {
    Write-Host "📦 Installing Chocolatey package manager..." -ForegroundColor Yellow
    
    if (-not (Test-Administrator)) {
        Write-Warning "Administrator privileges required to install Chocolatey"
        Write-Host "Please run this script as Administrator or install Chocolatey manually"
        return $false
    }
    
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Host "✅ Chocolatey installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Error "Failed to install Chocolatey: $_"
        return $false
    }
}

# Function to check Java installation
function Test-Java {
    try {
        $javaVersion = & java -version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Java found: $($javaVersion[0])" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "❌ Java not found" -ForegroundColor Red
        return $false
    }
    return $false
}

# Function to install Java
function Install-Java {
    Write-Host "☕ Installing Java 17..." -ForegroundColor Yellow
    
    if (-not (Test-Chocolatey)) {
        if (-not (Install-Chocolatey)) {
            return $false
        }
    }
    
    try {
        & choco install -y openjdk17
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Java 17 installed successfully" -ForegroundColor Green
            return $true
        }
        else {
            Write-Error "Failed to install Java via Chocolatey"
            return $false
        }
    }
    catch {
        Write-Error "Failed to install Java: $_"
        return $false
    }
}

# Function to check Maven installation
function Test-Maven {
    try {
        $mavenVersion = & mvn -version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Maven found: $($mavenVersion[0])" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "❌ Maven not found" -ForegroundColor Red
        return $false
    }
    return $false
}

# Function to install Maven
function Install-Maven {
    Write-Host "📦 Installing Maven..." -ForegroundColor Yellow
    
    if (-not (Test-Chocolatey)) {
        if (-not (Install-Chocolatey)) {
            return $false
        }
    }
    
    try {
        & choco install -y maven
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Maven installed successfully" -ForegroundColor Green
            return $true
        }
        else {
            Write-Error "Failed to install Maven via Chocolatey"
            return $false
        }
    }
    catch {
        Write-Error "Failed to install Maven: $_"
        return $false
    }
}

# Function to check Git installation
function Test-Git {
    try {
        $gitVersion = & git --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Git found: $gitVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "❌ Git not found" -ForegroundColor Red
        return $false
    }
    return $false
}

# Function to install Git
function Install-Git {
    Write-Host "📦 Installing Git..." -ForegroundColor Yellow
    
    if (-not (Test-Chocolatey)) {
        if (-not (Install-Chocolatey)) {
            return $false
        }
    }
    
    try {
        & choco install -y git
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Git installed successfully" -ForegroundColor Green
            return $true
        }
        else {
            Write-Error "Failed to install Git via Chocolatey"
            return $false
        }
    }
    catch {
        Write-Error "Failed to install Git: $_"
        return $false
    }
}

# Function to detect Windows version and architecture
function Get-WindowsInfo {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $arch = $env:PROCESSOR_ARCHITECTURE
    
    Write-Host "🖥️ System Information:" -ForegroundColor Cyan
    Write-Host "   OS: $($os.Caption)" -ForegroundColor White
    Write-Host "   Version: $($os.Version)" -ForegroundColor White
    Write-Host "   Architecture: $arch" -ForegroundColor White
    Write-Host "   Total Memory: $([math]::Round($os.TotalVisibleMemorySize / 1MB, 2)) GB" -ForegroundColor White
    Write-Host ""
}

# Function to check for GPU capabilities
function Test-GpuCapabilities {
    Write-Host "🔍 Checking GPU capabilities..." -ForegroundColor Yellow
    
    try {
        # Check for NVIDIA GPU
        $nvidiaGpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }
        if ($nvidiaGpu) {
            Write-Host "✅ NVIDIA GPU detected: $($nvidiaGpu.Name)" -ForegroundColor Green
        }
        
        # Check for AMD GPU
        $amdGpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" }
        if ($amdGpu) {
            Write-Host "✅ AMD GPU detected: $($amdGpu.Name)" -ForegroundColor Green
        }
        
        # Check for Intel GPU
        $intelGpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.Name -match "Intel" }
        if ($intelGpu) {
            Write-Host "✅ Intel GPU detected: $($intelGpu.Name)" -ForegroundColor Green
        }
        
        if (-not ($nvidiaGpu -or $amdGpu -or $intelGpu)) {
            Write-Host "ℹ️ No discrete GPU detected - CPU fallback will be used" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "⚠️ Could not detect GPU information: $_" -ForegroundColor Yellow
    }
}

# Main execution
try {
    # Display system information
    Get-WindowsInfo
    
    # Check administrator privileges
    if (-not (Test-Administrator)) {
        Write-Warning "Some operations may require administrator privileges"
    }
    
    # Check and install Java if needed or requested
    if ($InstallJava -or -not (Test-Java)) {
        if (-not (Install-Java)) {
            Write-Error "Java installation failed"
            exit 1
        }
    }
    
    # Check and install Maven if needed or requested
    if ($InstallMaven -or -not (Test-Maven)) {
        if (-not (Install-Maven)) {
            Write-Error "Maven installation failed"
            exit 1
        }
    }
    
    # Check and install Git if needed or requested
    if ($InstallGit -or -not (Test-Git)) {
        if (-not (Install-Git)) {
            Write-Error "Git installation failed"
            exit 1
        }
    }
    
    # Check GPU capabilities
    Test-GpuCapabilities
    
    Write-Host ""
    Write-Host "🎉 Windows environment setup completed!" -ForegroundColor Green
    Write-Host "✅ Your Windows system is ready for OpenNLP GPU" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Open a new PowerShell/Command Prompt to refresh environment variables" -ForegroundColor White
    Write-Host "  2. Navigate to the project directory" -ForegroundColor White
    Write-Host "  3. Run: mvn clean compile" -ForegroundColor White
    Write-Host "  4. Run: mvn exec:java -Dexec.mainClass=org.apache.opennlp.gpu.tools.GpuDiagnostics" -ForegroundColor White
}
catch {
    Write-Error "Setup failed: $_"
    exit 1
}
