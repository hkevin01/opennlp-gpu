#!/bin/bash

# VSCode Extensions Installer
# Installs all required and recommended extensions for OpenNLP GPU development

set -e

echo "📦 Installing VSCode Extensions for OpenNLP GPU Development..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if VSCode is installed
if ! command -v code >/dev/null 2>&1; then
    print_error "VSCode not found. Please install VSCode first:"
    echo "  https://code.visualstudio.com/"
    exit 1
fi

print_status "VSCode found"

# Required extensions for Java development
REQUIRED_EXTENSIONS=(
    "redhat.java:Java Language Support"
    "vscjava.vscode-java-debug:Java Debugger" 
    "vscjava.vscode-java-test:Java Test Runner"
    "vscjava.vscode-maven:Maven for Java"
    "redhat.vscode-xml:XML Language Support"
    "vscjava.vscode-java-pack:Extension Pack for Java"
)

# Recommended extensions for better development experience
RECOMMENDED_EXTENSIONS=(
    "github.copilot:GitHub Copilot"
    "ms-vscode.vscode-json:JSON Language Features"
    "ms-vscode.test-adapter-converter:Test Explorer UI"
    "streetsidesoftware.code-spell-checker:Code Spell Checker"
    "eamodio.gitlens:GitLens"
    "ms-vscode.vscode-markdown:Markdown All in One"
)

# Optional extensions for advanced features
OPTIONAL_EXTENSIONS=(
    "ms-vscode.remote-containers:Dev Containers"
    "ms-azuretools.vscode-docker:Docker"
    "bradlc.vscode-tailwindcss:Tailwind CSS IntelliSense"
    "esbenp.prettier-vscode:Prettier Code Formatter"
)

install_extension() {
    local ext_id=$1
    local ext_name=$2
    
    echo "Installing $ext_name..."
    if code --install-extension "$ext_id" --force >/dev/null 2>&1; then
        print_status "$ext_name installed"
    else
        print_error "Failed to install $ext_name"
        return 1
    fi
}

# Install required extensions
echo ""
echo "🔧 Installing Required Extensions:"
for ext in "${REQUIRED_EXTENSIONS[@]}"; do
    IFS=':' read -r ext_id ext_name <<< "$ext"
    install_extension "$ext_id" "$ext_name"
done

# Install recommended extensions
echo ""
echo "💡 Installing Recommended Extensions:"
for ext in "${RECOMMENDED_EXTENSIONS[@]}"; do
    IFS=':' read -r ext_id ext_name <<< "$ext"
    if install_extension "$ext_id" "$ext_name"; then
        continue
    else
        print_warning "Skipping $ext_name (optional)"
    fi
done

# Ask about optional extensions
echo ""
read -p "Install optional extensions? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎨 Installing Optional Extensions:"
    for ext in "${OPTIONAL_EXTENSIONS[@]}"; do
        IFS=':' read -r ext_id ext_name <<< "$ext"
        install_extension "$ext_id" "$ext_name" || print_info "Skipped $ext_name"
    done
fi

# Create extensions.json for workspace recommendations
echo ""
echo "📝 Creating workspace extension recommendations..."

mkdir -p .vscode

cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "redhat.java",
        "vscjava.vscode-java-debug",
        "vscjava.vscode-java-test", 
        "vscjava.vscode-maven",
        "redhat.vscode-xml",
        "github.copilot",
        "ms-vscode.vscode-json",
        "streetsidesoftware.code-spell-checker",
        "eamodio.gitlens"
    ],
    "unwantedRecommendations": [
        "ms-vscode.vscode-typescript-next",
        "ms-vscode.node-debug2"
    ]
}
EOF

print_status "Created .vscode/extensions.json"

# Verify installation
echo ""
echo "🔍 Verifying Installation:"

INSTALLED_EXTENSIONS=$(code --list-extensions 2>/dev/null || echo "")

success_count=0
total_count=0

for ext in "${REQUIRED_EXTENSIONS[@]}"; do
    IFS=':' read -r ext_id ext_name <<< "$ext"
    total_count=$((total_count + 1))
    
    if echo "$INSTALLED_EXTENSIONS" | grep -q "$ext_id"; then
        print_status "$ext_name"
        success_count=$((success_count + 1))
    else
        print_error "$ext_name (not installed)"
    fi
done

echo ""
echo "📊 Installation Summary:"
print_info "Successfully installed: $success_count/$total_count required extensions"

if [ $success_count -eq $total_count ]; then
    print_status "All required extensions installed successfully!"
else
    print_warning "Some extensions failed to install"
    print_info "Try running this script again or install manually"
fi

echo ""
print_info "Next Steps:"
echo "  1. Restart VSCode"
echo "  2. Run: ./scripts/check_ide_setup.sh"
echo "  3. Open the project in VSCode"
echo "  4. Wait for Java Language Server to initialize"

echo ""
print_info "Extension installation complete!"
