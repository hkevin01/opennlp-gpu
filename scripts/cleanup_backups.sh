#!/bin/bash
# Script to remove all backup files from the project

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧹 OpenNLP GPU Project Backup Cleanup Tool${NC}"

# Check if we're in the right directory
if [ ! -f "pom.xml" ]; then
    echo -e "${RED}❌ No pom.xml found. Please run from project root.${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Scanning for backup files...${NC}"

# Find and count different types of backup files
POM_BACKUPS=$(find . -name "pom.xml.backup.*" -type f 2>/dev/null | wc -l)
JAVA_BACKUPS=$(find . -name "*.java.backup*" -type f 2>/dev/null | wc -l)
BROKEN_FILES=$(find . -name "*.java.broken*" -type f 2>/dev/null | wc -l)
TMP_FILES=$(find . -name "*.tmp" -type f 2>/dev/null | wc -l)
SWAP_FILES=$(find . -name ".*.swp" -type f 2>/dev/null | wc -l)
EDITOR_BACKUPS=$(find . -name "*~" -type f 2>/dev/null | wc -l)
OTHER_BACKUPS=$(find . -name "*.backup*" -type f ! -name "pom.xml.backup.*" ! -name "*.java.backup*" 2>/dev/null | wc -l)

TOTAL_FILES=$((POM_BACKUPS + JAVA_BACKUPS + BROKEN_FILES + TMP_FILES + SWAP_FILES + EDITOR_BACKUPS + OTHER_BACKUPS))

echo -e "${YELLOW}📊 Found backup files:${NC}"
echo -e "  • POM backups: ${POM_BACKUPS}"
echo -e "  • Java backups: ${JAVA_BACKUPS}"
echo -e "  • Broken files: ${BROKEN_FILES}"
echo -e "  • Temp files: ${TMP_FILES}"
echo -e "  • Swap files: ${SWAP_FILES}"
echo -e "  • Editor backups: ${EDITOR_BACKUPS}"
echo -e "  • Other backups: ${OTHER_BACKUPS}"
echo -e "  • ${YELLOW}Total: ${TOTAL_FILES}${NC}"

if [ $TOTAL_FILES -eq 0 ]; then
    echo -e "${GREEN}✅ No backup files found - project is already clean!${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}⚠️  This will permanently delete ${TOTAL_FILES} backup files.${NC}"
read -p "Continue? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🚫 Cleanup cancelled by user.${NC}"
    exit 0
fi

echo -e "${BLUE}🗑️  Removing backup files...${NC}"

# Remove POM backup files
if [ $POM_BACKUPS -gt 0 ]; then
    echo -e "${BLUE}  Removing POM backup files...${NC}"
    find . -name "pom.xml.backup.*" -type f -delete 2>/dev/null
    echo -e "${GREEN}  ✓ Removed ${POM_BACKUPS} POM backup files${NC}"
fi

# Remove Java backup files
if [ $JAVA_BACKUPS -gt 0 ]; then
    echo -e "${BLUE}  Removing Java backup files...${NC}"
    find . -name "*.java.backup*" -type f -delete 2>/dev/null
    echo -e "${GREEN}  ✓ Removed ${JAVA_BACKUPS} Java backup files${NC}"
fi

# Remove broken files
if [ $BROKEN_FILES -gt 0 ]; then
    echo -e "${BLUE}  Removing broken files...${NC}"
    find . -name "*.java.broken*" -type f -delete 2>/dev/null
    echo -e "${GREEN}  ✓ Removed ${BROKEN_FILES} broken files${NC}"
fi

# Remove temp files
if [ $TMP_FILES -gt 0 ]; then
    echo -e "${BLUE}  Removing temp files...${NC}"
    find . -name "*.tmp" -type f -delete 2>/dev/null
    echo -e "${GREEN}  ✓ Removed ${TMP_FILES} temp files${NC}"
fi

# Remove swap files
if [ $SWAP_FILES -gt 0 ]; then
    echo -e "${BLUE}  Removing swap files...${NC}"
    find . -name ".*.swp" -type f -delete 2>/dev/null
    echo -e "${GREEN}  ✓ Removed ${SWAP_FILES} swap files${NC}"
fi

# Remove editor backup files
if [ $EDITOR_BACKUPS -gt 0 ]; then
    echo -e "${BLUE}  Removing editor backup files...${NC}"
    find . -name "*~" -type f -delete 2>/dev/null
    echo -e "${GREEN}  ✓ Removed ${EDITOR_BACKUPS} editor backup files${NC}"
fi

# Remove other backup files
if [ $OTHER_BACKUPS -gt 0 ]; then
    echo -e "${BLUE}  Removing other backup files...${NC}"
    find . -name "*.backup*" -type f ! -name "pom.xml.backup.*" ! -name "*.java.backup*" -delete 2>/dev/null
    echo -e "${GREEN}  ✓ Removed ${OTHER_BACKUPS} other backup files${NC}"
fi

echo -e "${GREEN}🎉 Cleanup complete! Removed ${TOTAL_FILES} backup files.${NC}"

# Verify cleanup
REMAINING=$(find . \( -name "*.backup*" -o -name "*.broken*" -o -name "*.tmp" -o -name ".*.swp" -o -name "*~" \) 2>/dev/null | wc -l)
if [ $REMAINING -eq 0 ]; then
    echo -e "${GREEN}✅ Project is now clean - no backup files remaining.${NC}"
else
    echo -e "${YELLOW}⚠️  ${REMAINING} backup files still remain.${NC}"
    echo -e "${BLUE}💡 Run 'find . -name \"*.backup*\" -o -name \"*.broken*\" -o -name \"*.tmp\"' to see what remains.${NC}"
fi

# Optional: Add to .gitignore
if [ -f ".gitignore" ] && ! grep -q "\.backup" .gitignore; then
    echo ""
    read -p "Add backup file patterns to .gitignore? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}📝 Adding backup patterns to .gitignore...${NC}"
        cat >> .gitignore << 'EOF'

# Backup and temporary files
*.backup*
*.broken*
*.tmp
.*.swp
*~
pom.xml.backup.*
*.java.backup*
*.java.broken*
EOF
        echo -e "${GREEN}✓ Added backup file patterns to .gitignore${NC}"
    fi
fi
