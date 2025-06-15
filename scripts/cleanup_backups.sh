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
POM_BACKUPS=$(find . -name "pom.xml.backup.*" -type f | wc -l)
JAVA_BACKUPS=$(find . -name "*.java.backup*" -type f | wc -l)
BROKEN_FILES=$(find . -name "*.java.broken*" -type f | wc -l)
OTHER_BACKUPS=$(find . -name "*.backup*" -type f ! -name "pom.xml.backup.*" ! -name "*.java.backup*" | wc -l)

TOTAL_FILES=$((POM_BACKUPS + JAVA_BACKUPS + BROKEN_FILES + OTHER_BACKUPS))

echo -e "${YELLOW}📊 Found backup files:${NC}"
echo -e "  • POM backups: ${POM_BACKUPS}"
echo -e "  • Java backups: ${JAVA_BACKUPS}"
echo -e "  • Broken files: ${BROKEN_FILES}"
echo -e "  • Other backups: ${OTHER_BACKUPS}"
echo -e "  • Total: ${TOTAL_FILES}"

if [ $TOTAL_FILES -eq 0 ]; then
    echo -e "${GREEN}✅ No backup files found - project is already clean!${NC}"
    exit 0
fi

echo -e "${BLUE}🗑️  Removing backup files...${NC}"

# Remove POM backup files
if [ $POM_BACKUPS -gt 0 ]; then
    echo -e "${BLUE}  Removing POM backup files...${NC}"
    find . -name "pom.xml.backup.*" -type f -delete
    echo -e "${GREEN}  ✓ Removed ${POM_BACKUPS} POM backup files${NC}"
fi

# Remove Java backup files
if [ $JAVA_BACKUPS -gt 0 ]; then
    echo -e "${BLUE}  Removing Java backup files...${NC}"
    find . -name "*.java.backup*" -type f -delete
    echo -e "${GREEN}  ✓ Removed ${JAVA_BACKUPS} Java backup files${NC}"
fi

# Remove broken files
if [ $BROKEN_FILES -gt 0 ]; then
    echo -e "${BLUE}  Removing broken files...${NC}"
    find . -name "*.java.broken*" -type f -delete
    echo -e "${GREEN}  ✓ Removed ${BROKEN_FILES} broken files${NC}"
fi

# Remove other backup files
if [ $OTHER_BACKUPS -gt 0 ]; then
    echo -e "${BLUE}  Removing other backup files...${NC}"
    find . -name "*.backup*" -type f ! -name "pom.xml.backup.*" ! -name "*.java.backup*" -delete
    echo -e "${GREEN}  ✓ Removed ${OTHER_BACKUPS} other backup files${NC}"
fi

echo -e "${GREEN}🎉 Cleanup complete! Removed ${TOTAL_FILES} backup files.${NC}"

# Verify cleanup
REMAINING=$(find . -name "*.backup*" -o -name "*.broken*" | wc -l)
if [ $REMAINING -eq 0 ]; then
    echo -e "${GREEN}✅ Project is now clean - no backup files remaining.${NC}"
else
    echo -e "${YELLOW}⚠️  ${REMAINING} backup files still remain.${NC}"
fi
