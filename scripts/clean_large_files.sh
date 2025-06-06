#!/bin/bash

# Script to identify and handle large files before pushing to GitHub
# This helps prevent "large file rejected" errors when pushing

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Large File Detector and Cleaner ===${NC}"

# GitHub file size limit in bytes (100MB)
GITHUB_LIMIT=104857600

# Find the large file(s) that caused the error
echo -e "${BLUE}Scanning for files larger than GitHub's limit (100MB)...${NC}"

# Find large files in the git history
LARGE_FILES_IN_HISTORY=$(git rev-list --objects --all | 
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | 
  awk '/^blob/ && $3 >= '$GITHUB_LIMIT' { print $4 }')

# Find large files in working directory that are staged or tracked
LARGE_FILES_CURRENT=$(find . -type f -not -path "*/\.git/*" -size +100M | 
  while read file; do
    if git check-ignore -q "$file"; then
      continue  # Skip ignored files
    fi
    if git ls-files --error-unmatch "$file" > /dev/null 2>&1; then
      echo "$file"  # File is tracked
    elif git ls-files --cached "$file" > /dev/null 2>&1; then
      echo "$file"  # File is staged
    fi
  done)

# Combine results
LARGE_FILES=$(echo -e "$LARGE_FILES_IN_HISTORY\n$LARGE_FILES_CURRENT" | sort -u | grep -v "^$")

if [ -z "$LARGE_FILES" ]; then
  echo -e "${GREEN}No large files found in the current branch.${NC}"
else
  echo -e "${RED}Found large files that exceed GitHub's 100MB limit:${NC}"
  echo "$LARGE_FILES" | while read file; do
    SIZE=$(du -h "$file" 2>/dev/null | cut -f1)
    if [ -z "$SIZE" ]; then
      echo -e "  ${YELLOW}$file${NC} (in history, not in working directory)"
    else
      echo -e "  ${YELLOW}$file${NC} (size: $SIZE)"
    fi
  done
  
  echo -e "\n${BLUE}Options to resolve:${NC}"
  echo -e "1. ${YELLOW}Remove large files from history${NC} (destructive, rewrites history)"
  echo -e "2. ${YELLOW}Add files to .gitignore${NC} (prevents future tracking)"
  echo -e "3. ${YELLOW}Set up Git LFS${NC} (allows tracking large files)"
  echo -e "4. ${YELLOW}Exit${NC} (take no action now)"
  
  read -p "Select option (1-4): " OPTION
  
  case $OPTION in
    1)
      echo -e "${YELLOW}WARNING: This will rewrite git history. All team members will need to re-clone.${NC}"
      read -p "Are you sure you want to proceed? (y/n): " CONFIRM
      if [[ $CONFIRM == "y" ]]; then
        for file in $LARGE_FILES; do
          echo -e "${BLUE}Removing $file from git history...${NC}"
          git filter-branch --force --index-filter "git rm -rf --cached --ignore-unmatch \"$file\"" --prune-empty --tag-name-filter cat -- --all
        done
        echo -e "${GREEN}Files removed from history. Force push required with: git push --force${NC}"
      fi
      ;;
    2)
      echo -e "${BLUE}Adding files to .gitignore...${NC}"
      for file in $LARGE_FILES; do
        echo "$file" >> .gitignore
        git rm --cached "$file" 2>/dev/null
      done
      echo "*.hprof" >> .gitignore  # Add pattern for Java heap dumps
      echo -e "${GREEN}Updated .gitignore. Commit this change before pushing.${NC}"
      ;;
    3)
      echo -e "${BLUE}Setting up Git LFS...${NC}"
      if ! command -v git-lfs &> /dev/null; then
        echo -e "${YELLOW}Git LFS not installed. Please install it first:${NC}"
        echo -e "Ubuntu: sudo apt-get install git-lfs"
        echo -e "MacOS: brew install git-lfs"
        echo -e "Windows: https://git-lfs.github.com/"
      else
        git lfs install
        for file in $LARGE_FILES; do
          # Extract file extension
          EXT="${file##*.}"
          if [ "$EXT" != "$file" ]; then
            echo -e "${BLUE}Tracking *.$EXT files with Git LFS${NC}"
            git lfs track "*.$EXT"
          else
            echo -e "${BLUE}Tracking $file specifically with Git LFS${NC}"
            git lfs track "$file"
          fi
        done
        git lfs track "*.hprof"  # Track Java heap dumps with LFS
        git add .gitattributes
        echo -e "${GREEN}Git LFS configured. Commit these changes before pushing.${NC}"
      fi
      ;;
    *)
      echo -e "${YELLOW}No action taken. Please resolve large files before pushing.${NC}"
      ;;
  esac
fi

# Update .gitignore to prevent future issues with heap dumps
if ! grep -q "*.hprof" .gitignore; then
  echo -e "\n${BLUE}Updating .gitignore to exclude Java heap dumps...${NC}"
  echo "# Java heap dumps" >> .gitignore
  echo "*.hprof" >> .gitignore
  echo -e "${GREEN}Updated .gitignore to exclude *.hprof files.${NC}"
fi

echo -e "\n${GREEN}Script completed. Run this script again before pushing if you encounter large file errors.${NC}"

# Make the script executable
chmod +x "$0"
