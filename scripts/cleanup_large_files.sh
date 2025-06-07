#!/bin/bash
# Script to clean up large files from Git history
# Use this to fix the current situation with java_pid1266980.hprof

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧹 OpenNLP GPU Large File Cleanup Tool${NC}"
echo -e "${YELLOW}⚠️  This will rewrite Git history to remove large files${NC}"

# Confirm with user
read -p "This will permanently remove large files from Git history. Continue? (y/N): " CONFIRM
if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

echo -e "${BLUE}🔍 Scanning for large files in repository...${NC}"

# Find all large files (>100MB) in the repository
LARGE_FILES=$(find . -type f -size +100M -not -path "./.git/*" 2>/dev/null)

if [ -n "$LARGE_FILES" ]; then
    echo -e "${RED}❌ Large files found:${NC}"
    echo "$LARGE_FILES" | while IFS= read -r file; do
        SIZE=$(du -h "$file" | cut -f1)
        echo -e "${RED}  $file: $SIZE${NC}"
    done
    
    echo -e "${BLUE}🔧 Removing large files from working directory...${NC}"
    echo "$LARGE_FILES" | while IFS= read -r file; do
        rm -f "$file"
        echo -e "${GREEN}  ✓ Removed: $file${NC}"
        
        # Add to .gitignore
        if ! grep -q "^$(basename "$file")$" .gitignore 2>/dev/null; then
            echo "$(basename "$file")" >> .gitignore
        fi
        
        # Add pattern based on extension
        case "$file" in
            *.hprof)
                if ! grep -q "*.hprof" .gitignore; then
                    echo "*.hprof" >> .gitignore
                fi
                ;;
            *.log)
                if ! grep -q "*.log" .gitignore; then
                    echo "*.log" >> .gitignore
                fi
                ;;
        esac
    done
fi

# Remove specific known large files from Git history
KNOWN_LARGE_FILES=(
    "java_pid1266980.hprof"
    "*.hprof"
    "*.log"
)

echo -e "${BLUE}🔧 Removing known large files from Git history...${NC}"

for pattern in "${KNOWN_LARGE_FILES[@]}"; do
    echo -e "${BLUE}  Processing pattern: $pattern${NC}"
    git filter-branch --force --index-filter "git rm --cached --ignore-unmatch '$pattern'" --prune-empty --tag-name-filter cat -- --all
done

# Clean up backup refs
echo -e "${BLUE}🧹 Cleaning up backup references...${NC}"
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# Add patterns to .gitignore
echo -e "${BLUE}📝 Updating .gitignore...${NC}"
cat >> .gitignore << 'EOF'

# Large files and heap dumps
*.hprof
*.log
java_pid*.hprof

# Temporary files
*.tmp
*.temp

# IDE files
.idea/
*.iml
.vscode/

# OS files
.DS_Store
Thumbs.db

EOF

# Stage and commit .gitignore changes
git add .gitignore
git commit -m "Update .gitignore to prevent large files"

# Force garbage collection
echo -e "${BLUE}🗑️  Force garbage collection...${NC}"
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo -e "${GREEN}✅ Large file cleanup completed!${NC}"
echo -e "${YELLOW}⚠️  Git history has been rewritten${NC}"
echo -e "${YELLOW}📝 Next steps:${NC}"
echo -e "${YELLOW}   1. Verify the cleanup: git log --oneline${NC}"
echo -e "${YELLOW}   2. Force push to remote: git push --force-with-lease origin main${NC}"
echo -e "${YELLOW}   3. Other collaborators will need to re-clone the repository${NC}"

# Show repository size
echo -e "${BLUE}📊 Repository size after cleanup:${NC}"
du -sh .git
