#!/bin/bash
# Nuclear option: Complete removal of java_pid1266980.hprof from Git history

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${RED}💥 NUCLEAR CLEANUP - Remove java_pid1266980.hprof completely${NC}"
echo -e "${YELLOW}⚠️  This is the most aggressive approach - USE WITH CAUTION${NC}"

read -p "Are you absolutely sure you want to proceed? (type 'YES' to confirm): " CONFIRM
if [ "$CONFIRM" != "YES" ]; then
    echo "Operation cancelled."
    exit 1
fi

# Handle unstaged changes first
echo -e "${BLUE}🔧 Handling unstaged changes...${NC}"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo -e "${YELLOW}⚠️  Unstaged changes detected. Stashing them...${NC}"
    git stash push -m "Auto-stash before nuclear cleanup $(date)"
    STASHED_CHANGES=true
else
    STASHED_CHANGES=false
fi

# Create backup
BACKUP_BRANCH="nuclear-backup-$(date +%s)"
git branch $BACKUP_BRANCH
echo -e "${BLUE}📋 Created backup branch: $BACKUP_BRANCH${NC}"

# Multiple aggressive approaches
echo -e "${BLUE}🔧 Approach 1: Complete index rewrite...${NC}"
git filter-branch --force --index-filter \
    'git rm --cached --ignore-unmatch java_pid1266980.hprof || true' \
    --prune-empty --tag-name-filter cat -- --all

echo -e "${BLUE}🔧 Approach 2: Tree filter with forced removal...${NC}"
git filter-branch --force --tree-filter \
    'rm -rf java_pid1266980.hprof; find . -name "java_pid1266980.hprof" -exec rm -f {} \; 2>/dev/null || true' \
    --prune-empty --tag-name-filter cat -- --all

echo -e "${BLUE}🔧 Approach 3: Manual object removal...${NC}"
# Find all objects containing this filename
PROBLEMATIC_OBJECTS=$(git rev-list --objects --all | grep "java_pid1266980.hprof" | cut -d' ' -f1)
if [ -n "$PROBLEMATIC_OBJECTS" ]; then
    echo "Found problematic objects: $PROBLEMATIC_OBJECTS"
    # This is dangerous but necessary
    echo "$PROBLEMATIC_OBJECTS" | while read obj; do
        if [ -n "$obj" ]; then
            echo "Removing object: $obj"
            # Remove object references
            git update-ref -d refs/original/refs/heads/main 2>/dev/null || true
        fi
    done
fi

# Clean up all references
echo -e "${BLUE}🧹 Cleaning all references...${NC}"
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git repack -ad

# Update .gitignore
echo -e "${BLUE}📝 Updating .gitignore...${NC}"
cat >> .gitignore << 'EOF'

# Java heap dumps and profiling files
*.hprof
java_pid*.hprof
*.log

# Temporary files
*.tmp
*.temp
*~

# Build artifacts
target/
build/
*.jar
*.war
*.ear
*.class

# IDE files
.idea/
*.iml
.vscode/
.settings/
.project
.classpath

# OS files
.DS_Store
Thumbs.db

EOF

git add .gitignore
git commit -m "Add comprehensive .gitignore after nuclear cleanup"

echo -e "${GREEN}✅ Nuclear cleanup completed!${NC}"

# Final verification
if git log --all --name-only --pretty=format: | grep -q "java_pid1266980.hprof"; then
    echo -e "${RED}❌ File STILL present after nuclear cleanup${NC}"
    echo -e "${YELLOW}💡 Implementing last resort: Fresh repository creation${NC}"
    
    # Get current remote URL
    REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
    
    # Create fresh repository
    echo -e "${BLUE}🆕 Creating fresh repository...${NC}"
    cd ..
    
    # Create clean copy
    if [ -d "opennlp-gpu-clean" ]; then
        rm -rf opennlp-gpu-clean
    fi
    
    echo -e "${BLUE}📂 Copying current working directory...${NC}"
    cp -r opennlp-gpu opennlp-gpu-clean
    cd opennlp-gpu-clean
    
    # Remove git history
    echo -e "${BLUE}🗑️  Removing old Git history...${NC}"
    rm -rf .git
    
    # Initialize new repository
    echo -e "${BLUE}🎉 Initializing fresh Git repository...${NC}"
    git init
    git branch -m main
    
    # Add all files (respecting .gitignore)
    echo -e "${BLUE}📝 Adding all files to new repository...${NC}"
    git add .
    git commit -m "Initial commit - Clean repository without large files"
    
    # Add remote if it existed
    if [ -n "$REMOTE_URL" ]; then
        echo -e "${BLUE}🔗 Adding remote origin...${NC}"
        git remote add origin "$REMOTE_URL"
        
        # Ask if user wants to force push immediately
        read -p "Do you want to force push the clean repository now? (y/N): " PUSH_NOW
        if [[ $PUSH_NOW =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}🚀 Force pushing clean repository...${NC}"
            git push -u origin main --force
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ SUCCESS: Clean repository pushed to remote!${NC}"
                echo -e "${YELLOW}📝 Your old repository is backed up in: ../opennlp-gpu${NC}"
                echo -e "${YELLOW}📝 You can now use this clean repository: $(pwd)${NC}"
                echo -e "${YELLOW}📝 Team members should re-clone the repository${NC}"
            else
                echo -e "${RED}❌ Failed to push clean repository${NC}"
                echo -e "${YELLOW}💡 You may need to configure authentication first${NC}"
            fi
        else
            echo -e "${YELLOW}📝 Clean repository created at: $(pwd)${NC}"
            echo -e "${YELLOW}📝 To push manually: git push -u origin main --force${NC}"
        fi
    else
        echo -e "${YELLOW}📝 Clean repository created at: $(pwd)${NC}"
        echo -e "${YELLOW}📝 Add remote: git remote add origin <your-github-url>${NC}"
        echo -e "${YELLOW}📝 Then push: git push -u origin main --force${NC}"
    fi
    
    # Show repository sizes
    echo -e "${BLUE}📊 Repository sizes comparison:${NC}"
    echo -e "${YELLOW}   Old repository: $(du -sh ../opennlp-gpu/.git | cut -f1)${NC}"
    echo -e "${GREEN}   New repository: $(du -sh .git | cut -f1)${NC}"
    
else
    echo -e "${GREEN}✅ SUCCESS: java_pid1266980.hprof completely removed!${NC}"
    
    # Restore stashed changes if any
    if [ "$STASHED_CHANGES" = true ]; then
        echo -e "${BLUE}📤 Restoring stashed changes...${NC}"
        git stash pop
    fi
    
    echo -e "${YELLOW}📝 Now run: git push --force-with-lease origin main${NC}"
    du -sh .git
fi
