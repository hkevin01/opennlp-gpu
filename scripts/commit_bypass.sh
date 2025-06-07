#!/bin/bash
# Emergency commit script that bypasses pre-commit hooks
# Customized for OpenNLP GPU project
# Use only when pre-commit is broken and you need to commit urgently

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

PROJECT_NAME="OpenNLP GPU"
PROJECT_PREFIX="[GPU]"

# Default commit message if none provided
DEFAULT_COMMIT_MESSAGE="Emergency commit - bypass pre-commit hooks"

if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No commit message provided. Using default: '$DEFAULT_COMMIT_MESSAGE'${NC}"
    echo -e "${BLUE}Usage: $0 'commit message' [--fix-java] [--check-license] [--phase-tag] [--no-sync]${NC}"
    echo -e "${YELLOW}OpenNLP GPU Project Options:${NC}"
    echo -e "  --fix-java        Fix Java code formatting and imports"
    echo -e "  --check-license   Verify Apache license headers are present"
    echo -e "  --phase-tag       Auto-tag commit with current project phase"
    echo -e "  --gpu-test        Run quick GPU availability test before commit"
    echo -e "  --sync            Sync changes with remote after commit (default)"
    echo -e "  --no-sync         Skip syncing with remote repository"
    echo -e "\n${BLUE}Examples:${NC}"
    echo -e "${YELLOW}  $0 'Add matrix operations' --fix-java --phase-tag${NC}"
    echo -e "${YELLOW}  $0 'Fix memory leak in GPU provider' --check-license --no-sync${NC}"
    echo -e "${YELLOW}  $0  # Uses default commit message and syncs${NC}"
    
    COMMIT_MESSAGE="$DEFAULT_COMMIT_MESSAGE"
else
    COMMIT_MESSAGE="$1"
fi

FIX_JAVA=false
CHECK_LICENSE=false
PHASE_TAG=false
GPU_TEST=false
SYNC_REMOTE=true  # Changed to default to true

# Parse all arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix-java)
            FIX_JAVA=true
            shift
            ;;
        --check-license)
            CHECK_LICENSE=true
            shift
            ;;
        --phase-tag)
            PHASE_TAG=true
            shift
            ;;
        --gpu-test)
            GPU_TEST=true
            shift
            ;;
        --sync)
            SYNC_REMOTE=true
            shift
            ;;
        --no-sync)
            SYNC_REMOTE=false
            shift
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            shift
            ;;
    esac
done

echo -e "${PURPLE}🚀 $PROJECT_NAME Emergency Commit Tool${NC}"
echo -e "${YELLOW}⚠️  Bypassing pre-commit hooks for emergency commit${NC}"

# Check if we're in the right project directory
if [ ! -f "pom.xml" ] || ! grep -q "opennlp-gpu" pom.xml 2>/dev/null; then
    echo -e "${RED}❌ Error: Not in OpenNLP GPU project root directory${NC}"
    echo -e "${YELLOW}💡 Please run this script from the project root where pom.xml is located${NC}"
    exit 1
fi

# Quick GPU test if requested
if [ "$GPU_TEST" = true ]; then
    echo -e "${BLUE}🔍 Running quick GPU availability test...${NC}"
    
    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}  ✓ NVIDIA GPU detected${NC}"
    else
        echo -e "${YELLOW}  ⚠ NVIDIA GPU not detected or drivers not installed${NC}"
    fi
    
    # Check if OpenCL is available (basic check)
    if [ -d "/usr/lib/x86_64-linux-gnu" ] && ls /usr/lib/x86_64-linux-gnu/libOpenCL* &> /dev/null; then
        echo -e "${GREEN}  ✓ OpenCL libraries found${NC}"
    else
        echo -e "${YELLOW}  ⚠ OpenCL libraries not found${NC}"
    fi
fi

# Fix Java code formatting if requested
if [ "$FIX_JAVA" = true ]; then
    echo -e "${BLUE}☕ Fixing Java code formatting and imports...${NC}"

    # Get list of staged Java files
    JAVA_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.java$')

    if [ -n "$JAVA_FILES" ]; then
        echo "$JAVA_FILES" | while IFS= read -r file; do
            if [ -f "$file" ]; then
                # Remove trailing whitespace
                sed -i 's/[[:space:]]*$//' "$file"
                
                # Ensure consistent line endings
                sed -i -e '$a\' "$file"
                
                # Fix common Java formatting issues
                # Fix spacing around operators
                sed -i 's/\([^=!<>]\)=\([^=]\)/\1 = \2/g' "$file"
                sed -i 's/\([^=!<>]\)==\([^=]\)/\1 == \2/g' "$file"
                
                # Ensure proper spacing after commas
                sed -i 's/,\([^ ]\)/, \1/g' "$file"
                
                # Fix spacing around opening braces
                sed -i 's/){/) {/g' "$file"
                sed -i 's/else{/else {/g' "$file"
                
                echo -e "${GREEN}  ✓ Fixed: $file${NC}"
            fi
        done

        # Re-stage the fixed files
        echo "$JAVA_FILES" | xargs git add
        echo -e "${GREEN}✅ Java formatting fixed and files re-staged${NC}"
    else
        echo -e "${YELLOW}No staged Java files found${NC}"
    fi
fi

# Check Apache license headers if requested
if [ "$CHECK_LICENSE" = true ]; then
    echo -e "${BLUE}📜 Checking Apache license headers...${NC}"
    
    JAVA_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.java$')
    MISSING_LICENSE=false
    
    if [ -n "$JAVA_FILES" ]; then
        echo "$JAVA_FILES" | while IFS= read -r file; do
            if [ -f "$file" ] && ! grep -q "Licensed to the Apache Software Foundation" "$file"; then
                echo -e "${YELLOW}  ⚠ Missing license header: $file${NC}"
                MISSING_LICENSE=true
            fi
        done
        
        if [ "$MISSING_LICENSE" = true ]; then
            echo -e "${YELLOW}💡 Consider adding Apache license headers to new files${NC}"
            echo -e "${YELLOW}   Template available in existing project files${NC}"
        else
            echo -e "${GREEN}✅ All Java files have proper license headers${NC}"
        fi
    fi
fi

# Auto-tag with project phase if requested
if [ "$PHASE_TAG" = true ]; then
    echo -e "${BLUE}🏷️  Auto-tagging with project phase...${NC}"
    
    # Determine current phase based on file changes
    STAGED_FILES=$(git diff --cached --name-only)
    PHASE=""
    
    if echo "$STAGED_FILES" | grep -q "ml/.*\.java"; then
        PHASE="PHASE2-ML"
    elif echo "$STAGED_FILES" | grep -q "compute/.*\.java"; then
        PHASE="PHASE2-COMPUTE"
    elif echo "$STAGED_FILES" | grep -q "common/.*\.java"; then
        PHASE="PHASE2-COMMON"
    elif echo "$STAGED_FILES" | grep -q "test/.*\.java"; then
        PHASE="PHASE3-TEST"
    elif echo "$STAGED_FILES" | grep -q "docs/.*\.md"; then
        PHASE="PHASE3-DOCS"
    else
        PHASE="PHASE2-CORE"
    fi
    
    COMMIT_MESSAGE="$PROJECT_PREFIX [$PHASE] $COMMIT_MESSAGE"
    echo -e "${GREEN}  ✓ Tagged as: $PHASE${NC}"
fi

# Check for large files that might cause issues
echo -e "${BLUE}📦 Checking for large files...${NC}"

# Check if there are any staged changes
STAGED_CHANGES=$(git diff --cached --name-only)
if [ -z "$STAGED_CHANGES" ]; then
    echo -e "${YELLOW}⚠️  No staged changes found. Checking for unstaged changes...${NC}"
    
    # Check for unstaged changes
    UNSTAGED_CHANGES=$(git diff --name-only)
    UNTRACKED_FILES=$(git ls-files --others --exclude-standard)
    
    if [ -n "$UNSTAGED_CHANGES" ] || [ -n "$UNTRACKED_FILES" ]; then
        echo -e "${BLUE}📝 Found unstaged/untracked files. Checking for large files before staging...${NC}"
        
        # Check for large files before staging
        ALL_CHANGED_FILES=$(echo -e "$UNSTAGED_CHANGES\n$UNTRACKED_FILES" | grep -v "^$")
        LARGE_FILES_DETECTED=false
        if [ -n "$ALL_CHANGED_FILES" ]; then
            echo "$ALL_CHANGED_FILES" | while IFS= read -r file; do
                if [ -f "$file" ]; then
                    # Check file size (anything over 10MB)
                    SIZE_BYTES=$(stat -c%s "$file" 2>/dev/null || echo "0")
                    if [ "$SIZE_BYTES" -gt 10485760 ]; then
                        SIZE_HUMAN=$(du -h "$file" | cut -f1)
                        echo -e "${RED}❌ Large file detected: $file ($SIZE_HUMAN)${NC}"
                        
                        # Add to .gitignore automatically
                        echo -e "${BLUE}📝 Adding $file to .gitignore...${NC}"
                        echo "$file" >> .gitignore
                        echo -e "${GREEN}✅ Added $file to .gitignore${NC}"
                        
                        # Also add the file pattern if it's a common large file type
                        case "$file" in
                            *.hprof)
                                if ! grep -q "*.hprof" .gitignore; then
                                    echo "*.hprof" >> .gitignore
                                    echo -e "${GREEN}✅ Added *.hprof pattern to .gitignore${NC}"
                                fi
                                ;;
                            *.log)
                                if ! grep -q "*.log" .gitignore; then
                                    echo "*.log" >> .gitignore
                                    echo -e "${GREEN}✅ Added *.log pattern to .gitignore${NC}"
                                fi
                                ;;
                            *.jar)
                                if ! grep -q "*.jar" .gitignore; then
                                    echo "*.jar" >> .gitignore
                                    echo -e "${GREEN}✅ Added *.jar pattern to .gitignore${NC}"
                                fi
                                ;;
                            *.zip|*.tar.gz|*.tgz)
                                if ! grep -q "*.zip" .gitignore; then
                                    echo -e "*.zip\n*.tar.gz\n*.tgz" >> .gitignore
                                    echo -e "${GREEN}✅ Added archive patterns to .gitignore${NC}"
                                fi
                                ;;
                        esac
                        
                        LARGE_FILES_DETECTED=true
                    fi
                fi
            done
            
            # If large files were detected, stage .gitignore and remove large files from staging
            if [ "$LARGE_FILES_DETECTED" = true ]; then
                echo -e "${BLUE}📝 Staging updated .gitignore...${NC}"
                git add .gitignore
                
                # Remove large files from the list of files to be staged
                echo -e "${BLUE}📝 Filtering out large files from staging...${NC}"
                FILTERED_FILES=""
                echo "$ALL_CHANGED_FILES" | while IFS= read -r file; do
                    if [ -f "$file" ]; then
                        SIZE_BYTES=$(stat -c%s "$file" 2>/dev/null || echo "0")
                        if [ "$SIZE_BYTES" -le 10485760 ]; then
                            FILTERED_FILES="$FILTERED_FILES $file"
                        fi
                    fi
                done
                
                # Stage only the non-large files
                if [ -n "$FILTERED_FILES" ]; then
                    git add $FILTERED_FILES
                    echo -e "${GREEN}✅ Staged only files under 10MB${NC}"
                else
                    echo -e "${YELLOW}⚠️  Only .gitignore changes to commit${NC}"
                fi
            else
                # No large files, stage everything as normal
                git add -A
                echo -e "${GREEN}✅ All changes have been staged${NC}"
            fi
        else
            git add -A
            echo -e "${GREEN}✅ All changes have been staged${NC}"
        fi
        
        # Update staged files for final check
        STAGED_CHANGES=$(git diff --cached --name-only)
    else
        echo -e "${RED}❌ No changes to commit${NC}"
        exit 1
    fi
fi

# Final check for large files in staged changes (should be none now, but double-check)
LARGE_FILES=$(echo "$STAGED_CHANGES" | xargs -I {} sh -c 'if [ -f "{}" ]; then find "{}" -size +10M 2>/dev/null; fi')
if [ -n "$LARGE_FILES" ]; then
    echo -e "${RED}⚠️  Somehow large files are still in staging area:${NC}"
    echo "$LARGE_FILES" | while IFS= read -r file; do
        if [ -n "$file" ]; then
            SIZE=$(du -h "$file" | cut -f1)
            echo -e "${RED}  $file ($SIZE) - removing from staging${NC}"
            git reset HEAD "$file"
            echo "$file" >> .gitignore
        fi
    done
    git add .gitignore
    echo -e "${GREEN}✅ Large files removed from staging and added to .gitignore${NC}"
fi

# Check total size of staged changes
TOTAL_SIZE_BYTES=$(git diff --cached --name-only | xargs -I {} sh -c 'if [ -f "{}" ]; then stat -c%s "{}"; fi' 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
if [ "$TOTAL_SIZE_BYTES" -gt 10485760 ]; then  # Reduced to 10MB total (was 50MB)
    TOTAL_SIZE_HUMAN=$(echo "$TOTAL_SIZE_BYTES" | awk '{printf "%.1f MB", $1/1024/1024}')
    echo -e "${RED}❌ Large commit detected: $TOTAL_SIZE_HUMAN total${NC}"
    echo -e "${RED}❌ This is too large for a typical code commit${NC}"
    echo -e "${YELLOW}💡 Investigating large files...${NC}"
    
    # List all staged files with sizes
    echo -e "${BLUE}📋 Staged files breakdown:${NC}"
    git diff --cached --name-only | while IFS= read -r file; do
        if [ -f "$file" ]; then
            SIZE_BYTES=$(stat -c%s "$file" 2>/dev/null || echo "0")
            SIZE_HUMAN=$(du -h "$file" | cut -f1)
            if [ "$SIZE_BYTES" -gt 1048576 ]; then  # Files over 1MB
                echo -e "${RED}  $file: $SIZE_HUMAN (TOO LARGE)${NC}"
                # Auto-remove files over 1MB
                git reset HEAD "$file"
                echo "$file" >> .gitignore
                echo -e "${BLUE}    → Removed from staging and added to .gitignore${NC}"
            else
                echo -e "${GREEN}  $file: $SIZE_HUMAN${NC}"
            fi
        fi
    done
    
    # Update .gitignore and re-calculate
    git add .gitignore
    TOTAL_SIZE_BYTES=$(git diff --cached --name-only | xargs -I {} sh -c 'if [ -f "{}" ]; then stat -c%s "{}"; fi' 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
    TOTAL_SIZE_HUMAN=$(echo "$TOTAL_SIZE_BYTES" | awk '{printf "%.1f MB", $1/1024/1024}')
    echo -e "${GREEN}✅ New commit size after cleanup: $TOTAL_SIZE_HUMAN${NC}"
    
    if [ "$TOTAL_SIZE_BYTES" -gt 10485760 ]; then  # Still too large
        echo -e "${RED}❌ Commit is still too large after cleanup${NC}"
        echo -e "${YELLOW}💡 Please manually review and remove unnecessary files${NC}"
        echo -e "${YELLOW}💡 Consider committing in smaller chunks${NC}"
        exit 1
    fi
fi

# Enhanced check for common large file types that shouldn't be in git
echo -e "${BLUE}🔍 Scanning for inappropriate file types...${NC}"
INAPPROPRIATE_PATTERNS=(
    "\.class$"          # Java compiled classes
    "\.jar$"            # Java archives
    "\.war$"            # Web archives
    "\.ear$"            # Enterprise archives
    "\.zip$"            # Zip archives
    "\.tar\.gz$"        # Compressed archives
    "\.rar$"            # WinRAR archives
    "\.7z$"             # 7-Zip archives
    "\.iso$"            # Disk images
    "\.dmg$"            # Mac disk images
    "\.exe$"            # Windows executables
    "\.msi$"            # Windows installers
    "\.deb$"            # Debian packages
    "\.rpm$"            # RedHat packages
    "\.bin$"            # Binary files
    "\.so$"             # Shared libraries
    "\.dll$"            # Windows libraries
    "\.dylib$"          # Mac libraries
    "\.a$"              # Static libraries (but not .sha, .java, etc.)
    "\.lib$"            # Windows libraries
    "\.hprof$"          # Java heap dumps
    "\.log$"            # Log files (usually large)
    "\.out$"            # Output files
    "\.tmp$"            # Temporary files
    "\.cache$"          # Cache files
    "\.pid$"            # Process ID files
    "\.lock$"           # Lock files
    "^target/"          # Maven build directory
    "^build/"           # Gradle build directory
    "^\.idea/"          # IntelliJ IDEA files
    "^\.vscode/"        # VS Code files
    "\.iml$"            # IntelliJ module files
    "\.swp$"            # Vim swap files
    "\.swo$"            # Vim swap files
    "~$"                # Backup files
    "^\.DS_Store$"      # Mac system files
    "^Thumbs\.db$"      # Windows thumbnail cache
    "^node_modules/"    # Node.js dependencies
)

FOUND_INAPPROPRIATE=false
for pattern in "${INAPPROPRIATE_PATTERNS[@]}"; do
    MATCHING_FILES=$(git diff --cached --name-only | grep -E "$pattern" || true)
    if [ -n "$MATCHING_FILES" ]; then
        echo -e "${RED}❌ Inappropriate files found matching pattern: $pattern${NC}"
        echo "$MATCHING_FILES" | while IFS= read -r file; do
            if [ -n "$file" ]; then
                echo -e "${RED}  Removing: $file${NC}"
                git reset HEAD "$file"
                
                # Convert regex pattern back to gitignore pattern for common cases
                GITIGNORE_PATTERN=""
                case "$pattern" in
                    "\.class$") GITIGNORE_PATTERN="*.class" ;;
                    "\.jar$") GITIGNORE_PATTERN="*.jar" ;;
                    "\.war$") GITIGNORE_PATTERN="*.war" ;;
                    "\.ear$") GITIGNORE_PATTERN="*.ear" ;;
                    "\.zip$") GITIGNORE_PATTERN="*.zip" ;;
                    "\.tar\.gz$") GITIGNORE_PATTERN="*.tar.gz" ;;
                    "\.rar$") GITIGNORE_PATTERN="*.rar" ;;
                    "\.7z$") GITIGNORE_PATTERN="*.7z" ;;
                    "\.iso$") GITIGNORE_PATTERN="*.iso" ;;
                    "\.dmg$") GITIGNORE_PATTERN="*.dmg" ;;
                    "\.exe$") GITIGNORE_PATTERN="*.exe" ;;
                    "\.msi$") GITIGNORE_PATTERN="*.msi" ;;
                    "\.deb$") GITIGNORE_PATTERN="*.deb" ;;
                    "\.rpm$") GITIGNORE_PATTERN="*.rpm" ;;
                    "\.bin$") GITIGNORE_PATTERN="*.bin" ;;
                    "\.so$") GITIGNORE_PATTERN="*.so" ;;
                    "\.dll$") GITIGNORE_PATTERN="*.dll" ;;
                    "\.dylib$") GITIGNORE_PATTERN="*.dylib" ;;
                    "\.a$") GITIGNORE_PATTERN="*.a" ;;
                    "\.lib$") GITIGNORE_PATTERN="*.lib" ;;
                    "\.hprof$") GITIGNORE_PATTERN="*.hprof" ;;
                    "\.log$") GITIGNORE_PATTERN="*.log" ;;
                    "\.out$") GITIGNORE_PATTERN="*.out" ;;
                    "\.tmp$") GITIGNORE_PATTERN="*.tmp" ;;
                    "\.cache$") GITIGNORE_PATTERN="*.cache" ;;
                    "\.pid$") GITIGNORE_PATTERN="*.pid" ;;
                    "\.lock$") GITIGNORE_PATTERN="*.lock" ;;
                    "^target/") GITIGNORE_PATTERN="target/" ;;
                    "^build/") GITIGNORE_PATTERN="build/" ;;
                    "^\.idea/") GITIGNORE_PATTERN=".idea/" ;;
                    "^\.vscode/") GITIGNORE_PATTERN=".vscode/" ;;
                    "\.iml$") GITIGNORE_PATTERN="*.iml" ;;
                    "\.swp$") GITIGNORE_PATTERN="*.swp" ;;
                    "\.swo$") GITIGNORE_PATTERN="*.swo" ;;
                    "~$") GITIGNORE_PATTERN="*~" ;;
                    "^\.DS_Store$") GITIGNORE_PATTERN=".DS_Store" ;;
                    "^Thumbs\.db$") GITIGNORE_PATTERN="Thumbs.db" ;;
                    "^node_modules/") GITIGNORE_PATTERN="node_modules/" ;;
                    *) GITIGNORE_PATTERN="$file" ;;  # Fallback to exact filename
                esac
                
                # Add to .gitignore if not already there
                if [ -n "$GITIGNORE_PATTERN" ] && ! grep -q "^$GITIGNORE_PATTERN$" .gitignore 2>/dev/null; then
                    echo "$GITIGNORE_PATTERN" >> .gitignore
                fi
            fi
        done
        FOUND_INAPPROPRIATE=true
    fi
done

if [ "$FOUND_INAPPROPRIATE" = true ]; then
    git add .gitignore
    echo -e "${GREEN}✅ Inappropriate files removed and patterns added to .gitignore${NC}"
    
    # Re-stage remaining valid files
    echo -e "${BLUE}📝 Re-staging remaining valid files...${NC}"
    git add -A
    
    # Update staged changes list
    STAGED_CHANGES=$(git diff --cached --name-only)
fi

# Perform the commit
echo -e "${BLUE}💾 Performing emergency commit...${NC}"
git commit --no-verify -m "$COMMIT_MESSAGE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Emergency commit completed for $PROJECT_NAME${NC}"
    echo -e "${BLUE}📝 Commit message: '$COMMIT_MESSAGE'${NC}"
    
    # Sync with remote if requested
    if [ "$SYNC_REMOTE" = true ]; then
        echo -e "\n${BLUE}🔄 Syncing changes with remote...${NC}"
        
        # Get current branch
        CURRENT_BRANCH=$(git branch --show-current)
        
        # Check if remote tracking branch exists
        REMOTE_BRANCH=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            echo -e "${BLUE}📡 Pulling latest changes from $REMOTE_BRANCH...${NC}"
            git pull --rebase
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Successfully pulled and rebased${NC}"
            else
                echo -e "${YELLOW}⚠️  Pull/rebase had conflicts. Please resolve manually.${NC}"
                echo -e "${YELLOW}   Run: git status to see conflicts${NC}"
                echo -e "${YELLOW}   After resolving: git rebase --continue${NC}"
            fi
            
            echo -e "${BLUE}🚀 Pushing changes to remote...${NC}"
            git push
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Successfully pushed to remote${NC}"
            else
                echo -e "${RED}❌ Failed to push to remote${NC}"
                echo -e "${YELLOW}💡 You may need to force push if rebase changed history: git push --force-with-lease${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  No upstream branch set. Setting up tracking...${NC}"
            
            # Try to push and set upstream
            REMOTE_NAME=$(git remote | head -n1)
            if [ -n "$REMOTE_NAME" ]; then
                echo -e "${BLUE}🔗 Setting upstream to $REMOTE_NAME/$CURRENT_BRANCH${NC}"
                git push -u "$REMOTE_NAME" "$CURRENT_BRANCH"
                
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✅ Successfully pushed and set upstream${NC}"
                else
                    echo -e "${RED}❌ Failed to push and set upstream${NC}"
                fi
            else
                echo -e "${RED}❌ No remote configured. Cannot sync.${NC}"
                echo -e "${YELLOW}💡 Add a remote first: git remote add origin <url>${NC}"
            fi
        fi
    fi
    
    # Show helpful next steps
    echo -e "\n${PURPLE}📋 Next Steps:${NC}"
    echo -e "${YELLOW}1. Verify commit: git log --oneline -1${NC}"
    echo -e "${YELLOW}2. Run tests: mvn test${NC}"
    echo -e "${YELLOW}3. Check build: mvn compile${NC}"
    
    if [ "$FIX_JAVA" = false ]; then
        echo -e "${YELLOW}4. Fix pre-commit issues and amend if needed${NC}"
    fi
    
    # Project-specific reminders
    echo -e "\n${PURPLE}🔧 OpenNLP GPU Reminders:${NC}"
    echo -e "${YELLOW}• Update project progress docs if this completes a milestone${NC}"
    echo -e "${YELLOW}• Test on both CPU and GPU if hardware-related changes${NC}"
    echo -e "${YELLOW}• Update user guide if API changes were made${NC}"
    
    if [ "$SYNC_REMOTE" = true ]; then
        echo -e "${YELLOW}• Changes have been synced with remote repository${NC}"
        echo -e "${YELLOW}• Team members can now pull your latest changes${NC}"
    else
        echo -e "${YELLOW}• Remember to push changes: git push (or use --sync next time)${NC}"
    fi
    
else
    echo -e "${RED}❌ Commit failed${NC}"
    echo -e "${YELLOW}💡 Check for remaining issues and try again${NC}"
    exit 1
fi
