#!/bin/bash

# Script to fix git reference issues
# Checks and repairs upstream tracking and remote references

# Color definitions for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Git Repository Reference Fixer ===${NC}"
echo -e "${BLUE}Checking git repository status...${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Display current branch and status
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${GREEN}Current branch: ${CURRENT_BRANCH}${NC}"

# Check remote configuration
echo -e "${BLUE}Checking remote configuration...${NC}"
REMOTES=$(git remote)

if [ -z "$REMOTES" ]; then
    echo -e "${YELLOW}No remotes found. Do you want to add a remote? (y/n)${NC}"
    read -r add_remote
    if [[ $add_remote == "y" ]]; then
        echo -e "${YELLOW}Enter remote name (usually 'origin'):${NC}"
        read -r remote_name
        echo -e "${YELLOW}Enter remote URL:${NC}"
        read -r remote_url
        git remote add "$remote_name" "$remote_url"
        echo -e "${GREEN}Remote $remote_name added.${NC}"
    fi
else
    echo -e "${GREEN}Found remotes: ${REMOTES}${NC}"
fi

# Check upstream tracking for current branch
echo -e "${BLUE}Checking upstream tracking for ${CURRENT_BRANCH}...${NC}"

TRACKING_BRANCH=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null)
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Current branch is not tracking any upstream branch.${NC}"
    
    if [ -n "$REMOTES" ]; then
        echo -e "${YELLOW}Available remotes:${NC}"
        echo "$REMOTES"
        echo -e "${YELLOW}Which remote do you want to use? (default: origin)${NC}"
        read -r selected_remote
        selected_remote=${selected_remote:-origin}
        
        # Check if remote branch exists
        git fetch "$selected_remote"
        REMOTE_BRANCHES=$(git ls-remote --heads "$selected_remote" | grep -F "$CURRENT_BRANCH" | wc -l)
        
        if [ "$REMOTE_BRANCHES" -gt 0 ]; then
            echo -e "${GREEN}Setting upstream branch to ${selected_remote}/${CURRENT_BRANCH}${NC}"
            git branch --set-upstream-to="${selected_remote}/${CURRENT_BRANCH}" "$CURRENT_BRANCH"
        else
            echo -e "${YELLOW}Remote branch ${selected_remote}/${CURRENT_BRANCH} doesn't exist yet.${NC}"
            echo -e "${YELLOW}Would you like to push and set upstream? (y/n)${NC}"
            read -r push_upstream
            if [[ $push_upstream == "y" ]]; then
                git push -u "$selected_remote" "$CURRENT_BRANCH"
            fi
        fi
    fi
else
    echo -e "${GREEN}Current branch is tracking: ${TRACKING_BRANCH}${NC}"
fi

# Check for stale references
echo -e "${BLUE}Checking for stale references...${NC}"
git fetch --prune

# Fix any broken references
echo -e "${BLUE}Repairing references...${NC}"
git gc --prune=now

echo -e "${BLUE}Checking remote URLs...${NC}"
git remote -v

echo -e "${GREEN}Git repository references have been checked and fixed.${NC}"
echo -e "${YELLOW}If you're still experiencing issues, consider running:${NC}"
echo -e "  git fetch --all"
echo -e "  git remote prune origin"
echo -e "  git fsck --full"

# Make script executable
chmod +x "$(dirname "$0")/fix_git_refs.sh"

echo -e "${GREEN}Script is now executable. Run it anytime with: ./scripts/fix_git_refs.sh${NC}"
