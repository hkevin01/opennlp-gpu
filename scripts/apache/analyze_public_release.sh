#!/bin/bash

# OpenNLP GPU - Public Release Cleanup Script
# Identifies files that should be excluded or cleaned up before public release

echo "🧹 OpenNLP GPU - Public Release Cleanup Analysis"
echo "================================================="
echo

# Files to definitely exclude from public release
echo "📋 FILES TO EXCLUDE FROM PUBLIC RELEASE:"
echo "========================================"

echo "🔒 Development/Internal Documentation:"
echo "  - docs/project/session_progress_summary.md (internal session tracking)"
echo "  - docs/project/project_progress_main.md (internal progress tracking)"
echo "  - docs/testing/test_plan_progress_main.md (internal test progress)"
echo "  - docs/overview/project_progress.md (duplicate internal docs)"
echo "  - docs/overview/project_plan.md (duplicate internal docs)"
echo "  - PROJECT_COMPLETION_REPORT.md (internal completion report)"
echo

echo "🔧 Development/Legacy Scripts:"
echo "  - scripts/session_progress_summary.sh (session tracking)"
echo "  - scripts/nuclear_cleanup.sh (dangerous cleanup)"
echo "  - scripts/commit_bypass.sh (git bypass)"
echo "  - scripts/cleanup_backups.sh (internal cleanup)"
echo "  - scripts/cleanup_large_files.sh (internal cleanup)"
echo "  - scripts/clean_large_files.sh (internal cleanup)"
echo "  - scripts/organize_docs.sh (internal doc organization)"
echo "  - scripts/legacy/ (entire directory)"
echo "  - scripts/git-tools/ (internal git utilities)"
echo "  - scripts/fixes/ (internal fix scripts)"
echo "  - scripts/development/ (development utilities)"
echo

echo "📝 Outdated/Redundant Documentation:"
echo "  - docs/development/lombok-*.md (all lombok-related docs - project no longer uses lombok)"
echo "  - docs/development/gradle-*.md (gradle-related docs - project uses Maven)"
echo "  - docs/apache/apache_proposal_email.md (old version, use apache_proposal_email_final.md)"
echo

echo "⚙️  IDE/Environment Specific:"
echo "  - .env (environment variables)"
echo "  - .copilot/config.yml (copilot configuration)"
echo "  - Some VSCode configs may be too developer-specific"
echo

echo "🗂️  Build/Temporary Artifacts:"
echo "  - build/ directory (CMake build artifacts)"
echo "  - target/ directory (Maven build artifacts)"
echo "  - test-output/ directory (test artifacts)"
echo "  - .jdk/ directory (downloaded JDK)"
echo

echo
echo "📋 FILES TO KEEP/CLEAN UP:"
echo "=========================="

echo "✅ Keep (Core User Documentation):"
echo "  - README.md"
echo "  - docs/setup/getting_started.md"
echo "  - docs/guides/user_guide.md"
echo "  - docs/setup/gpu_prerequisites_guide.md"
echo "  - docs/development/technical_architecture.md"
echo "  - docs/performance/performance_benchmarks.md"
echo "  - CONTRIBUTING.md"
echo "  - LICENSE"
echo

echo "✅ Keep (Apache Contribution):"
echo "  - docs/apache/apache_contribution_guide.md"
echo "  - docs/apache/apache_contribution_summary.md"
echo "  - docs/apache/apache_fork_instructions.md"
echo "  - docs/apache/apache_proposal_email_final.md"
echo "  - scripts/apache_contribution_assistant.sh"
echo "  - scripts/generate_proposal.sh"
echo "  - scripts/project_status_summary.sh"
echo

echo "✅ Keep (Essential Scripts):"
echo "  - scripts/check_gpu_prerequisites.sh"
echo "  - scripts/validate_java_environment.sh"
echo "  - scripts/fix_java_environment.sh"
echo "  - scripts/setup_vscode_java.sh"
echo "  - scripts/vscode_java_guard*.sh"
echo

echo "🔧 Clean/Review (Configuration):"
echo "  - .vscode/ (review for user-friendliness vs developer-specific settings)"
echo "  - .github/ (review workflows for public repo)"
echo "  - pom.xml (ensure no internal/private dependencies)"
echo

echo
echo "🚀 RECOMMENDED ACTIONS:"
echo "======================"
echo

echo "1. 📁 Create .gitignore entries for build artifacts:"
echo "   - Add /build/ to .gitignore"
echo "   - Add /target/ to .gitignore (if not already)"
echo "   - Add /test-output/ to .gitignore"
echo "   - Add /.jdk/ to .gitignore"
echo

echo "2. 🗑️  Remove internal development files:"
echo "   - Delete session/progress tracking docs"
echo "   - Delete legacy/development scripts"
echo "   - Delete lombok-related documentation"
echo "   - Delete gradle-related documentation"
echo

echo "3. 📝 Update documentation:"
echo "   - Review README.md for clarity and completeness"
echo "   - Ensure all code examples work"
echo "   - Update any placeholder URLs (github.com/yourusername/)"
echo

echo "4. ⚙️  Review configuration files:"
echo "   - Simplify .vscode/settings.json for general users"
echo "   - Review .github/workflows for public CI"
echo "   - Check pom.xml for any internal dependencies"
echo

echo "5. 🧪 Test clean installation:"
echo "   - Test fresh clone and build process"
echo "   - Verify all user-facing scripts work"
echo "   - Run GPU diagnostics on clean system"
echo

echo
echo "💡 NOTES:"
echo "========="
echo "• Keep apache_proposal_email_final.md (not apache_proposal_email.md)"
echo "• Keep comprehensive test suites (they demonstrate quality)"
echo "• Keep technical_architecture.md (shows engineering depth)"
echo "• Keep performance_benchmarks.md (proves value proposition)"
echo "• Consider creating CHANGELOG.md for version tracking"
echo "• Consider adding CODE_OF_CONDUCT.md for Apache compliance"
echo

# Optionally create the actual cleanup
read -p "🤔 Would you like to create a cleanup script to remove these files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📝 Creating public_release_cleanup.sh script..."
    
    cat > scripts/public_release_cleanup.sh << 'EOF'
#!/bin/bash

# OpenNLP GPU - Public Release Cleanup Script
# Removes internal development files before public release

set -e

echo "🧹 Cleaning up for public release..."

# Remove internal documentation
rm -f docs/project/session_progress_summary.md
rm -f docs/project/project_progress_main.md  
rm -f docs/testing/test_plan_progress_main.md
rm -f PROJECT_COMPLETION_REPORT.md
rm -rf docs/overview/

# Remove lombok-related docs (no longer used)
rm -f docs/development/lombok-*.md

# Remove gradle-related docs (using Maven)
rm -f docs/development/gradle-*.md

# Remove old proposal email (keep final version)
rm -f docs/apache/apache_proposal_email.md

# Remove development scripts
rm -f scripts/session_progress_summary.sh
rm -f scripts/nuclear_cleanup.sh
rm -f scripts/commit_bypass.sh
rm -f scripts/cleanup_*.sh
rm -f scripts/clean_large_files.sh
rm -f scripts/organize_docs.sh
rm -rf scripts/legacy/
rm -rf scripts/git-tools/
rm -rf scripts/fixes/
rm -rf scripts/development/

# Remove build artifacts
rm -rf build/
rm -rf target/
rm -rf test-output/
rm -rf .jdk/

# Remove environment files
rm -f .env
rm -rf .copilot/

echo "✅ Cleanup complete!"
echo ""
echo "📋 Manual tasks remaining:"
echo "  1. Review .vscode/ settings for public use"
echo "  2. Review .github/ workflows"
echo "  3. Update README.md placeholder URLs"
echo "  4. Test fresh clone and build"
echo "  5. Add CHANGELOG.md and CODE_OF_CONDUCT.md if needed"

EOF

    chmod +x scripts/public_release_cleanup.sh
    echo "✅ Created scripts/public_release_cleanup.sh"
    echo "📋 Review the script before running: ./scripts/public_release_cleanup.sh"
fi
