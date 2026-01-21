#!/bin/bash
# Helper script to initialize git repository and prepare for GitHub push

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "ST2HE GitHub Setup"
echo "=================="
echo ""

# Check if already a git repo
if [ -d .git ]; then
    echo "✓ Git repository already initialized"
else
    echo "Initializing git repository..."
    git init
    echo "✓ Git repository initialized"
fi

# Check git config
if ! git config user.name > /dev/null 2>&1; then
    echo ""
    echo "Git user.name not configured. Please set it:"
    echo "  git config user.name 'Your Name'"
    echo "  git config user.email 'your.email@example.com'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Add all files
echo ""
echo "Adding files to git..."
git add .
echo "✓ Files added"

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "No changes to commit (everything already committed)"
else
    echo ""
    echo "Creating initial commit..."
    git commit -m "Initial commit: ST2HE inference and sample generation framework

- Inference module with ST2HEInference class
- Sample generation utilities
- Batch processing scripts
- Documentation and examples
- Configuration files"
    echo "✓ Initial commit created"
fi

echo ""
echo "=============================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub: https://github.com/new"
echo "2. DO NOT initialize with README, .gitignore, or license"
echo "3. Run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/ST2HE.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Or if using SSH:"
echo "   git remote add origin git@github.com:YOUR_USERNAME/ST2HE.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
