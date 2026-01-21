# Instructions to Push ST2HE to GitHub

## Step 1: Initialize Git Repository

```bash
cd /ix/yufeihuang/timothy/he2exp/st2he
git init
```

## Step 2: Configure Git (if not already done)

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Step 3: Add All Files

```bash
git add .
```

## Step 4: Make Initial Commit

```bash
git commit -m "Initial commit: ST2HE inference and sample generation framework"
```

## Step 5: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `ST2HE` (or your preferred name)
3. Description: "ST2HE: Virtual Histology from High-Resolution Spatial Transcriptomics"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 6: Add Remote and Push

After creating the repository on GitHub, you'll see instructions. Use these commands:

```bash
git remote add origin https://github.com/YOUR_USERNAME/ST2HE.git
git branch -M main
git push -u origin main
```

If you're using SSH instead of HTTPS:
```bash
git remote add origin git@github.com:YOUR_USERNAME/ST2HE.git
git branch -M main
git push -u origin main
```

## Alternative: Use the Script

We've also created a helper script `setup_github.sh` that automates steps 1-4.
