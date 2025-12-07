<#
Publish script for Windows PowerShell.
Usage: Open PowerShell, cd to project folder and run:
    .\publish_to_github.ps1

This script does:
 - checks for git
 - initializes a repo if needed
 - adds files (honors .gitignore)
 - commits and pushes to the provided remote

NOTE: You will be prompted for Git credentials if not using SSH keys.
#>

$repoPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Write-Host "Project path: $repoPath"

# Check git availability
try {
    git --version | Out-Null
} catch {
    Write-Error "Git was not found in PATH. Please install Git for Windows: https://git-scm.com/download/win"
    exit 1
}

Set-Location $repoPath

Write-Host "Initializing git (if not already a repo)..."
git init

Write-Host "Showing largest files (top 30) - verify no large models will be added:" 
Get-ChildItem -Recurse -File | Sort-Object Length -Descending | Select-Object FullName,@{Name='SizeMB';Expression={[math]::Round($_.Length/1MB,2)}} -First 30 | Format-Table -AutoSize

Write-Host "Adding files (will respect .gitignore)..."
git add .

Write-Host "Staged files (short status):"
git status --short

Write-Host "Commiting..."
git commit -m "Initial commit: add project code, README, .gitignore, requirements" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "No changes to commit or commit failed (this may be OK)."
}

Write-Host "Configuring remote and pushing..."
# remove origin if exists
git remote remove origin 2>$null
git remote add origin https://github.com/whichbear/Ercp-Navigation-System.git
git branch -M main

Write-Host "Pushing to origin/main... you may be prompted for credentials or PAT."
git push -u origin main

Write-Host "Done. If push failed due to large files in history, contact me and I can help clean the history or configure Git LFS."
