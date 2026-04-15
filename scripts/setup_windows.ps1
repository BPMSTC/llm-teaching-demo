$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot "llm-create-demo"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

Write-Host "Repository root: $repoRoot"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath"
    python -m venv $venvPath
} else {
    Write-Host "Using existing virtual environment at $venvPath"
}

. $activateScript

Write-Host "Bootstrapping Python environment..."
& $pythonExe (Join-Path $repoRoot "scripts\bootstrap_env.py")

Write-Host "Registering Jupyter kernel..."
& $pythonExe -m ipykernel install --user --name llm-create-demo --display-name "Python (llm-create-demo)"

Write-Host "Done. Start Jupyter with:"
Write-Host "  llm-create-demo\Scripts\Activate.ps1"
Write-Host "  jupyter lab"
