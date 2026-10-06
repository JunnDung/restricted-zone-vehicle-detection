param([string]$Python = "", [switch]$Cuda)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" } }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (-not $Python) {
        $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $bundledPython) { $Python = $bundledPython }
        elseif (Get-Command py -ErrorAction SilentlyContinue) { $Python = (& py -3.12 -c 'import sys; print(sys.executable)').Trim(); Check-Exit }
        else { throw 'Install Python 3.12, then run setup.ps1 -Python C:\path\python.exe' }
    }
    & $Python -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'; Check-Exit
    & $Python -m venv .venv; Check-Exit
}
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $projectPython -m ensurepip --upgrade; Check-Exit
& $projectPython -m pip install --upgrade pip; Check-Exit
$torchIndex = if ($Cuda) { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
& $projectPython -m pip install torch==2.11.0 torchvision==0.26.0 --index-url $torchIndex; Check-Exit
& $projectPython -m pip install -r requirements-lock.txt; Check-Exit
& $projectPython -m pip check; Check-Exit
& $projectPython -m unittest discover -s tests -v; Check-Exit
Write-Host 'Setup complete. Run run.cmd, then open http://127.0.0.1:8003'
