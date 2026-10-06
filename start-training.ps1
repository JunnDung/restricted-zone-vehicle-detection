param([switch]$Resume)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$trainScript = Join-Path $PSScriptRoot 'scripts\train_large.py'
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($trainScript) }
if ($existing) { Write-Host "Training already running: PID $($existing.ProcessId)"; exit 0 }
if (-not (Test-Path -LiteralPath 'datasets\large\manifest.json')) { throw 'Prepare datasets first: .venv\Scripts\python.exe scripts\prepare_large.py' }
New-Item -ItemType Directory -Force runtime | Out-Null
$arguments = '-u "' + $trainScript + '" --epochs 30 --batch 8 --imgsz 640 --backup-github'
if ($Resume) { $arguments += ' --resume' }
$process = Start-Process -FilePath $projectPython -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'runtime\training.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'runtime\training.stderr.log')
$process.Id | Set-Content runtime\training.pid
Write-Host "Training launched: PID $($process.Id). Progress: dashboard / Model and guide. Logs: runtime\training.*.log"
