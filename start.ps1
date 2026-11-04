# Starts the Engineering Test Bench dashboard.
#   .\start.ps1                 default profile (config\bench.yaml)
#   .\start.ps1 -Profile dev    cheap OpenRouter models (config\dev.yaml)
# If PowerShell refuses to run scripts on this machine, use start.cmd with the
# same arguments, or: powershell -ExecutionPolicy Bypass -File .\start.ps1
#
# Installs requirements.txt into .venv on first use and again whenever that
# file changes. The marker .venv\.bench-ready holds the hash of the
# requirements it was installed from. It is written only after pip succeeds,
# so an install that failed or was interrupted is retried on the next start
# instead of leaving a .venv that looks ready but is not.
param(
    [string]$Profile = "",
    [int]$Port = 8765,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$marker = Join-Path $PSScriptRoot ".venv\.bench-ready"
$reqHash = (Get-FileHash (Join-Path $PSScriptRoot "requirements.txt")).Hash
# A marker that is empty because a write was interrupted reads as "" and fails the match.
$ready = (Test-Path $python) -and (Test-Path $marker) -and ("$(Get-Content $marker -Raw)".Trim() -eq $reqHash)
if (-not $ready) {
    Write-Host "Installing requirements into .venv..."
    # A venv whose creation was interrupted before ensurepip finished has
    # python.exe but no pip, and pip install would fail on it forever.
    $pip = Join-Path $PSScriptRoot ".venv\Scripts\pip.exe"
    if ((Test-Path $python) -and -not (Test-Path $pip)) {
        Remove-Item .venv -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path $python)) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw "The Python launcher 'py' was not found. Install Python 3.13 from python.org (it includes the launcher), then run start again."
        }
        py -3.13 -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            Remove-Item .venv -Recurse -Force -ErrorAction SilentlyContinue
            throw "Could not create .venv with 'py -3.13'. Install Python 3.13, then run start again."
        }
    }
    & $python -m pip install --disable-pip-version-check -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed (exit $LASTEXITCODE). Check the network, then run start again to retry the install."
    }
    Set-Content -Path $marker -Value $reqHash
}

$arguments = @("-m", "bench", "--port", $Port)
if ($Profile) { $arguments += @("--profile", $Profile) }
if ($NoBrowser) { $arguments += "--no-browser" }
& $python @arguments
exit $LASTEXITCODE
