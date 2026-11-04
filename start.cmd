@rem Starts the bench without changing the PowerShell execution policy.
@rem Takes the same arguments as start.ps1, e.g.  start.cmd -Profile dev
@powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
@rem A double-clicked window would close at once and lose the error message.
@if errorlevel 1 pause
