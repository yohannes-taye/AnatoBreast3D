[CmdletBinding()]
param(
    [string]$CondaEnv = "breastseg",
    [string]$Out = "runs\recon3d_live",
    [string]$Config = "runs\_tmp_live.yaml",
    [string]$ViewerHost = "127.0.0.1",
    [int]$ViewerPort = 8765,
    [int]$RerunWebPort = 9090,
    [int]$RerunGrpcPort = 9876,
    [int]$ColorWidth = 1280,
    [int]$ColorHeight = 720,
    [switch]$OpenBrowser,
    [switch]$NoRerun,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$OutPath = Join-Path $Root $Out
$LogDir = Join-Path $OutPath "logs"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ViewerLog = Join-Path $LogDir "viewer_$Stamp.log"
$LiveLog = Join-Path $LogDir "live_recon_$Stamp.log"

function Quote-DisplayArg {
    param([string]$Value)
    if ($Value -notmatch "[\s`"]") {
        return $Value
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Join-DisplayCommand {
    param([string[]]$Parts)
    return (($Parts | ForEach-Object { Quote-DisplayArg $_ }) -join " ")
}

function ConvertTo-PowerShellArray {
    param([string[]]$Values)
    $quoted = $Values | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }
    return "@(" + ($quoted -join ", ") + ")"
}

function ConvertTo-PowerShellLiteral {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

$ViewerPythonArgs = @(
    "tools\serve_recon3d_viewer.py",
    "--host", $ViewerHost,
    "--port", [string]$ViewerPort
)
if ($OpenBrowser) {
    $ViewerPythonArgs += "--open"
}

$LivePythonArgs = @(
    "tools\live_recon.py",
    "--out", $Out,
    "--config", $Config,
    "--color-size", [string]$ColorWidth, [string]$ColorHeight,
    "--reuse-existing-sync-config",
    "--rerun-web-port", [string]$RerunWebPort,
    "--rerun-grpc-port", [string]$RerunGrpcPort
)
if ($NoRerun) {
    $LivePythonArgs += "--no-rerun"
}

$ViewerDisplay = Join-DisplayCommand (@("conda", "run", "-n", $CondaEnv, "python") + $ViewerPythonArgs)
$LiveDisplay = Join-DisplayCommand (@("conda", "run", "-n", $CondaEnv, "python") + $LivePythonArgs)
$ViewerUrl = "http://$ViewerHost`:$ViewerPort/viewer/live.html"
$RerunUrl = "http://localhost:$RerunWebPort`?url=rerun%2Bhttp://localhost:$RerunGrpcPort/proxy"

Write-Host "Recon3D live startup"
Write-Host "Project: $Root"
Write-Host "Viewer: $ViewerUrl"
Write-Host "Rerun:  $RerunUrl"
Write-Host "Logs:   $LogDir"
Write-Host ""
Write-Host "Viewer command:"
Write-Host "  $ViewerDisplay"
Write-Host "Live command:"
Write-Host "  $LiveDisplay"

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run only. No processes were started."
    exit 0
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ViewerRunner = Join-Path $LogDir "run_viewer_$Stamp.ps1"
$LiveRunner = Join-Path $LogDir "run_live_recon_$Stamp.ps1"
$RootLiteral = ConvertTo-PowerShellLiteral $Root
$ViewerLogLiteral = ConvertTo-PowerShellLiteral $ViewerLog
$LiveLogLiteral = ConvertTo-PowerShellLiteral $LiveLog
$ViewerArgsLiteral = ConvertTo-PowerShellArray (@("run", "-n", $CondaEnv, "python") + $ViewerPythonArgs)
$LiveArgsLiteral = ConvertTo-PowerShellArray (@("run", "-n", $CondaEnv, "python") + $LivePythonArgs)

$ViewerRunnerText = @"
`$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $RootLiteral
`$cmd = $ViewerArgsLiteral
Write-Host 'Starting Recon3D viewer/API server'
Write-Host 'Log: $ViewerLog'
Write-Host ('conda ' + (`$cmd -join ' '))
& conda @cmd 2>&1 | Tee-Object -FilePath $ViewerLogLiteral
Write-Host ''
Write-Host 'Viewer server exited. Press Enter to close this window.'
Read-Host | Out-Null
"@

$LiveRunnerText = @"
`$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $RootLiteral
`$cmd = $LiveArgsLiteral
Write-Host 'Starting Recon3D live reconstruction pipeline'
Write-Host 'Log: $LiveLog'
Write-Host ('conda ' + (`$cmd -join ' '))
& conda @cmd 2>&1 | Tee-Object -FilePath $LiveLogLiteral
Write-Host ''
Write-Host 'Live pipeline exited. Press Enter to close this window.'
Read-Host | Out-Null
"@

Set-Content -Path $ViewerRunner -Value $ViewerRunnerText -Encoding UTF8
Set-Content -Path $LiveRunner -Value $LiveRunnerText -Encoding UTF8

Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", $ViewerRunner) -WorkingDirectory $Root
Start-Sleep -Seconds 2
Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", $LiveRunner) -WorkingDirectory $Root

Write-Host ""
Write-Host "Started viewer and live pipeline in separate PowerShell windows."
Write-Host "Open: $ViewerUrl"
Write-Host "Logs:"
Write-Host "  $ViewerLog"
Write-Host "  $LiveLog"
