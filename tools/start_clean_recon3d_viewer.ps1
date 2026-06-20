param(
    [int]$ViewerPort = 8765,
    [switch]$NoOpen,
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$Ports = @($ViewerPort, 9090, 9876, 9091, 9877, 9111, 9899)

foreach ($Port in @($ViewerPort, 8765, 8766, 8767, 8768, 8769)) {
    $Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $Listener) {
        continue
    }
    try {
        Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/replay/stop" -TimeoutSec 1 | Out-Null
    } catch {
    }
}

Start-Sleep -Seconds 1

$TargetPids = @()
foreach ($Port in $Ports) {
    $Connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        Where-Object { $_.State -eq "Listen" }
    foreach ($Connection in $Connections) {
        if ($Connection.OwningProcess -and $Connection.OwningProcess -ne 0) {
            $TargetPids += [int]$Connection.OwningProcess
        }
    }
}

$TargetPids = $TargetPids | Sort-Object -Unique
foreach ($PidToStop in $TargetPids) {
    try {
        Stop-Process -Id $PidToStop -Force -ErrorAction Stop
    } catch {
        Write-Warning ("Could not stop PID {0}: {1}" -f $PidToStop, $_.Exception.Message)
    }
}

Start-Sleep -Seconds 1
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs") | Out-Null

$StdOut = Join-Path $ProjectRoot "logs\viewer_server.out.log"
$StdErr = Join-Path $ProjectRoot "logs\viewer_server.err.log"

$PythonExe = Join-Path $env:USERPROFILE ".conda\envs\breastseg\python.exe"
if (Test-Path $PythonExe) {
    $Exe = $PythonExe
    $Args = @(
        "-u",
        "tools\serve_recon3d_viewer.py",
        "--host",
        "127.0.0.1",
        "--port",
        [string]$ViewerPort,
        "--clean-existing"
    )
} else {
    $Exe = "conda"
    $Args = @(
        "run",
        "--no-capture-output",
        "-n",
        "breastseg",
        "python",
        "tools\serve_recon3d_viewer.py",
        "--host",
        "127.0.0.1",
        "--port",
        [string]$ViewerPort,
        "--clean-existing"
    )
}
if (-not $NoOpen) {
    $Args += "--open"
}

if ($Background) {
    $Process = Start-Process -FilePath $Exe `
        -ArgumentList $Args `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru

    Start-Sleep -Seconds 5

    try {
        $Status = (Invoke-WebRequest -Uri "http://127.0.0.1:$ViewerPort/viewer/live.html" -UseBasicParsing -TimeoutSec 5).StatusCode
        Write-Output ("Viewer ready: http://127.0.0.1:{0}/viewer/live.html status={1} pid={2}" -f $ViewerPort, $Status, $Process.Id)
    } catch {
        Write-Output ("Viewer process started pid={0}, but health check failed: {1}" -f $Process.Id, $_.Exception.Message)
    }
} else {
    Write-Output ("Starting viewer in foreground: http://127.0.0.1:{0}/viewer/live.html" -f $ViewerPort)
    & $Exe @Args
}
