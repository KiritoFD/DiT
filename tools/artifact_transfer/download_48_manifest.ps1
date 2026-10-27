param(
    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$root = [IO.Path]::GetFullPath([string]$manifest.local_root)
New-Item -ItemType Directory -Force -Path $root | Out-Null
if (-not $LogPath) {
    $LogPath = Join-Path $root "download.log"
}
$lockPath = Join-Path $root "download.lock"
$lock = $null
$writer = $null

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format o), $Message
    Add-Content -LiteralPath $LogPath -Value $line
    Write-Output $line
}

try {
    $lock = [IO.File]::Open($lockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $writer = [IO.StreamWriter]::new($lock)
    $writer.WriteLine($PID)
    $writer.Flush()

    foreach ($item in @($manifest.files | Sort-Object priority)) {
        $destination = Join-Path $root ([string]$item.relative)
        $destination = [IO.Path]::GetFullPath($destination)
        if (-not $destination.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
            throw "destination escapes local root: $destination"
        }
        $expected = [int64]$item.size
        if ((Test-Path -LiteralPath $destination) -and (Get-Item -LiteralPath $destination).Length -eq $expected) {
            Write-Log "SKIP $($item.label): already complete ($expected bytes)"
            continue
        }

        $drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($root).Substring(0, 1))
        if ($drive.Free -lt ($expected + 5GB)) {
            throw "insufficient free space before $($item.label): free=$($drive.Free), need=$($expected + 5GB)"
        }

        $parent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        $partial = "$destination.part"
        $batchPath = Join-Path $root ("sftp-{0}.txt" -f $PID)
        $remote = ([string]$item.remote).Replace('"', '\"')
        $local = $partial.Replace('\\', '/').Replace('"', '\"')
        Set-Content -LiteralPath $batchPath -Encoding ascii -Value ("reget `"{0}`" `"{1}`"" -f $remote, $local)
        Write-Log "START/RESUME $($item.label): $($item.remote) -> $destination ($expected bytes)"
        & sftp -b $batchPath -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 48
        $sftpExit = $LASTEXITCODE
        Remove-Item -LiteralPath $batchPath -Force -ErrorAction SilentlyContinue
        if ($sftpExit -ne 0) {
            throw "sftp reget failed for $($item.label), exit=$sftpExit; partial file retained"
        }
        $actual = (Get-Item -LiteralPath $partial).Length
        if ($actual -ne $expected) {
            throw "size mismatch for $($item.label): expected=$expected actual=$actual"
        }
        Move-Item -LiteralPath $partial -Destination $destination -Force

        $doneRemote = "$($item.remote).done"
        & ssh -o BatchMode=yes 48 "test -f '$doneRemote'"
        if ($LASTEXITCODE -eq 0) {
            & scp -o BatchMode=yes "48:$doneRemote" "$destination.done"
            if ($LASTEXITCODE -ne 0) {
                throw "failed to download done manifest for $($item.label)"
            }
        }
        Write-Log "DONE $($item.label): $actual bytes"
    }

    $completed = [ordered]@{
        completed_at = (Get-Date -Format o)
        manifest = [IO.Path]::GetFullPath($ManifestPath)
        root = $root
        total_bytes = [int64]$manifest.total_bytes
    }
    $completed | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root "completed.json")
    Write-Log "ALL DONE"
}
catch {
    Write-Log "FAILED: $($_.Exception.Message)"
    throw
}
finally {
    if ($null -ne $writer) { $writer.Dispose() }
    elseif ($null -ne $lock) { $lock.Dispose() }
    if (Test-Path -LiteralPath $lockPath) {
        Remove-Item -LiteralPath $lockPath -Force
    }
}
