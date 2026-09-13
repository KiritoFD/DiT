$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== Displays with active resolution (DWM composites here) ==="
Get-CimInstance Win32_VideoController | ForEach-Object {
    if ($_.CurrentHorizontalResolution -gt 0) {
        Write-Host ("{0}: {1}x{2} @ {3}Hz  storeDriver={4}" -f `
            $_.Name, $_.CurrentHorizontalResolution, $_.CurrentVerticalResolution, `
            $_.CurrentRefreshRate, $_.DriverVersion)
    }
}

Write-Host ""
Write-Host "=== Is the AMD iGPU driving any display? ==="
Get-CimInstance Win32_VideoController | ForEach-Object {
    if ($_.Name -like '*AMD*') {
        $has = $_.CurrentHorizontalResolution -gt 0
        Write-Host ("AMD active display: {0}" -f $has)
    }
}

Write-Host ""
Write-Host "=== NVIDIA driver store version decode ==="
$nv = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*NVIDIA*' }
if ($nv) {
    Write-Host ("NVIDIA store DriverVersion: {0}" -f $nv.DriverVersion)
    $parts = $nv.DriverVersion.Split('.')
    $minor = [int]$parts[1]
    $branch = if ($minor -eq 15) { "R560" } elseif ($minor -eq 16) { "R570" } elseif ($minor -eq 17) { "R580" } else { "unknown" }
    Write-Host ("Minor={0} -> heuristic branch {1}" -f $minor, $branch)
}
