$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== DWM Process ==="
$dwm = Get-Process dwm -ErrorAction SilentlyContinue
if ($dwm) {
    Write-Host ("DWM PID: {0}  WorkingSet: {1:N0} MB" -f $dwm.Id, ($dwm.WorkingSet / 1MB))
} else {
    Write-Host "dwm.exe not found"
}

Write-Host ""
Write-Host "=== GPU Engine counters for dwm.exe (which GPU it runs on) ==="
$gs = Get-Counter -ListSet 'GPU Engine'
foreach ($ctr in $gs.Counter) {
    if ($ctr -like '*dwm*') {
        $sample = Get-Counter -Counter $ctr
        $name = $ctr.Split('\')[-1]
        Write-Host ("{0} : {1:N0}" -f $name, $sample.CounterSamples[0].CookedValue)
    }
}

Write-Host ""
Write-Host "=== GPU Process Memory for dwm.exe ==="
$pms = Get-Counter -ListSet 'GPU Process Memory'
foreach ($ctr in $pms.Counter) {
    if ($ctr -like '*dwm*') {
        $sample = Get-Counter -Counter $ctr
        $name = $ctr.Split('\')[-1]
        Write-Host ("{0} : {1:N0} bytes ({2:N1} MB)" -f $name, $sample.CounterSamples[0].CookedValue, ($sample.CounterSamples[0].CookedValue / 1MB))
    }
}

Write-Host ""
Write-Host "=== HAGS (Hardware Accelerated GPU Scheduling) ==="
$key = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
if (Test-Path $key) {
    $v = Get-ItemProperty -Path $key -Name HwSchMode -ErrorAction SilentlyContinue
    if ($v) {
        Write-Host ("HwSchMode = {0}  (2=Enabled, 1=Disabled)" -f $v.HwSchMode)
    } else {
        Write-Host "HwSchMode not present"
    }
}

Write-Host ""
Write-Host "=== DWM visual effects registry ==="
$dkey = 'HKCU:\Software\Microsoft\Windows\DWM'
if (Test-Path $dkey) {
    $d = Get-ItemProperty -Path $dkey
    $d | Get-Member -MemberType NoteProperty | ForEach-Object {
        Write-Host ("{0} = {1}" -f $_.Name, $d.($_.Name))
    }
} else {
    Write-Host "DWM key not present"
}

Write-Host ""
Write-Host "=== Per-GPU utilization snapshot ==="
Get-Counter -ListSet 'GPU Adapter Memory' | Select-Object -ExpandProperty Counter | ForEach-Object {
    $s = Get-Counter -Counter $_ -ErrorAction SilentlyContinue
    if ($s) { Write-Host ("{0} : {1:N0} MB" -f $_.Split('\')[-1], ($s.CounterSamples[0].CookedValue / 1MB)) }
}
