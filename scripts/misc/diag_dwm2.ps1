$ErrorActionPreference = 'SilentlyContinue'

$dwm = Get-Process dwm
$pidText = "pid_" + $dwm.Id

Write-Host ("DWM PID = {0}  (instance token: {1})" -f $dwm.Id, $pidText)
Write-Host ("WorkingSet = {0:N0} MB" -f ($dwm.WorkingSet/1MB))
Write-Host ""

Write-Host "=== GPU Process Memory (all instances) ==="
$pms = Get-Counter -ListSet 'GPU Process Memory'
foreach ($ctr in $pms.Counter) {
    $sample = Get-Counter -Counter $ctr -ErrorAction SilentlyContinue
    if ($sample) {
        $val = $sample.CounterSamples[0].CookedValue
        $inst = $ctr.Split('\')[2]   # instance part
        if ($val -gt 0) {
            Write-Host ("{0}  [{1}] : {2:N1} MB" -f $inst, $ctr.Split('\')[-1], ($val/1MB))
        }
    }
}

Write-Host ""
Write-Host "=== GPU Engine (all instances, non-zero) ==="
$gs = Get-Counter -ListSet 'GPU Engine'
foreach ($ctr in $gs.Counter) {
    $sample = Get-Counter -Counter $ctr -ErrorAction SilentlyContinue
    if ($sample) {
        $val = $sample.CounterSamples[0].CookedValue
        if ($val -gt 0) {
            Write-Host ("{0} : {1:N0}" -f $ctr, $val)
        }
    }
}

Write-Host ""
Write-Host "=== Dedicated GPU memory per adapter ==="
Get-Counter -ListSet 'GPU Adapter Memory' | Select-Object -ExpandProperty Counter | ForEach-Object {
    $s = Get-Counter -Counter $_ -ErrorAction SilentlyContinue
    if ($s) { Write-Host ("{0} : {1:N1} MB" -f $_.Split('\')[-1], ($s.CounterSamples[0].CookedValue/1MB)) }
}
