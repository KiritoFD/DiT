$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== Applying DWM / HAGS / virtual-display tweaks ==="

# 1) Disable GameViewer Virtual Display Adapter (3rd display surface for DWM)
$dev = Get-PnpDevice -FriendlyName '*GameViewer*' -ErrorAction SilentlyContinue
if ($dev) {
    Disable-PnpDevice -InstanceId $dev.InstanceId -Confirm:$false
    Write-Host ("Disabled virtual display: {0}" -f $dev.InstanceId)
} else {
    Write-Host "GameViewer virtual display not found (already disabled or absent)."
}

# 2) Disable Hardware-Accelerated GPU Scheduling (HAGS)
$key = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }
Set-ItemProperty -Path $key -Name 'HwSchMode' -Value 1 -Type DWord
Write-Host "HAGS set to Disabled (HwSchMode=1). Reboot/reative required to take effect."

# 3) DWM visual effects: turn off transparency glass, Aero Peek; let thumbnails hibernate
$dkey = 'HKCU:\Software\Microsoft\Windows\DWM'
if (-not (Test-Path $dkey)) { New-Item -Path $dkey -Force | Out-Null }
Set-ItemProperty -Path $dkey -Name 'ColorizationGlassAttribute' -Value 0 -Type DWord
Set-ItemProperty -Path $dkey -Name 'EnableAeroPeek' -Value 0 -Type DWord
Set-ItemProperty -Path $dkey -Name 'AlwaysHibernateThumbnails' -Value 1 -Type DWord
Write-Host "DWM: GlassAttribute=0, AeroPeek=0, AlwaysHibernateThumbnails=1"

# 4) System performance: best performance (disable animations)
$sp = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\VisualEffects'
if (-not (Test-Path $sp)) { New-Item -Path $sp -Force | Out-Null }
Set-ItemProperty -Path $sp -Name 'VisualFXSetting' -Value 2 -Type DWord
Write-Host "VisualFXSetting=2 (adjust for best performance)"

Write-Host ""
Write-Host "Done. NOTE: HAGS change needs a reboot to fully apply; DWM reg changes apply on next logon / dwm restart."
