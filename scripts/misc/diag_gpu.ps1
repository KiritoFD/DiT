$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== Video controllers with a live display ==="
Get-CimInstance Win32_VideoController | ForEach-Object {
    if ($_.ConfigManagerErrorCode -eq 0) {
        Write-Host ("Name: {0}" -f $_.Name)
        Write-Host ("  Status: {0}" -f $_.Status)
        Write-Host ("  VideoMode: {0}" -f $_.VideoModeDescription)
        Write-Host ("  RefreshRate: {0} Hz" -f $_.CurrentRefreshRate)
        Write-Host ("  DriverVersion: {0}" -f $_.DriverVersion)
        Write-Host ("  PNPDeviceID: {0}" -f $_.PNPDeviceID)
        Write-Host ("  InstalledDisplayDrivers: {0}" -f $_.InstalledDisplayDrivers)
    }
}

Write-Host ""
Write-Host "=== All video controllers (status) ==="
Get-CimInstance Win32_VideoController | ForEach-Object {
    Write-Host ("{0}  [ErrorCode={1}]" -f $_.Name, $_.ConfigManagerErrorCode)
}
