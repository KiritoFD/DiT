$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== existing pub keys ==="
$sshDir = Join-Path $env:USERPROFILE ".ssh"
if (Test-Path $sshDir) {
    Get-ChildItem $sshDir -File | Where-Object { $_.Name -like "*.pub" } | ForEach-Object {
        Write-Host ("  {0}  ({1} bytes)" -f $_.Name, $_.Length)
    }
} else {
    Write-Host "  no .ssh directory"
}

# Generate key if none exists
$pub = Join-Path $sshDir "id_ed25519.pub"
if (-not (Test-Path $pub)) {
    Write-Host "=== generating ed25519 key ==="
    ssh-keygen -t ed25519 -N "" -f (Join-Path $sshDir "id_ed25519") -C "dwm-agent"
    Write-Host "key generated"
} else {
    Write-Host "=== key already exists, skip generate ==="
}

Write-Host "=== sshpass availability ==="
if (Get-Command sshpass -ErrorAction SilentlyContinue) {
    Write-Host "  sshpass present"
} else {
    Write-Host "  sshpass NOT present"
}
