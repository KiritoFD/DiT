$ErrorActionPreference = 'SilentlyContinue'

$sshpass = Get-Command sshpass -ErrorAction SilentlyContinue
if ($sshpass) { Write-Host ("sshpass already at: {0}" -f $sshpass.Source); exit }

# Try scoop
if (Get-Command scoop -ErrorAction SilentlyContinue) {
    Write-Host "scoop found, installing sshpass..."
    scoop install sshpass
    if (Get-Command sshpass -ErrorAction SilentlyContinue) { Write-Host "installed via scoop"; exit }
}

# Try choco
if (Get-Command choco -ErrorAction SilentlyContinue) {
    Write-Host "choco found, installing sshpass..."
    choco install sshpass -y
    if (Get-Command sshpass -ErrorAction SilentlyContinue) { Write-Host "installed via choco"; exit }
}

# Try git-bash sshpass (not bundled usually). Fallback: download prepackaged binary.
$binDir = Join-Path $env:USERPROFILE "bin"
if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Path $binDir | Out-Null }
$url = "https://github.com/radio24/sshpass-windows/releases/download/v1.0.0/sshpass.exe"
$dest = Join-Path $binDir "sshpass.exe"
Write-Host ("Downloading sshpass from {0} ..." -f $url)
try {
    Invoke-WebRequest -Uri $url -OutFile $dest -ErrorAction Stop
    Write-Host ("Downloaded to {0}" -f $dest)
    # add to PATH for this session
    $env:PATH = $env:PATH + ";" + $binDir
    if (Test-Path $dest) { Write-Host "sshpass ready" }
} catch {
    Write-Host ("Download failed: {0}" -f $_.Exception.Message)
}
