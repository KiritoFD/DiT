$ErrorActionPreference = 'SilentlyContinue'
foreach ($c in @('plink','putty','pscp')) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { Write-Host ("{0} -> {1}" -f $c, $cmd.Source) } else { Write-Host ("{0} -> NOT found" -f $c) }
}
# git-bash path
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    $gp = Split-Path (Split-Path $git.Source)
    $usrbin = Join-Path $gp "usr\bin"
    Write-Host ("git usr/bin: {0}" -f $usrbin)
    if (Test-Path $usrbin) {
        Get-ChildItem $usrbin -File | Where-Object { $_.Name -like 'ssh*' -or $_.Name -like 'plink*' } | ForEach-Object { Write-Host ("  {0}" -f $_.Name) }
    }
}
