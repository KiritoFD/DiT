$root = 'g:/GitHub/DiT'
Write-Host '=== Top-level dirs with size ==='
$files = Get-ChildItem $root -Recurse -File -ErrorAction SilentlyContinue
$byDir = $files | Group-Object { if ($_.DirectoryName -eq $root) { '(root)' } else { $_.DirectoryName.Substring($root.Length + 1).Split('\')[0] } }
$byDir | ForEach-Object {
    $sum = ($_.Group | Measure-Object Length -Sum).Sum
    '{0,12:N0} MB  {1}  ({2} files)' -f ($sum/1MB), $_.Name, $_.Count
} | Sort-Object -Descending

Write-Host ''
Write-Host '=== Top 15 individual largest files ==='
$files | Sort-Object Length -Descending | Select-Object -First 15 | ForEach-Object {
    '{0,10:N0} MB  {1}' -f ($_.Length/1MB), $_.FullName.Substring($root.Length+1)
}

Write-Host ''
Write-Host '=== Total size ==='
'{0:N0} MB total, {1} files' -f (($files | Measure-Object Length -Sum).Sum/1MB), $files.Count

Write-Host ''
Write-Host '=== Key dirs detail ==='
foreach ($d in @('train.csv','val.csv','dataset','MCCD','example_100','labels','results_lora','scratch','pretrained_models')) {
    $p = Join-Path $root $d
    if (Test-Path $p) {
        if ((Get-Item $p).PSIsContainer) {
            $c = Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue
            $s = ($c | Measure-Object Length -Sum).Sum
            '{0,12:N0} MB  {1}/  ({2} files)' -f ($s/1MB), $d, $c.Count
        } else {
            '{0,12:N0} MB  {1}  (file)' -f ((Get-Item $p).Length/1MB), $d
        }
    } else { "  MISSING  $d" }
}