$ports = @(36430, 36431, 36432)
foreach ($p in $ports) {
    $r = Test-NetConnection -ComputerName 10.176.54.17 -Port $p -InformationLevel Quiet
    Write-Host ("Port {0} : {1}" -f $p, $r)
}
