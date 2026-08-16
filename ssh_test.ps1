$key = "$env:USERPROFILE\.ssh\id_ed25519"
"=== 1. TCP 22 via serve ==="
$t = Test-NetConnection -ComputerName 100.72.205.57 -Port 22 -WarningAction SilentlyContinue
"TCP 22: $($t.TcpTestSucceeded)"

"=== 2. plain ssh using serve TCP, accept-new host key ==="
ssh -v -o ConnectTimeout=20 `
    -o StrictHostKeyChecking=accept-new `
    -o UserKnownHostsFile="$env:USERPROFILE\.ssh\known_hosts" `
    -o BatchMode=yes `
    -o PreferredAuthentications=publickey `
    -i $key `
    -p 22 root@100.72.205.57 "hostname; whoami" 2>&1

"EXIT=$LASTEXITCODE"
