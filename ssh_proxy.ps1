# Use system ssh with Windows tailscaled socket as the ProxyCommand transport.
# This avoids the `tailscale ssh` wrapper which is broken/unsupported in parts on Windows.
$sock = "$env:LOCALAPPDATA\Tailscale\tailscaled.sock"
if (-not (Test-Path $sock)) { $sock = "npipe://./pipe/tailscale/tailscaled" }
"Using socket: $sock"
$proxy = "tailscale --socket=$sock ssh --"
$cmd = "ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=publickey -o BatchMode=yes " + "root@100.72.205.57 hostname"
"Trying plain ssh via tailscaled proxy to node..."
# Direct attempt: plain ssh to the node's Tailscale IP with ProxyCommand through tailscaled
ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$env:USERPROFILE\.ssh\known_hosts" -o ProxyCommand="cmd /c tailscale ssh --paranoid=false %h" root@100.72.205.57 "hostname; whoami" 2>&1
"EXIT=$LASTEXITCODE"
