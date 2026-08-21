$kh = "$env:USERPROFILE\.ssh\known_hosts"
# Remove any stale entry for the node so Tailscale can re-verify via coordination server.
if (Test-Path $kh) {
  (Get-Content $kh) | Where-Object { $_ -notmatch '100\.72\.205\.57' -and $_ -notmatch '7a9405e8fb76' } | Set-Content $kh
  "cleaned known_hosts of stale node entries"
}
"=== tailscale ssh fresh ==="
tailscale ssh root@7a9405e8fb76 "hostname; whoami; echo OK" 2>&1
"EXIT=$LASTEXITCODE"
