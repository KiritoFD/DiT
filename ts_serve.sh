#!/bin/sh
S=/var/run/tailscale/tailscaled.sock

echo "=== 1. Ensure tailscale up (userspace) + ssh cap ==="
tailscale --socket=$S up --tun=userspace-networking --ssh 2>&1 | head
sleep 2

echo "=== 2. Expose tailnet TCP :22 -> container sshd :36430 ==="
tailscale --socket=$S serve --tcp 22 tcp://127.0.0.1:36430 2>&1 | head
sleep 2

echo "=== 3. serve status ==="
tailscale --socket=$S serve status 2>&1 | head -n10

echo "=== 4. sshd listen check ==="
ss -tlnp 2>/dev/null | grep -E 'sshd|:36430' || netstat -tlnp 2>/dev/null | grep -E 'sshd|:36430'

echo "=== 5. authorized_keys ==="
mkdir -p /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
echo "existing keys:"; wc -l /root/.ssh/authorized_keys

echo "=== 6. root sshd config root login ==="
grep -iE 'PermitRootLogin|AuthorizedKeysFile|PasswordAuthentication' /etc/ssh/sshd_config /etc/ssh/sshd_config.d/* 2>/dev/null | grep -v '^\s*#'

echo "=== DONE ==="
