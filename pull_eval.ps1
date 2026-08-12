# Pull latest 3Cond full-training ckpt, run remote eval, pull comparison png back.
# Usage:  powershell -ExecutionPolicy Bypass -File pull_eval.ps1  [n] [t]
param(
    [int]$n = 8,
    [int]$t = 150
)
$ErrorActionPreference = "Continue"
$REMOTE_HOST = "root@10.176.54.17"
$PORT = "36430"
$REMOTE = "/root/Workspace/xy/DiT"
$LOCAL = "g:/GitHub/DiT"

# 1) Find latest ckpt on remote
$latest = ssh -o ConnectTimeout=8 $REMOTE_HOST -p $PORT "ls -t $REMOTE/results/full_3cond/*/checkpoints/*.pt 2>/dev/null | head -1"
if (-not $latest -or -not (Test-Path $latest)) {
    Write-Host "[pull_eval] No ckpt found yet. Retry later."
    exit 0
}
$ckptName = Split-Path $latest -Leaf   # e.g. 0002500.pt
$stepNum = ($ckptName -replace '\.pt$','')
Write-Host "[pull_eval] Latest ckpt: $latest (step $stepNum)"

# 2) Track already-evaluated steps to avoid re-running
$marker = "$LOCAL/pull_eval_done.txt"
$done = @()
if (Test-Path $marker) { $done = @(Get-Content $marker) }
if ($done -contains $stepNum) {
    Write-Host "[pull_eval] step $stepNum already evaluated. Done."
    exit 0
}

# 3) Run remote eval (uses remote python + dataset). Skip if already evaluated.
$outName = "eval_step_$stepNum.png"
Write-Host "[pull_eval] Running remote eval for step $stepNum ..."
ssh -o ConnectTimeout=10 $REMOTE_HOST -p $PORT "cd $REMOTE && /opt/conda/bin/python eval_full_3cond.py '$latest' '$outName' $n $t 2>&1 | tail -5"

# 4) Pull png back
scp -o ConnectTimeout=10 -P $PORT "${REMOTE_HOST}:${REMOTE}/${outName}" "$LOCAL/$outName" 2>&1 | Out-Null
if (Test-Path "$LOCAL/$outName") {
    Write-Host "[pull_eval] Pulled $outName ($((Get-Item "$LOCAL/$outName").Length) bytes)"
    $stepNum | Add-Content $marker
} else {
    Write-Host "[pull_eval] WARNING: failed to pull $outName"
}
