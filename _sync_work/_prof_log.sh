#!/bin/bash
cd /root/Workspace/xy/DiT
LOG=assets/results/s28_std_dino_pretrain/20260831-192249-s28-std-dino-pretrain/log.txt
echo "=== Steps/Sec series (every logged step) ==="
grep -a -o 'step=[0-9]*.*Steps/Sec: [0-9.]*' $LOG | grep -a -o 'step=[0-9]* .*Steps/Sec: [0-9.]*' | head -40
echo ""
echo "=== eval / checkpoint log lines ==="
grep -a -E 'Saved checkpoint|eval|Eval|Sampling|Validation' $LOG | head -20
echo ""
echo "=== unique logged steps and their Steps/Sec ==="
grep -a -o 'Steps/Sec: [0-9.]*' $LOG | awk '{a+=$2; n++} END{print "avg", a/n, "over", n, "logged"}'
echo ""
echo "=== count eval_pending / eval dirs (eval cost) ==="
ls assets/results/s28_std_dino_pretrain/20260831-192249-s28-std-dino-pretrain/checkpoints/ | grep -c eval
echo ""
echo "=== dataset: num shards, samples per shard ==="
ls /root/Workspace/xy/DiT/data/latents/final_latents_fame/*.npy 2>/dev/null | wc -l
