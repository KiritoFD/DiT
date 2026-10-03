#!/bin/bash
cd /root/Workspace/xy/DiT
LOG=assets/results/s28_std_dino_pretrain/20260831-192249-s28-std-dino-pretrain/log.txt
echo "=== s28 startup (data preload / dataset build) ==="
grep -a -i -E 'preload|dataset|shard|worker|loading.*data|build' $LOG | head -15
echo ""
echo "=== Steps/Sec distribution (last runs) ==="
grep -a -o 'Steps/Sec: [0-9.]*' $LOG | tail -20 | awk '{print $2}' | sort -n | tail -5
echo "(min, recent) latest steps/sec:"
grep -a -o 'Steps/Sec: [0-9.]*' $LOG | tail -3
echo ""
echo "=== Memory === "
grep -a -o 'Mem: [0-9.]*G/[0-9.]*G' $LOG | tail -3
echo ""
echo "=== resolved config infra fields ==="
grep -a -E 'batch_size|num_workers|preload|latent_shards|grad_acc' src/train/configs/s28_std_dino_pretrain.json
echo ""
echo "=== dataset shards (IO source) ==="
ls /root/Workspace/xy/DiT/data/latents/final_latents_fame/ 2>/dev/null | head -5
ls /root/Workspace/xy/DiT/data/latents/final_latents_fame/ 2>/dev/null | wc -l
echo ""
echo "=== nvidia driver / torch ==="
/opt/conda/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('cudnn', torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'na')"
