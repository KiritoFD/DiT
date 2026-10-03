#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== ctrl_fame_1pix_v1 resolved config ==="
cat assets/results/ctrl_fame_1pix_v1/*/resolved_config.json 2>/dev/null | /opt/conda/bin/python -c "
import json,sys
d=json.load(sys.stdin)
for k in ['main_ckpt','csv','skel_root','skel_latent_shards_dir','use_ids_char_embedder','char_embed_dim','batch_size','max_steps','lr','gpu_eval_skel_root','gpu_eval_skel_latent_shards_dir','train_ctrl_only']:
    print(f'  {k}: {d.get(k)}')
" 2>/dev/null || echo "(no resolved config)"
echo ""
echo "=== s26 train.log: skel 条件是否生效 ==="
grep -a -E 'skel|struct|cond' assets/results/s26_ctrl_gt_skel/20260831-122923-s26-ctrl-gt-skel-1px/log.txt 2>/dev/null | head -10
echo ""
echo "=== s26 训练 loss 趋势 (ctrl 是否学到) ==="
grep -a -o 'Total: [0-9.]*' assets/results/s26_ctrl_gt_skel/20260831-122923-s26-ctrl-gt-skel-1px/log.txt 2>/dev/null | head -3
grep -a -o 'Total: [0-9.]*' assets/results/s26_ctrl_gt_skel/20260831-122923-s26-ctrl-gt-skel-1px/log.txt 2>/dev/null | tail -3
echo ""
echo "=== ctrl_fame_1pix_v1 训练 loss 趋势 ==="
grep -a -o 'Total: [0-9.]*' assets/results/ctrl_fame_1pix_v1/*/log.txt 2>/dev/null | head -3
grep -a -o 'Total: [0-9.]*' assets/results/ctrl_fame_1pix_v1/*/log.txt 2>/dev/null | tail -3
echo ""
echo "=== 1px GT skel latent 数据 ==="
ls data/skel/final_skel_latents_fame_1px/ 2>/dev/null | head -3
ls data/skel/final_skel_latents_fame_1px/ 2>/dev/null | wc -l
ls data/skel/final_skel_latents_eval_1px/ 2>/dev/null | head -3
ls data/skel/final_skel_latents_eval_1px/ 2>/dev/null | wc -l
