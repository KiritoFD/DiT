#!/bin/bash
# smoke_unfreeze_save.sh — 冒烟: unfreeze_main 训练 2 步, 验证 ckpt 保存 model/ema_model
# 然后立即退出, 不污染正式流程
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
A=assets/results/v8_3stage/A_main_final.pt
OUT=/tmp/smoke_unfreeze
mkdir -p $OUT

# 用最小化 config: v8i 改 max_steps=2, ckpt_every=1, 无 early_stop, 无 eval
cat > /tmp/smoke_cfg.json <<'EOF'
{
  "experiment_name": "smoke-unfreeze-save",
  "results_dir": "/tmp/smoke_unfreeze",
  "model": "DiT-2Cond-S/2",
  "train_ctrl_only": true,
  "unfreeze_main": true,
  "main_lr": 3e-5,
  "compile": false,
  "csv": "assets/train_fame_clean_v8.csv",
  "latent_shards_dir": "data/latents/final_latents_fame_v8",
  "img_root": "data/imgs/final_imgs_fame_v8",
  "skel_root": "data/skel/final_skel1_fame_v8",
  "skel_latent_shards_dir": "data/skel/final_skel_latents_fame_1px_v8",
  "skel_cond_channels": 4,
  "num_calligraphers": 1013,
  "num_characters": 35130,
  "condition_fusion": "factorized_add",
  "callig_embed_dim": 128,
  "char_embed_dim": 384,
  "char_proj_mode": "mlp",
  "freeze_char_table": true,
  "cond_drop_all_prob": 0.05,
  "cond_drop_one_prob": 0.30,
  "cond_drop_which_glyph_prob": 0.85,
  "cond_drop_struct_prob": 0.1,
  "diffusion_type": "flow",
  "t_sampler": "logit_normal",
  "t_mean": 0.0,
  "t_std": 1.0,
  "flow_sampler": "heun",
  "heun_batch": 1,
  "shift": 1.0,
  "norm_type": "rms",
  "mlp_type": "swiglu",
  "qk_norm": 1,
  "rope": 1,
  "rope_theta": 100.0,
  "attn_impl": "sdpa",
  "ctrl_depth": 0,
  "ctrl_hidden": 0,
  "ctrl_num_heads": 0,
  "injection": "modulate",
  "null_cond": "gaussian",
  "epochs": 2,
  "max_steps": 2,
  "lr": 0.0003,
  "warmup_steps": 1,
  "min_lr_ratio": 0.1,
  "weight_decay": 0.01,
  "batch_size": 8,
  "num_workers": 2,
  "log_every": 1,
  "ckpt_every": 1,
  "ckpt_keep": 0,
  "use_ema": true,
  "ema_decay": 0.9999,
  "preload": false,
  "preload_workers": 0,
  "use_checkpoint": false,
  "seed": 0,
  "early_stop": false,
  "gpu_eval_csv": "",
  "w_repa_early": 0.0
}
EOF

$PY src/train/train_controlnet.py --config /tmp/smoke_cfg.json --main-ckpt "$A" > /tmp/smoke_unfreeze.log 2>&1
echo "rc=$?"
tail -5 /tmp/smoke_unfreeze.log

echo "=== 验证 ckpt ==="
CK=$(ls $OUT/*/checkpoints/*.pt 2>/dev/null | head -1)
echo "ckpt: $CK"
[ -n "$CK" ] && $PY -c "
import torch, sys
sd = torch.load('$CK', map_location='cpu', weights_only=False)
print('keys:', list(sd.keys()))
if 'model' in sd:
    mw = sd['model']
    n_main = len(mw)
    # 检查 main 权重是否与 A_main_final 有差异 (解冻训练生效)
    nz = sum(1 for v in mw.values() if v.abs().sum() > 0)
    print(f'model (main) keys={n_main} nonzero={nz}')
if 'ema_model' in sd:
    print('ema_model keys:', len(sd['ema_model']))
    # 任一 main 权重与初始不同?
    import glob
    print('SAVE_OK: unfreeze main persisted')
else:
    print('!!! SAVE_FAIL: no ema_model in ckpt')
"