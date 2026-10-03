"""生成 4 个注入方式 A/B 配置 (16ch Flux VAE / 从零 / 同预算 / 23 槽位)。

基底 = v47_purestd_xattn12_top10.json (我们刚跑的那份), 只改必要字段:
  16ch: latent_channels=16 / 三个 shard 目录换成 flux16 / vae_scaling_factor=0.3611 /
        eval_vae_path=data/pretrained/flux_vae_eval   (shift 已折进 shard, 解码零改动)
  从零: 不带 pretrained/resume; max_steps=30000; 每 2500 步 eval200fix(187)+seen(20)
四臂:
  A_adaln       DiT-2Cond-S/2  + adaln×4                     (基线: 全局位置调制)
  B_xattn       DiT-2Cond-Sp/2 + xattn×12                    (把 std 骨架当 token 上下文)
  C_every_layer B + style_ctx_every_layer                    (实验2: 每层注入 / IP-Adapter 式)
  D_k4_styleca  S/2 + K=4 风格字典 + CalligStyleCrossAttn     (实验1: 骨架空间寻址, v15b 形态)
"""
import json
import os

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
BASE = "src/train/configs/v47_purestd_xattn12_top10.json"
OUTD = "src/train/configs"

base = json.load(open(BASE, encoding="utf-8"))
COMMON = {
    "latent_channels": 16,
    # ★ 条件 g 骨架 latent 的通道数也是 16 (glyph_embedder 的入通道;
    #   不设会保持 4 -> conv 尺寸失配, 冒烟已抓到)
    "glyph_latent_channels": 16,
    "latent_shards_dir": "exp-std/data/shards_img_flux16",
    "skel_latent_shards_dir": "exp-std/data/shards_std_flux16",
    "eval_skel_latent_shards_dir": "exp-std/data/shards_std_flux16_fixed_eval200",
    "vae_scaling_factor": 0.3611,
    "eval_vae_path": "data/pretrained/flux_vae_eval",
    "results_dir": "exp-std/runs_inject",
    "max_steps": 30000,
    "ckpt_every": 2500,
    "epoch_steps": 2500,
    "gpu_eval_every": 2500,
    "in_mem_eval_sets": "eval200fix:exp-std/csv/eval200_fixed.csv:187,"
                        "seen:exp-std/csv/seen20.csv:20",
    "eval_csv": "exp-std/csv/eval200_fixed.csv",
    "data_csv": "exp-std/csv/train.csv",
    "w_repa": 0.03,
    "repa_cache_dir": "data/dino_cache/top10_v1",
}

ARMS = {
    "A_adaln": dict(model="DiT-2Cond-S/2", glyph_inject_mode="adaln",
                    glyph_inject_layers=4, global_batch_size=256),
    "B_xattn": dict(model="DiT-2Cond-Sp/2", glyph_inject_mode="xattn",
                    glyph_inject_layers=12, global_batch_size=128),
    "C_every_layer": dict(model="DiT-2Cond-Sp/2", glyph_inject_mode="xattn",
                          glyph_inject_layers=12, global_batch_size=128,
                          style_ctx_every_layer=True),
    "D_k4_styleca": dict(model="DiT-2Cond-S/2", glyph_inject_mode="adaln",
                         glyph_inject_layers=4, global_batch_size=256,
                         callig_spatial=False, callig_multi_style_k=4,
                         callig_style_ca=True, callig_embed_dim=384,
                         callig_emb_pretrained="assets/multistyle_k4_top10.pt",
                         freeze_callig_table=False,
                         style_anchor_weight=0.01, style_anchor_mode="mean",
                         glyph_drop_prob=0.1),
}

for name, delta in ARMS.items():
    c = dict(base)
    c.update(COMMON)
    c.update(delta)
    c["experiment_name"] = f"v48-inject-{name}"
    c["_comment"] = (f"v48 注入 A/B 臂 {name}: 16ch Flux VAE + 从零 + 30k 步; "
                     f"条件=std 骨架(w7) / 目标=真迹图; 判分 eval200fix(187)+seen(20)。"
                     f"唯一变量=风格/条件注入方式。")
    p = os.path.join(OUTD, f"v48_inject_{name}.json")
    json.dump(c, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    keys = {k: c.get(k) for k in ("model", "glyph_inject_mode", "glyph_inject_layers",
                                  "style_ctx_every_layer", "callig_multi_style_k",
                                  "callig_style_ca", "global_batch_size", "max_steps")}
    print(f"[write] {p}")
    print(f"        {keys}")
print("\n[note] D 臂需要 callig_spatial=False (模型里与 callig_multi_style_k 互斥, 已设)")
print("[note] eval_vae_path 需要 in_mem_eval 的那 1 行补丁支持 (见 patch_evalevaepath)")
