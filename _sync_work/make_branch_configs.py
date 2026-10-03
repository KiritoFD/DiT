"""生成"共用主干 + 只换注入头"的便宜 A/B 配置 (4ch 线, 从已有 40k ckpt 续)。

为什么便宜: 注入头全是 zero-init(step0 恒等) -> 从同一主干出发, 相对排序可信;
           每臂只跑 3k 步(约 13 分钟) 而非 30k(1.5h)。SPADE 臂后续加一份配置即可。
基底 = v47_purestd_xattn12_top10.json (4ch / 23 槽位 / 我们已训到 40k 的那条线)
"""
import json
import os

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
BASE = "src/train/configs/v47_purestd_xattn12_top10.json"
import glob                                                     # noqa: E402

_cks = sorted(glob.glob("exp-std/runs_purestd/*p1.0/checkpoints/*.pt"))
assert _cks, "找不到主干 ckpt"
CKPT = max(_cks, key=lambda p: int(os.path.basename(p)[:-3]) if os.path.basename(p)[:-3].isdigit() else -1)
print(f"[base] 主干 ckpt = {CKPT}")
BUDGET = 3000
base = json.load(open(BASE, encoding="utf-8"))

COMMON = {
    "results_dir": "exp-std/runs_branch",
    "resume_full": CKPT,
    "max_steps": 40000 + BUDGET,          # 绝对步数续训
    "early_stop": False,
    "ckpt_every": 1000,
    "epoch_steps": 1000,
    "gpu_eval_every": 1000,
    "in_mem_eval_sets": "eval200fix:exp-std/csv/eval200_fixed.csv:187,"
                        "seen:exp-std/csv/seen20.csv:20",
    "eval_csv": "exp-std/csv/eval200_fixed.csv",
    "repa_cache_dir": "data/dino_cache/top10_v1",
    "w_repa": 0.03,
}

ARMS = {
    # ★ 控制臂 = 原样续训(我�are的主线本来就是 xattn×12)。A 与 B 曾写成同内容(重复臂),
    #   这里把 B 改成另一个**现有模式** adaln×4, 保证四臂互不相同。
    "A_xattn_cont": dict(),
    "B_adaln4": dict(model="DiT-2Cond-S/2", glyph_inject_mode="adaln",
                     glyph_inject_layers=4, global_batch_size=256),
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
    # E_spade 由 patch_spade 完成后追加 (需要新代码)
}

for name, delta in ARMS.items():
    c = dict(base)
    c.update(COMMON)
    c.update(delta)
    c["experiment_name"] = f"v49branch-{name}"
    c["_comment"] = (f"v49 分支 A/B 臂 {name}: 共用主干(40k ckpt) + 只换注入头 + {BUDGET} 步; "
                     f"唯一变量=风格/条件注入方式。基线臂 A 是原样续训。")
    p = os.path.join("src/train/configs", f"v49_branch_{name}.json")
    json.dump(c, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[write] {p}  model={c.get('model')} inj={c.get('glyph_inject_mode')}"
          f" layers={c.get('glyph_inject_layers')} bsz={c.get('global_batch_size')}")
print(f"[resume] {CKPT}")
print(f"[budget] 每臂 {BUDGET} 步, max_steps={40000 + BUDGET}")
