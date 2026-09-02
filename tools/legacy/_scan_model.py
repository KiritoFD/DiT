# -*- coding: utf-8 -*-
"""
_scan_model.py — 统计 DiT 模型各组件参数量与条件维度，产出 model_components.csv。

在 CPU 上实例化 DiT-2Cond-S/2（约 46M 参数，无需 GPU/权重），逐模块统计：
  - 参数量、占比、形状
  - 冻结状态（s21 默认：char 表冻结）
  - 条件信号的维度对比（这是理解 0.50 vs 0.80 的关键）

产出 model_components.csv + condition_dims.csv
"""
import os, sys, csv, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def human(n):
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def main():
    from src.model import DiT_2Cond_models

    cfg = dict(
        input_size=32, in_channels=4,
        num_calligraphers=1013, num_characters=35130,
        use_checkpoint=False, learn_sigma=False,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=384,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        cond_drop_which_glyph_prob=0.75,
        skel_head_enabled=False, use_glyph_cond=False,
        glyph_scale_init=0.4, char_proj_mode="mlp",
        freeze_char_table=True,
        norm_type="rms", mlp_type="swiglu", qk_norm=True,
        rope=True, rope_theta=100.0, attn_impl="sdpa",
    )
    model = DiT_2Cond_models["DiT-2Cond-S/2"](**cfg)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"# model: DiT-2Cond-S/2  total params = {total:,} ({human(total)})")

    # ── 逐模块统计 ────────────────────────────────────────────────────────
    rows = []
    groups = {}
    for name, p in model.named_parameters():
        # 归组：取前两段名字
        parts = name.split(".")
        if parts[0] == "blocks":
            grp = f"blocks[{parts[1]}].{parts[2]}" if len(parts) > 2 else "blocks"
            top = "blocks (transformer)"
        elif parts[0] == "joint_blocks":
            grp = f"joint_blocks[{parts[1]}].{parts[2]}" if len(parts) > 2 else "joint_blocks"
            top = "joint_blocks"
        else:
            grp = ".".join(parts[:2])
            top = parts[0]
        d = groups.setdefault(top, {"n": 0, "shapes": []})
        d["n"] += p.numel()
        if len(d["shapes"]) < 4:
            d["shapes"].append(f"{name}:{tuple(p.shape)}")

    for top, d in sorted(groups.items(), key=lambda x: -x[1]["n"]):
        rows.append({
            "component": top, "params": d["n"],
            "pct_of_total": round(d["n"] / total * 100, 2),
            "frozen_by_default": "",
            "example_shapes": " | ".join(d["shapes"]),
        })

    # 标注冻结项（s21/fame 默认 freeze_char_table=true）
    frozen_note = {
        "y_char_embedder": "YES (freeze_char_table=true)",
        "y_embedder": "no",
    }
    for r in rows:
        r["frozen_by_default"] = frozen_note.get(r["component"], "no")

    os.makedirs("5script", exist_ok=True)
    with open("5script/model_components.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "params", "pct_of_total",
                                          "frozen_by_default", "example_shapes"])
        w.writeheader()
        w.writerows(rows)
    print(f"# wrote 5script/model_components.csv ({len(rows)} rows)")

    for r in rows[:14]:
        print(f"  {r['component']:<26} {r['params']:>10,}  {r['pct_of_total']:>6}%  "
              f"{r['frozen_by_default']}")

    # ── 条件信号维度对比（核心洞察）────────────────────────────────────────
    conds = [
        {"signal": "书家 ID embedding", "source": "y_embedder (num_calligraphers=1013)",
         "raw_dim": 1013, "embed_dim": 128, "spatial": "1 (全局标量条件)",
         "note": "全局标签，无空间信息"},
        {"signal": "字 ID (DINO glyph)", "source": "y_char_embedder (35130×384, DINO CLS)",
         "raw_dim": 35130, "embed_dim": 384, "spatial": "1 (全局标量条件)",
         "note": "有效秩仅 34.1/384；CLS 为全局摘要，丢失空间结构"},
        {"signal": "标准字形 latent g", "source": "std_glyph_latent_v2 (VAE latent)",
         "raw_dim": 4096, "embed_dim": 4096, "spatial": "4×32×32 (空间图)",
         "note": "w_glyph_cond：编码字形本身，含完整空间结构；fame 覆盖 53.1%"},
        {"signal": "骨架 latent (ControlNet)", "source": "final_skel_latents_fame",
         "raw_dim": 4096, "embed_dim": 4096, "spatial": "4×32×32 (空间图)",
         "note": "GT 骨架→VAE latent；含完整笔画拓扑，是 0.50→0.80 的主因"},
    ]
    with open("5script/condition_dims.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["signal", "source", "raw_dim",
                                          "embed_dim", "spatial", "note"])
        w.writeheader()
        w.writerows(conds)
    print(f"\n# wrote 5script/condition_dims.csv ({len(conds)} rows)")
    print("\n# 条件信号维度对比：")
    print(f"  {'signal':<26}{'embed_dim':>10}  {'spatial':<20}")
    for c in conds:
        print(f"  {c['signal']:<26}{c['embed_dim']:>10}  {c['spatial']:<20}")

    ratio = 4096 / 34.1
    print(f"\n# 空间条件(骨架/字形) vs 字ID有效维度 之比 ≈ {ratio:.0f}x")
    print("  → 这解释了 base(无结构条件) 0.50 与 ControlNet(有结构条件) 0.80 的差距")


if __name__ == "__main__":
    main()
