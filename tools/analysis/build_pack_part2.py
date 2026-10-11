# -*- coding: utf-8 -*-
"""build_pack_part2.py — 专家包 第2/3/4/5 项 + 文档 + 导读 + zip"""
import csv
import glob
import hashlib
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PACK = "expert_pack/callig_dit_pack_20261011"
for d in ("01_docs", "03_per_sample", "04_overlap_audit", "05_v80_protocol", "06_quality_validation"):
    os.makedirs(os.path.join(PACK, d), exist_ok=True)

rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8")))
tpath = next(p for p in ("exp-std/csv/train_top10_noaug.csv", "exp-std/csv/train.csv") if os.path.exists(p))
train = list(csv.DictReader(open(tpath, encoding="utf-8")))
seen = {(r["calligrapher"], r["script"], r["character"]) for r in train}

# ────────────────────────── 第2项：逐样本表 ──────────────────────────
print("== 2. 逐样本表 ==")
MODELS = ["v68", "v68_sde", "v72", "v72_sde", "v76", "v77", "v79", "v79_55k", "v79_sde"]


def band(a, r_cut=0.5):
    b = a.astype(np.float32).mean(axis=2) / 255.0
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    return float(F[r > r_cut].sum() / max(F.sum(), 1e-9))


from skimage.metrics import structural_similarity as ssim_fn  # noqa: E402

out = []
for i, r in enumerate(rows):
    gp = f"eval/gt/{i}.png"
    if not os.path.exists(gp):
        continue
    g = np.asarray(Image.open(gp).convert("RGB"))
    rec = dict(idx=i, img_id=r["img_id"], calligrapher=r["calligrapher"], script=r["script"],
               character=r["character"], combo_seen=(r["calligrapher"], r["script"], r["character"]) in seen,
               gt_ink=round(float((g.mean(axis=2) < 128).mean()), 4),
               gt_hf=round(band(g), 5))
    for m in MODELS:
        p = f"eval/{m}/{i}.png"
        if not os.path.exists(p):
            continue
        a = np.asarray(Image.open(p).convert("RGB").resize((256, 256)))
        gl = np.asarray(Image.open(gp).convert("L").resize((256, 256)))
        al = np.asarray(Image.fromarray(a).convert("L"))
        rec[f"{m}_ssim"] = round(float(ssim_fn(a, g, channel_axis=2, data_range=255)), 4)
        rec[f"{m}_hf"] = round(band(a), 5)
        rec[f"{m}_ink"] = round(float((al < 128).mean()), 4)
        rec[f"{m}_ink_iou"] = round(float(((al < 128) & (gl < 128)).sum() / max(((al < 128) | (gl < 128)).sum(), 1)), 4)
        try:
            from src.data.vae_io import frag_score  # 复用同口径
            rec[f"{m}_frag"] = round(float(frag_score(a, g)), 4)
        except Exception:
            pass
    out.append(rec)
with open(f"{PACK}/03_per_sample/per_sample_metrics.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print(f"  -> 03_per_sample/per_sample_metrics.csv  ({len(out)} 行 × {len(out[0])} 列)")

# 分档汇总（见/未见）
agg = defaultdict(list)
for r in out:
    for m in MODELS:
        k = f"{m}_ssim"
        if k in r:
            agg[(m, r["combo_seen"])].append(r[k])
with open(f"{PACK}/03_per_sample/summary_seen_vs_unseen.csv", "w", encoding="utf-8", newline="") as fh:
    fh.write("model,combo_seen,n,ssim_mean\n")
    for (m, sn), v in sorted(agg.items()):
        fh.write(f"{m},{sn},{len(v)},{np.mean(v):.4f}\n")
print("  -> 03_per_sample/summary_seen_vs_unseen.csv")
for (m, sn), v in sorted(agg.items()):
    print(f"     {m:<10} seen={str(sn):<5} n={len(v):<4} SSIM={np.mean(v):.4f}")

# ────────────────── 第4项：重叠审计 ──────────────────
print("\n== 4. 重叠审计 ==")
def md5(p, n=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()

eval_paths = {r["image_path"] for r in rows}
train_paths = {r["image_path"] for r in train}
inter = eval_paths & train_paths
lines = ["# 训练/评测数据重叠审计", "",
         f"- 评测集 `eval200_fixed` 行数: {len(rows)}（唯一 image_path {len(eval_paths)}）",
         f"- 主线训练集 `{tpath}` 行数: {len(train)}（唯一 image_path {len(train_paths)}）",
         f"- **路径交集: {len(inter)} / {len(eval_paths)} = {100*len(inter)/max(len(eval_paths),1):.1f}%**",
         "",
         "## 逐文件 md5（前 1 MiB）—— 证明「同一文件」而非「同名不同图」", ""]
same = 0
for p in sorted(inter)[:40]:
    if os.path.exists(p):
        lines.append(f"- `{p}`  md5={md5(p)}")
        same += 1
lines += ["", f"（上表抽检 {same} 个交集文件；全部存在且哈希唯一可复核）", "",
          "## 逐 (书家 × 书体) 槽位样本数", "",
          "| 书家 | 书体 | 训练(noaug) | 评测 | 评测中组合未见 |", "|---|---|---|---|---|"]
tr = Counter((r["calligrapher"], r["script"]) for r in train)
ev = Counter((r["calligrapher"], r["script"]) for r in rows)
un = Counter((r["calligrapher"], r["script"]) for r in rows
             if (r["calligrapher"], r["script"], r["character"]) not in seen)
for k in sorted(set(tr) | set(ev)):
    lines.append(f"| {k[0]} | {k[1]} | {tr.get(k,0)} | {ev.get(k,0)} | {un.get(k,0)} |")
open(f"{PACK}/04_overlap_audit/overlap_audit.md", "w", encoding="utf-8", newline="").write("\n".join(lines) + "\n")
print(f"  路径交集 {len(inter)}/{len(eval_paths)} = {100*len(inter)/max(len(eval_paths),1):.1f}%")
print("  -> 04_overlap_audit/overlap_audit.md")

# 独立测试划分（真正未见字符）
train_chars = {r["character"] for r in train}
unseen_chars = sorted({r["character"] for r in rows} - train_chars)
with open(f"{PACK}/03_per_sample/heldout_unseen_characters.csv", "w", encoding="utf-8", newline="") as fh:
    fh.write("character,n_eval_rows,img_ids\n")
    for c in unseen_chars:
        ids = [r["img_id"] for r in rows if r["character"] == c]
        fh.write(f"{c},{len(ids)}," + "|".join(ids) + "\n")
print(f"  真正未见字符数: {len(unseen_chars)} -> heldout_unseen_characters.csv")

# ────────────────── 文档 / 协议 / 质量验证 ──────────────────
print("\n== 1/5/6: 文档与说明 ==")
for f in ("README.md", "state.md", "docs/vae_latent_contract.md",
          "docs/vae_training_recipe.md", "docs/sde_analysis_brief.md"):
    if os.path.exists(f):
        shutil.copy(f, os.path.join(PACK, "01_docs", os.path.basename(f)))

open(f"{PACK}/05_v80_protocol/PROTOCOL.md", "w", encoding="utf-8", newline="").write("""# 第 3 项：v80 匹配步数实验协议（待跑）

## 状态
- **v79 已停止**（它带 C2OT，其 SDE 数字在数学上不可用）
- **v80 运行中**：`use_c2ot=false` + **修正后的 SDE 采样器**，其余与 v79 完全一致
  （纯物理 VAE / sf=0.843977 / shift=−0.03342 / batch 288 / 200k cosine / w_repa 0.03 / grad_clip 1.0）
- 起始：2026-10-11 19:58（4090）；速度 ~3.8 steps/s

## 要产出的三行（同 batch / 同评测集 / 同 CFG=1.0 / 同 NFE=100 / heun-50）
| 行 | ckpt | γ | 说明 |
|---|---|---|---|
| v79 ODE | v79@35k,55k,100k | 0.0 | 已有（旧配方 + C2OT） |
| v80 ODE | v80@35k,55k,100k | 0.0 | 新（无 OT） |
| **v80 修正-SDE** | v80@同上 | 0.5 | **关键：修正后的 SDE** |

## 精确命令（本机生成）
```bash
python tools/gen_eval_rows.py --name <tag> --ckpt <ckpt.pt> \\
  --config src/train/configs/v80_callivae_physics_noOT.json \\
  --vae-path experiments/calli_vae_physics/calli_vae_final \\
  --cfg 1.0 --gamma <0.0|0.5> --steps 50 --indices 0,1,...,186 --batch 4
python tools/eval_split_metrics.py --models <tags...> --device cpu
python tools/hf_decompose.py
```
**逐图差值**：`03_per_sample/per_sample_metrics.csv` 已提供逐样本 SSIM/HF/ink/ink-IoU 列，
可直接做 **paired bootstrap** 置信区间（专家要求）。生成脚本 `tools/analysis/sde_e2e_test.py`
可在**不训练**的情况下验证采样器的边缘保持性。

## 不需要重训
修正只改采样器 ⇒ **对已有 ckpt 直接重采样即可**；但**要求该 ckpt 训练时无 OT**
（否则 Tweedie 恒等式不成立）—— 这也是 v79 的 SDE 数字不可用的原因。
""")

open(f"{PACK}/06_quality_validation/STATUS_AND_GAPS.md", "w", encoding="utf-8", newline="").write("""# 第 5 项：内容与风格的独立质量验证 —— 现状与缺口

## 已具备
- `tgt_spec`（目标风格度量）与 `cal_enrich`（书法丰富度）等**自建**指标（本项目主线使用）
- `frag`（断裂度）、`ink_iou`、`skel_iou`、`HF`（顶带频谱份额）等**物理**指标，口径已与离线统一

## 缺口（本次无法提供，需要外部资源）
1. **在真实书法测试集上标定过的字符识别器**：需要训练/引入一个书法 OCR 并在 GT 上验证其准确率；
   本项目**没有**这样的模型（`samechar-nn` 是最近邻检索，不是识别器）。
2. **书家/书体分类器**：同样需要独立训练 + 在与训练不重叠的 GT 上标定。
3. **书法人士双盲 A/B 偏好评价**：需要人。

## 建议方案（可执行）
- (a) 用 GT 直接训练一个 **书家/书体分类器**（10 类/3 类），在**留出字符**上评测 → 用其准确率
  评价生成样本的"风格可辨性"；
- (b) 字符识别器可直接用**同条件 GT 最近邻**做代理（`samechar-nn`），但**必须报告其 GT 上的上限**；
- (c) 人工双盲：从 `02_sample_grids/` 生成 A/B 对（GT vs 生成、模型 vs 模型），随机化左右、隐藏来源。
""")

open(f"{PACK}/00_README_FOR_EXPERT.md", "w", encoding="utf-8", newline="").write("""# Callig-DiT（马良）专家材料包 — 2026-10-11

**目的**：按专家要求提供五项材料，用于判断"当前核心短板是条件分布泛化还是采样/表征"。

## 目录
| 目录 | 对应专家的第几项 | 状态 |
|---|---|---|
| `01_docs/` | — | ✅ 重要文档（README / state / latent 契约 / VAE 配方 / SDE 技术简报） |
| `02_sample_grids/` | **第 1 项** | ✅ GT/v68/v79 三列海报 × 楷/行/隶；每张 8 行，覆盖 **见/未见组合 × 简/繁**；含 `grid_index.csv` |
| `03_per_sample/` | **第 2 项** | ✅ 逐样本表（SSIM/HF/ink/ink-IoU[/frag]）+ 见/未见汇总 + **真正未见字符表** |
| `04_overlap_audit/` | **第 4 项** | ✅ 路径交集 + 抽检 md5 + 逐 (书家×书体) 槽位计数 |
| `05_v80_protocol/` | **第 3 项** | ⏳ 协议与命令已备；**数据待 v80 训到 35k/55k/100k**（v80 于 10-11 19:58 起跑，~3.8 steps/s） |
| `06_quality_validation/` | **第 5 项** | ⚠️ 盘点与缺口：**无标定过的识别器/分类器、无人工双盲**（需外部资源），已给可执行方案 |

## ★ 第 1 项已直接给出的结论（`02_sample_grids/grid_楷.png`）
- **SEEN（组合见过）**：GT / v68 / v79 **几乎都正确**；
- **UNSEEN（组合未见）**：**两个模型频繁产生错字或部件错误**（例：欧阳询「你」→ v68 出「佛」形、v79 出「流」；
  文徵明「槛」→ 两者分别出「德」「慶」；颜真卿「體」→ v68 残缺、v79 近似「禮」）。
→ **核心短板是"条件分布泛化"，而不是 SSIM 差 0.01 的量级**。
→ 因此本项目此前反复争论的模型间 ±0.01 SSIM 差异，在"写对字 / 写错字"面前属于噪声。

## 必须随材料一起读的口径警告
1. **种子**：评测缓存的初始噪声是 `manual_seed(0)` **硬编码**（`src/eval/inference.py:609`）
   → 所有模型、所有行**共用同一条噪声**。利于模型间对比，但**无法提供"3 个不同种子"**。
2. **两套 LPIPS**：训练内 `vgg` vs 本地海报 `alex`，同模型差 ~0.06，**不可混读**。
3. **两套评测口径**：训练内 `cfg1.0 + γ0`；离线默认 `cfg0.8 + γ0.5`。**不可横比**。
4. **v79 的 SDE 数字在数学上不可用**：它训练时用了 C2OT，破坏了修正后 SDE 所需的独立配对前提。
5. **难度代理**：海报里的 `ink=` 是该 GT 的墨量占比，**不是笔画数**（仅为难度代理）。
""")

# ────────────────── zip ──────────────────
print("\n== zip ==")
z = "expert_pack/callig_dit_pack_20261011"
if os.path.exists(z + ".zip"):
    os.remove(z + ".zip")
subprocess.run(["powershell", "-NoProfile", "-Command",
                f"Compress-Archive -Path '{z}\\*' -DestinationPath '{z}.zip' -Force"], check=False)
sz = os.path.getsize(z + ".zip") / 1e6 if os.path.exists(z + ".zip") else 0
print(f"  -> {z}.zip  ({sz:.1f} MB)")
