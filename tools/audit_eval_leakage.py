# -*- coding: utf-8 -*-
"""audit_eval_leakage.py — 审计 eval200_fixed(187) 是否真的**未参与训练**。

为什么必须查: 三张海报的档位是"按 v68 每样本 SSIM 降序"切的。若 eval 与训练集有重叠,
v68 在 Best 档的高分里有相当部分是**记住**的样本 -> 海报会把记忆伪装成泛化。

查 5 个层级 (从强到弱), 与**实际训练用的两份清单**都比:
  train.csv                (26,002 原始行, v66/v67/v68 的底料)
  train_top10_aug_sym.csv  (77,823 = 原始 + tp/tn, v68 训练用的**就是这份**)

  ① img_id          完全相同的主键
  ② image_path      同一个物理文件
  ③ src_image_path  同一个**原始作品**(增强行继承源行的该列) -> 最能抓 v4 增强的泄漏
  ④ (书家, 书体, 字) 组合见过
  ⑤ (书家, 字)      同书家同字(不同书体也算见过)
"""
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL = "exp-std/csv/eval200_fixed.csv"
TRAINS = [("train.csv (26,002 原始)", "exp-std/csv/train.csv"),
          ("train_top10_aug_sym.csv (77,823 含增强)", "exp-std/csv/train_top10_aug_sym.csv")]


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def keys(rs):
    return {
        "img_id": {str(r.get("img_id", "")).strip() for r in rs},
        "image_path": {str(r.get("image_path", "")).strip() for r in rs},
        "src_image_path": {str(r.get("src_image_path", "")).strip() for r in rs},
        "slot_char": {(str(r.get("calligrapher", "")).strip(), str(r.get("script", "")).strip(),
                       str(r.get("character", "")).strip()) for r in rs},
        "callig_char": {(str(r.get("calligrapher", "")).strip(),
                         str(r.get("character", "")).strip()) for r in rs},
        "char": {str(r.get("character", "")).strip() for r in rs},
    }


def main():
    ev = rows(EVAL)
    print(f"[eval] {EVAL}: {len(ev)} 行")
    print(f"       样例 image_path = {ev[0]['image_path']}")
    print(f"       样例 img_id     = {ev[0].get('img_id')}   src_image_path = {ev[0].get('src_image_path')}")
    ek = keys(ev)

    for label, tp in TRAINS:
        tr = rows(tp)
        tk = keys(tr)
        print()
        print("=" * 88)
        print(f"### vs {label}   ({len(tr)} 行)")
        for k, cn in (("img_id", "① img_id 主键"), ("image_path", "② image_path 同一文件"),
                      ("src_image_path", "③ src_image_path 同一原始作品"),
                      ("slot_char", "④ (书家,书体,字) 组合"), ("callig_char", "⑤ (书家,字)"),
                      ("char", "   (仅字)")):
            hit = ek[k] & tk[k]
            hit.discard("")
            n = len(hit)
            flag = "✓ 无" if n == 0 else f"⚠ **{n} 个重叠**"
            print(f"  {cn:34s} {flag}")
            if n:
                print(f"      样例: {list(hit)[:6]}")

    # 逐样本列出 ③ 的重叠证据 (最要紧的一项)
    print()
    print("=" * 88)
    print("### 逐样本: eval 的 src_image_path 是否出现在 v68 训练清单里")
    tr = rows(TRAINS[1][1])
    tk = keys(tr)
    bad = []
    for i, r in enumerate(ev):
        sp = str(r.get("src_image_path", "")).strip()
        ip = str(r.get("image_path", "")).strip()
        why = []
        if sp and sp in tk["src_image_path"]:
            why.append("src_image_path 命中")
        if ip in tk["image_path"]:
            why.append("image_path 命中")
        if str(r.get("img_id", "")).strip() in tk["img_id"]:
            why.append("img_id 命中")
        if why:
            bad.append((i, r.get("character"), r.get("calligrapher"), r.get("script"), why, sp))
    if not bad:
        print("  ✓ 187 行**全部**三项都不命中 —— eval200_fixed 与 v68 训练数据无重叠")
    else:
        print(f"  ⚠ {len(bad)}/{len(ev)} 行命中:")
        for i, ch, cal, sc, why, sp in bad[:20]:
            print(f"    idx={i:3d} {cal}·{sc}·{ch}  {why}  src={sp}")

    # 附带: eval 的字有多少在训练字表里 (字级覆盖, 只能说明"字见过", 不等于样本重叠)
    print()
    ch_hit = ek["char"] & tk["char"]
    print(f"[附] eval 的 {len(ek['char'])} 个不同字中, 有 {len(ch_hit)} 个在训练字表里 "
          f"(字级覆盖 — 属正常, 不等于样本泄漏)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
