# -*- coding: utf-8 -*-
"""
probe_dino_info_content.py — 回答「DINO CLS 到底有多少字形信息可用」。

为什么之前"检索 top-1 4%"不能下结论
----------------------------------
早先结论「DINO CLS 不行」（库外检索 top-1 仅 4%）基于**线性最近邻检索**，
但它测的是原始特征的低层相似性，**不能代表模型经 char_proj 非线性投影后
能提取出多少信息**。文档 12 自己写过：「跨书体检索 top-1 1.9% 但比随机
基线 0.014% 高 130 倍」——有信号，只是线性检索吃不到。

关键事实：DINO 表是「每个 (script, char) 一行」的**平均特征**，
不是样本集。所以无法做需要每类多样本的分类训练（上一个版本因此空转）。
正确的探测方式是**度量特征本身的几何结构**：

指标 1 归一化类内/类间距离比（Dunn-like）:
  同字不同书体（正样本对）的余弦距离 vs 不同字（负样本对）的余弦距离
  比值 < 0.8 -> 特征可分离，CLS 有信息
  比值 > 1.2 -> 特征重叠，CLS 信息不足

指标 2 同字跨书体最近邻准确率:
  对每个 glyph，取同字其他书体作"正类"，看 DINO 空间里最近邻是否落在
  同字上（top-1 命中率）。这是文档 12 跨书体检索 1.9% 的严格版。

指标 3 有效维 / 主成分:
  复核有效秩，看信息集中在多少维。

结论导向
--------
- 指标1 比值 < 0.8 且 指标2 > 30%  -> CLS 有信息，问题在注入/幅度，CLS 可用
- 指标1 比值 0.8-1.2 且 指标2 10-30% -> 部分信息，需 char_proj 放大
- 指标1 比值 > 1.2 且 指标2 < 10%    -> CLS 特征确实缺字形信息，需换表示

用法（远程）
-----------
  python tools/probe_dino_info_content.py
"""
import os, sys, json, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np


def main():
    emb_path = "pretrained_models/dino_embeddings/glyph_dino_embeddings_384.npy"
    idx_path = "pretrained_models/dino_embeddings/glyph_dino_index.json"
    if not (os.path.isfile(emb_path) and os.path.isfile(idx_path)):
        print("# missing dino files")
        return

    emb = np.load(emb_path).astype(np.float64)
    glyphs = json.load(open(idx_path, encoding="utf-8"))
    glyphs = glyphs.get("glyphs", glyphs)
    N = min(len(glyphs), emb.shape[0])
    emb = emb[:N]
    glyphs = glyphs[:N]

    NUM_CH = 7026
    pairs = []
    for g in glyphs:
        sid, cid = (int(g[0]), int(g[1])) if isinstance(g, (list, tuple)) \
            else divmod(int(g), NUM_CH)
        pairs.append((sid, cid))
    sids = np.array([p[0] for p in pairs])
    cids = np.array([p[1] for p in pairs])
    E = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    print(f"# {len(emb)} glyphs, {len(set(cids.tolist()))} distinct chars, "
          f"{len(set(sids.tolist()))} scripts")
    print(f"# script 分布: { {int(s): int((sids==s).sum()) for s in sorted(set(sids.tolist()))} }")

    # ── 指标3：有效秩 / 主成分 ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("指标3: 有效秩 / 主成分能量")
    print("=" * 72)
    Em = E - E.mean(0, keepdims=True)
    # 用 SVD（大数据集 20468x384，直接 np.linalg.svd 可行）
    _, s, _ = np.linalg.svd(Em, full_matrices=False)
    var = s ** 2
    cum = np.cumsum(var) / var.sum()
    print(f"  奇异值总能量: {var.sum():.1f}")
    print(f"  PC1 能量占比: {var[0]/var.sum()*100:.1f}%")
    # 有效秩: 累计能量达 90% 的维度数
    for th in (0.5, 0.9, 0.99):
        k = int(np.searchsorted(cum, th)) + 1
        print(f"  累计能量 {th*100:.0f}% 需要前 {k} 个主成分")

    # ── 指标2: 同字跨书体最近邻命中率 ──────────────────────────────────
    print("\n" + "=" * 72)
    print("指标2: 同字跨书体最近邻命中率")
    print("=" * 72)
    # 同一字出现在多个书体
    char2rows = collections.defaultdict(list)
    for i, (s, c) in enumerate(pairs):
        char2rows[c].append(i)
    multi = {c: rows for c, rows in char2rows.items() if len(rows) >= 2}

    sim = E @ E.T
    top1_hit = 0
    total = 0
    for c, rows in multi.items():
        same = set(rows)
        for i in rows:
            # 排除自己，找最近邻
            order = np.argsort(-sim[i])
            for j in order:
                if j != i:
                    break
            total += 1
            if j in same:
                top1_hit += 1
    if total:
        print(f"  同字跨书体最近邻 top-1 命中率 = {top1_hit/total*100:.1f}% "
              f"({top1_hit}/{total})")

    # ── 指标1: 类内/类间距离比（Dunn-like）─────────────────────────────
    print("\n" + "=" * 72)
    print("指标1: 类内(同字) / 类间(不同字) 距离比")
    print("=" * 72)
    rng = np.random.default_rng(0)

    # 正样本对：同字跨书体
    same_d = []
    for c, rows in multi.items():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                same_d.append(float(1.0 - sim[rows[a], rows[b]]))
    # 负样本对：不同字（同书体 + 跨书体）
    diff_d = []
    for _ in range(40000):
        a, b = rng.integers(0, len(pairs), 2)
        if cids[a] != cids[b]:
            diff_d.append(float(1.0 - sim[a, b]))
    # 对照：同字同书体（如果存在多本）
    # （DINO 表每 glyph 一行，通常没有同字同书体的多行，跳过）

    same_d = np.array(same_d)
    diff_d = np.array(diff_d)
    ratio = same_d.mean() / diff_d.mean()
    print(f"  同字跨书体 平均距离 = {same_d.mean():.4f} (n={len(same_d)})")
    print(f"  不同字    平均距离 = {diff_d.mean():.4f} (n={len(diff_d)})")
    print(f"  距离比 = {ratio:.4f}  {'可分离' if ratio < 0.8 else '中等' if ratio < 1.2 else '重叠'}")

    # ── 汇总判断 ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("结论")
    print("=" * 72)
    nn = top1_hit / total if total else 0
    if ratio < 0.8 and nn > 0.3:
        print("  => CLS 有充分字形信息，问题在注入/幅度，CLS 可用")
        print("     （应修：幅度平衡 + 逐层注入，而不是换特征）")
    elif ratio < 1.2 and nn > 0.1:
        print("  => CLS 有部分信息，需 char_proj 容量放大 + 可学习幅度")
    else:
        print("  => CLS 特征确实缺字形信息，需换表示（patch tokens）")

    os.makedirs("5script", exist_ok=True)
    with open("5script/dino_probe.json", "w", encoding="utf-8") as f:
        json.dump({"ratio": float(ratio), "same_dist": float(same_d.mean()),
                   "diff_dist": float(diff_d.mean()), "top1_nn": nn,
                   "n_multi_char": len(multi)}, f, ensure_ascii=False, indent=2)
    print("\nsaved -> 5script/dino_probe.json")


if __name__ == "__main__":
    main()
