# -*- coding: utf-8 -*-
"""DINO cls/patch 直接作为 char/glyph embedding —— 零可训练参数版本。

用户核心观点
------------
如果 DINO 初始信号足够好，就不应该用可训练投影网络去扭曲它，应直接把原始
特征作为条件注入模型（冻结查表，可训练参数量 = 0，自然满足 <=2M）。

评测方法（避免像素级骨架度量对非刚性字形结构的误判）
--------------------------------------------------
不用"像素 IoU"当外形真值（它对笔画位置敏感，土/士 结构相似却 IoU 低）。
改用**判别性评测**：
  - 正例：人工标注的形近字对（土/士、大/太、王/玉…）——这些字结构相似。
  - 负例：随机不同字对——结构通常不同。
  - 指标：CLS 余弦能否把正例排在负例前面。
      * 形近对余弦均值 - 随机对余弦均值（分离度）
      * AUC（用余弦做形近/随机二分类）
若 CLS 给形近对显著更高余弦 -> CLS 正确捕捉"外形像的字接近" -> 可零参数直接注入。
"""
import os, sys, json, io, time, glob, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
from collections import defaultdict

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
DEV = "cuda"
_t0 = time.time()
def log(*a): print(" ".join(str(x) for x in a), flush=True)


# 人工标注的形近字对（结构相似，人类公认）
SIM_PAIRS = [
    ("土","士"),("大","太"),("人","入"),("日","曰"),("天","夫"),("刀","力"),
    ("申","由"),("王","玉"),("牛","午"),("己","已"),("未","末"),("木","林"),
    ("休","体"),("侯","候"),("风","凤"),("几","凡"),("戊","戍"),("己","巳"),
    ("亳","毫"),("宋","宗"),("东","车"),("冈","同"),("三","王"),("十","干"),
    ("大","天"),("夫","天"),("犬","太"),("人","个"),("手","毛"),("毛","手"),
    ("子","孑"),("戊","戌"),("戍","戌"),("刀","刁"),("万","方"),("鸟","乌"),
    ("贝","见"),("龙","尤"),("失","矢"),("句","向"),("因","困"),("同","回"),
    ("问","间"),("门","们"),("口","曰"),("田","由"),("甲","申"),("白","百"),
    ("吉","古"),("夫","夭"),("天","夭"),("王","主"),("玉","主"),("人","入"),
]

charid2char = {}
for r in csv.DictReader(open("assets/train_fame.csv", encoding="utf-8")):
    charid2char[int(r["character_id"])] = r["character"]

def glyph_to_char(emb, glyphs):
    acc = defaultdict(list)
    for (sid, cid), e in zip(glyphs, emb):
        acc[cid].append(e)
    out = {}
    for cid, es in acc.items():
        ch = charid2char.get(cid)
        if not ch or len(ch) != 1:
            continue
        v = np.mean(es, axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        out[ch] = v
    return out

def auc(scores_pos, scores_neg):
    """二分类 AUC: 正例(形近)分数应高于负例(随机)。越接近 1 越好。"""
    s = np.concatenate([scores_pos, scores_neg])
    lab = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(s)                 # 升序，正例分数越高排越后
    lab_sorted = lab[order]
    # 正例秩和（1-based rank）
    ranks = np.where(lab_sorted == 1)[0] + 1
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Mann-Whitney U -> AUC
    r_sum = ranks.sum()
    auc = (r_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

def eval_discriminative(emb_dict, label):
    """判别性评测：形近对余弦 vs 随机对余弦。"""
    # 可用形近对
    pos_pairs = [(a, b) for a, b in SIM_PAIRS if a in emb_dict and b in emb_dict]
    if not pos_pairs:
        log(f"  [{label}] 无形近对可评"); return
    pos_sim = np.array([emb_dict[a] @ emb_dict[b] for a, b in pos_pairs])
    # 随机负例对（不同字，避免形近）
    chars = list(emb_dict.keys())
    rng = np.random.default_rng(0)
    neg_sim = []
    while len(neg_sim) < len(pos_pairs) * 10:
        a, b = rng.choice(chars, 2, replace=False)
        if (a, b) in SIM_PAIRS or (b, a) in SIM_PAIRS:
            continue
        neg_sim.append(emb_dict[a] @ emb_dict[b])
    neg_sim = np.array(neg_sim)
    a = auc(pos_sim, neg_sim)
    log(f"  [{label:<12}] 形近对={len(pos_pairs):>3} 随机对={len(neg_sim):>4}")
    log(f"        形近余弦 mean={pos_sim.mean():.4f}±{pos_sim.std():.4f}")
    log(f"        随机余弦 mean={neg_sim.mean():.4f}±{neg_sim.std():.4f}")
    log(f"        分离度(形近-随机) = {pos_sim.mean()-neg_sim.mean():+.4f}  AUC = {a:.4f}")
    # 打印部分形近对
    for a, b in pos_pairs[:12]:
        log(f"        {a}/{b}: cos={emb_dict[a] @ emb_dict[b]:.3f}")
    return a


def main():
    log(f"device={DEV} torch={torch.__version__}")
    glyphs = [tuple(x) for x in json.load(open("data/pretrained/dino_embeddings/glyph_dino_index.json", encoding="utf-8"))["glyphs"]]

    log("=" * 78)
    log("零参数 DINO 特征直接注入 —— 外形一致性判别性评测")
    log("=" * 78)

    # CLS 768
    emb768 = np.load("data/pretrained/dino_embeddings/glyph_dino_embeddings.npy").astype(np.float32)
    emb768 /= (np.linalg.norm(emb768, axis=1, keepdims=True) + 1e-8)
    char_768 = glyph_to_char(emb768, glyphs)
    log(f"\n[CLS768 原始] 字符数={len(char_768)}")
    eval_discriminative(char_768, "CLS768原始")

    # CLS 384
    emb384 = np.load("data/pretrained/dino_embeddings/glyph_dino_embeddings_384.npy").astype(np.float32)
    emb384 /= (np.linalg.norm(emb384, axis=1, keepdims=True) + 1e-8)
    char_384 = glyph_to_char(emb384, glyphs)
    log(f"\n[CLS384 原始] 字符数={len(char_384)}")
    eval_discriminative(char_384, "CLS384原始")

    # Patch-mean 768
    log("\n" + "-" * 78)
    log("真迹 Patch 变体")
    log("-" * 78)
    try:
        pat = np.load("_sync_work/dino_patch_glyph_mean.npy").astype(np.float32)
        pglyphs = [tuple(x) for x in json.load(open("_sync_work/dino_patch_index.json", encoding="utf-8"))["glyphs"]]
        log(f"[PatchMean 原始] glyphs={pat.shape}")
        char_patch = glyph_to_char(pat, pglyphs)
        log(f"  字符级 patch 字符数={len(char_patch)}")
        eval_discriminative(char_patch, "PatchMean原始")
    except Exception as e:
        log(f"  patch 数据加载失败: {e}")

    # 标准字形 (kai) 变体
    log("\n" + "-" * 78)
    log("标准字形(kai) DINO 变体 —— 最干净的外形一致性真值")
    log("-" * 78)
    def load_std(script, feat):
        emb = np.load(f"_sync_work/std_{script}_{feat}.npy").astype(np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        idx = json.load(open(f"_sync_work/std_{script}_index.json", encoding="utf-8"))
        cps = idx["codepoints"]
        return {chr(cp): emb[i] for i, cp in enumerate(cps) if 0x4e00 <= cp < 0x9fff}
    for feat in ["cls", "patch_mean"]:
        d = load_std("kai", feat)
        log(f"\n[std-kai {feat}] 字符数={len(d)}")
        eval_discriminative(d, f"stdKai{feat}")
    # li 书体 cross-check
    for feat in ["cls", "patch_mean"]:
        d = load_std("li", feat)
        log(f"\n[std-li {feat}] 字符数={len(d)}")
        eval_discriminative(d, f"stdLi{feat}")

    # 区分度
    log("\n" + "-" * 78)
    log("区分度: 同字跨书体最近邻命中率 (CLS768 原始)")
    char2rows = defaultdict(list)
    for i, (sid, cid) in enumerate(glyphs):
        char2rows[cid].append(i)
    multi = {c: r for c, r in char2rows.items() if len(r) >= 2}
    sim = emb768 @ emb768.T
    hit = tot = 0
    for cid, rows in multi.items():
        same = set(rows)
        for i in rows:
            order = np.argsort(-sim[i])
            for j in order:
                if j != i:
                    break
            tot += 1
            if j in same:
                hit += 1
    log(f"  同字跨书体 top-1 命中率 = {hit/tot*100:.2f}% ({hit}/{tot})  n_multi_char={len(multi)}")

    log(f"\n[{time.time()-_t0:.1f}s] done")


if __name__ == "__main__":
    main()
