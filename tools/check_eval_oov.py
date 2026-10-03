"""查 eval 集的「字」是否在训练（表）里 —— 决定字表实验会不会白跑。

父问题: v52 用「字×书体表」替代 std 骨架。表只对**训练里出现过的 glyph** 有效;
eval 集若含训练没见过的字 -> 那些行只能拿到随机行 -> 指标无意义 -> 实验白跑。

同时查「用全体跑」的可行性: 预训练要 DINO 特征; 若缓存不覆盖 eval 图,
想训全体也训不了 (得先补提特征)。

用法: python tools/check_eval_oov.py
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))

TRAIN = "exp-std/csv/train.csv"
EVALS = [("eval200fix", "exp-std/csv/eval200_fixed.csv"),
         ("seen20", "exp-std/csv/seen20.csv")]
CACHE = "data/dino_cache/top10_v1"


def load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def cols_of(rows):
    return list(rows[0].keys()) if rows else []


def key_of(r):
    """优先用 glyph_id; 没有就用 script_id+char_id 合成。"""
    if "glyph_id" in r and r["glyph_id"] not in ("", None):
        try:
            return int(r["glyph_id"]), "glyph_id"
        except ValueError:
            pass
    for sc, ch in (("script_id", "char_id"), ("script_id", "character")):
        if sc in r and ch in r and r[sc] not in ("", None):
            try:
                return int(r[sc]) * 7026 + hash(r[ch]) % 7026, f"{sc}+{ch}(伪)"
            except ValueError:
                pass
    return None, "?"


def main():
    tr = load(TRAIN)
    if not tr:
        raise SystemExit(f"读不到 {TRAIN}")
    print(f"[train] {len(tr)} 行, 列: {cols_of(tr)}")

    tset, tchars, tmode = set(), set(), "?"
    for r in tr:
        k, m = key_of(r)
        tmode = m
        if k is not None:
            tset.add(k)
        if "character" in r:
            tchars.add(r["character"])
    print(f"[train] 唯一 glyph 键 {len(tset)} 个 (口径 {tmode}), 唯一 character {len(tchars)} 个")

    # ---- DINO 缓存覆盖 (决定能否"用全体跑") ----
    print(f"\n[cache] {CACHE}")
    if os.path.isdir(CACHE):
        files = glob.glob(os.path.join(CACHE, "**", "*"), recursive=True)
        nf = sum(1 for f in files if os.path.isfile(f))
        ext = Counter(os.path.splitext(f)[1] for f in files if os.path.isfile(f))
        print(f"  文件 {nf} 个, 扩展名分布 {dict(ext)}")
        idxs = glob.glob(os.path.join(CACHE, "**", "index*.json"), recursive=True)
        for i in idxs[:3]:
            try:
                d = json.load(open(i, encoding="utf-8"))
                n = len(d) if hasattr(d, "__len__") else "?"
                print(f"  {i}: {type(d).__name__} n={n}")
                if isinstance(d, dict):
                    ks = list(d.keys())[:5]
                    print(f"    样例键: {ks}")
            except Exception as e:                                # noqa: BLE001
                print(f"  {i}: 解析失败 {e}")
        npz = glob.glob("assets/dino_feat_top10.npz")
        if npz:
            import numpy as np
            z = np.load(npz[0])
            print(f"  ★ assets/dino_feat_top10.npz 键: {z.files}")
            for k in z.files:
                print(f"      {k}: {z[k].shape}")
    else:
        print("  (目录不存在)")

    # ---- eval 的 OOV 检查 ----
    overview = {}
    for name, path in EVALS:
        rows = load(path)
        if not rows:
            print(f"\n[{name}] 读不到 {path}")
            continue
        keys, chars = set(), Counter()
        oov_rows, oov_chars = [], Counter()
        for r in rows:
            k, _ = key_of(r)
            ch = r.get("character", "?")
            keys.add(k)
            chars[ch] += 1
            if k not in tset:
                oov_rows.append(r)
                oov_chars[ch] += 1
        print(f"\n[{name}] {len(rows)} 行, 列: {cols_of(rows)}")
        print(f"  唯一 glyph 键 {len(keys)}, 唯一 character {len(chars)}")
        print(f"  ★ glyph 不在训练表里的行: {len(oov_rows)}/{len(rows)} "
              f"({100*len(oov_rows)/len(rows):.1f}%)")
        if oov_rows:
            print(f"    涉及 {len(oov_chars)} 个字, 前 15: "
                  f"{list(oov_chars.items())[:15]}")
        ch_oov = {c for c in chars if c not in tchars}
        print(f"  仅按 character 看 (跨书体): 不在训练里的字 {len(ch_oov)}/{len(chars)}")
        if ch_oov:
            print(f"    例: {list(ch_oov)[:20]}")
        # 书家是否 OOV (次要)
        if "calligrapher" in rows[0] or "callig_id" in rows[0]:
            cc = "calligrapher" if "calligrapher" in rows[0] else "callig_id"
            tcallig = {r.get(cc) for r in tr}
            eco = {r.get(cc) for r in rows}
            print(f"  书家: eval {len(eco)} 个, 训练 {len(tcallig)} 个, "
                  f"eval 里不在训练的 = {sorted(x for x in eco if x not in tcallig)[:8]}")
        overview[name] = (len(rows), len(oov_rows), len(oov_chars), len(chars), len(ch_oov))

    # ---- 若要用全体 (train ∪ eval) 训表 ----
    union_keys = set(tset)
    for name, path in EVALS:
        rows = load(path)
        if not rows:
            continue
        for r in rows:
            k, _ = key_of(r)
            if k is not None:
                union_keys.add(k)
    print(f"\n[结论数据] train ∪ eval 的唯一 glyph 键 = {len(union_keys)} "
          f"(纯 train = {len(tset)})")
    print(f"  config 里的 num_characters = 35130 (表行数)")
    all_oov = sum(v[1] for v in overview.values())
    print(f"\n{'='*66}")
    if all_oov == 0:
        print("结论: ✓ 两个 eval 集的字**全部**在训练表内 -> 字表实验可以直接跑")
    else:
        print(f"结论: ✗ 共 {all_oov} 行 eval 的字不在训练表里 -> 这些行会拿到随机行, 指标无意义")
        print("      -> 按你的方案: **预训练表时用全体 (train ∪ eval)**, 让表覆盖 eval 的字")
        print("      -> 但需先确认 DINO 特征覆盖 eval 图 (见上面 [cache] 段); 不够就得补提特征")
    return 0


if __name__ == "__main__":
    sys.exit(main())
