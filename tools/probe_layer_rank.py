# -*- coding: utf-8 -*-
"""T3 — 层位分析：哪几层的风格可见性最低（作为 S2 --local-ca-at 候选）。

复用 D1 已落盘的 per_layer.delta_mod（每个 block 的 Δmod 相对值），
跨 7 个 ckpt 聚合，看层排名是否稳定。

用法:
    python tools/probe_layer_rank.py --d1-dir assets/d1_remote/assets
"""
import argparse
import glob
import io
import json
import os


def load_per_layer(d1_dir):
    """返回 {ckpt: [ (layer_idx, dmod_layer_value) ... ]}。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d1_dir, "d1_*.json"))):
        tag = os.path.basename(f)[3:-5]
        try:
            d = json.load(io.open(f, encoding="utf-8"))
        except Exception:
            continue
        pl = d.get("per_layer")
        if not pl:
            continue
        # per_layer 的每个条目是 dict: {shift_msa, scale_msa, gate_msa,
        # shift_mlp, scale_mlp, gate_mlp, _mean}
        rows = []
        for k, v in pl.items():
            rows.append((int(k), float(v.get("_mean", 0.0)), v))
        rows.sort()
        out[tag] = {"rows": rows, "delta_mod_mean": d.get("delta_mod_mean")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-dir", default="assets/d1_remote/assets")
    ap.add_argument("--top", type=int, default=3, help="每 ckpt 取最低几层")
    a = ap.parse_args()

    data = load_per_layer(a.d1_dir)
    if not data:
        print("未读到 per_layer")
        return

    n_layers = max(len(v["rows"]) for v in data.values())
    tags = list(data.keys())
    print("=" * 78)
    print("T3 — 逐层 Δmod（风格对 adaLN 的可见性，按层）")
    print("=" * 78)
    print("ckpt 数 = %d, 层数 = %d" % (len(tags), n_layers))

    # ---- 表 1：逐层 Δmod（行=层，列=ckpt）
    print()
    hdr = "%-5s" % "层"
    for t in tags:
        hdr += " %11s" % t[:11]
    hdr += " %10s" % "跨ckpt均值"
    print(hdr)
    print("-" * len(hdr))

    # 收集每层的跨 ckpt 值
    layer_vals = {}   # layer -> [values]
    for t in tags:
        for idx, val, _ in data[t]["rows"]:
            layer_vals.setdefault(idx, []).append(val)

    mean_by_layer = {}
    for idx in range(n_layers):
        vs = layer_vals.get(idx, [])
        mean_by_layer[idx] = sum(vs) / len(vs) if vs else None

    for idx in range(n_layers):
        line = "%-5d" % idx
        for t in tags:
            vs = {i: v for i, v, _ in data[t]["rows"]}
            line += " %11.4f" % vs[idx] if idx in vs else " %11s" % "-"
        m = mean_by_layer[idx]
        line += " %10s" % (("%.4f" % m) if m is not None else "-")
        print(line)

    # ---- 表 2：跨 ckpt 的层排名（越低越差）
    print()
    print("[排名] 每层在 %d 个 ckpt 中的名次均值（1 = 该 ckpt 里最低的层）" % len(tags))
    print("%-5s %10s %10s %10s" % ("层", "平均名次", "最低次数", "Δmod均值"))
    print("-" * 40)
    ranks = {}
    low_count = {}
    for t in tags:
        rows = sorted(data[t]["rows"], key=lambda r: r[1])   # 升序：最低在前
        for r, (idx, _val, _) in enumerate(rows):
            ranks.setdefault(idx, []).append(r + 1)
            if r < a.top:
                low_count[idx] = low_count.get(idx, 0) + 1

    order = sorted(range(n_layers),
                   key=lambda i: (sum(ranks.get(i, [99])) / max(1, len(ranks.get(i, [1])))))
    for idx in order:
        rr = ranks.get(idx, [])
        print("%-5d %10.2f %10d %10s"
              % (idx, sum(rr) / len(rr) if rr else -1,
                 low_count.get(idx, 0),
                 ("%.4f" % mean_by_layer[idx]) if mean_by_layer[idx] is not None else "-"))

    # ---- 稳定性检查：排名是否跨 ckpt 一致
    print()
    print("[稳定性] 各 ckpt 的最低 %d 层（看是否指向同一批层）" % a.top)
    for t in tags:
        rows = sorted(data[t]["rows"], key=lambda r: r[1])
        lows = [r[0] for r in rows[:a.top]]
        print("  %-18s 最低层 = %s    (dmod整体 %.4f)"
              % (t, lows, data[t]["delta_mod_mean"] or 0))

    print()
    print("[建议] 跨 ckpt 最稳定的低可见性层（按平均名次前 4）：")
    top4 = order[:4]
    print("  --local-ca-at %s" % (" ".join(str(i) for i in sorted(top4))))
    print()
    print("  解读：若低层（0-3）稳定垫底 -> 早期层做全局风格，局部通路应插在")
    print("        中后段（5-10），让局部细节在结构已定后注入；")
    print("        若高/中层垫底 -> 直接把 local_ca 插在那里。")


if __name__ == "__main__":
    main()
