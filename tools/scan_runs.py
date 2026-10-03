"""扫描全部历史 run, 输出"与 eval200 口径是否可比"的兼容性表。

为什么: eval200 是 (23 槽位 + std_w7 骨架当条件 + 真迹图目标 + train.csv) 这张桌子上的集子,
        拿它去评"骨架生成器 / 87 槽位 / 50k 数据"的 ckpt 是无效对比。所以先分类, 再评。
信息源: 每个 run 目录下的 resolved_config.json (不用 torch.load 807 个 .pt, 秒级完成)。
"""
import glob
import json
import os
from collections import defaultdict

os.chdir("/root/Workspace/xy/DiT")
ROOTS = ["assets/results", "_archive/20261003_twostage"]

cfgs = []
for r in ROOTS:
    cfgs += glob.glob(os.path.join(r, "**", "resolved_config.json"), recursive=True)

rows = []
for cp in sorted(cfgs):
    run = os.path.dirname(cp)
    try:
        c = json.load(open(cp, encoding="utf-8"))
    except Exception as e:                                    # noqa: BLE001
        print(f"[skip] {cp}: {e}")
        continue
    ckdir = os.path.join(run, "checkpoints")
    cks = sorted(glob.glob(os.path.join(ckdir, "*.pt")))
    steps = [os.path.basename(x)[:-3] for x in cks]

    def g(*keys, default=""):
        for k in keys:
            v = c.get(k)
            if v not in (None, ""):
                return v
        return default

    rows.append({
        "run": run,
        "model": g("model"),
        "gl_cond": g("skel_as_glyph_cond", default=False),
        "inj": g("glyph_inject_mode", default="adaln"),
        "ncallig": g("num_calligraphers", default=2),
        "nochar": g("no_char_cond", default=False),
        "data_csv": os.path.basename(str(g("data_csv"))),
        "skel_dir": os.path.basename(str(g("skel_latent_shards_dir"))),
        "nck": len(cks),
        "steps": f"{steps[0]}..{steps[-1]}" if steps else "-",
    })

# ── 汇总 ────────────────────────────────────────────────────────────
print(f"[scan] run 总数 = {len(rows)}  (含 ckpt 的 = {sum(1 for r in rows if r['nck'])})\n")

hdr = f"{'run':<62} {'model':<16} {'glc':<4} {'inj':<6} {'ncal':<5} {'data':<28} {'skel_dir':<24} {'ck':<3} steps"
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda x: x["run"]):
    print(f"{r['run'][:62]:<62} {str(r['model'])[:16]:<16} {str(r['gl_cond'])[:4]:<4} "
          f"{str(r['inj'])[:6]:<6} {str(r['ncallig'])[:5]:<5} {r['data_csv'][:28]:<28} "
          f"{r['skel_dir'][:24]:<24} {r['nck']:<3} {r['steps']}")

# ── 可比性分类 ──────────────────────────────────────────────────────
print("\n[分类]")
groups = defaultdict(list)
for r in rows:
    if not r["nck"]:
        continue
    key = (str(r["model"]), bool(r["gl_cond"]), str(r["inj"]),
           str(r["ncallig"]), r["data_csv"], r["skel_dir"])
    groups[key].append(r)

for key, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    n = sum(x["nck"] for x in rs)
    print(f"\n  model={key[0]}  glyph_cond={key[1]}  inj={key[2]}  n_callig={key[3]}")
    print(f"    data={key[4]}  skel_dir={key[5]}")
    print(f"    run 数={len(rs)}  ckpt 数={n}")
    for x in rs[:6]:
        print(f"      - {x['run'][:80]} ({x['nck']} ckpt)")
    if len(rs) > 6:
        print(f"      ... 另 {len(rs) - 6} 个")
