import os, glob, re, subprocess

BASE = "/root/Workspace/xy/DiT"

# ---- 1) eval step0030000 规模 ----
exps = [
    ("s21",  "assets/results/s21_fame_flow_v2/*/checkpoints"),
    ("s25",  "assets/results/s25_ids_pretrain/*/checkpoints"),
    ("s28",  "assets/results/s28_std_dino_pretrain/*/checkpoints"),
    ("s30",  "assets/results/s30_dino_char_strong_pretrain/*/checkpoints"),
    ("s26",  "assets/results/s26_ctrl_gt_skel/*/checkpoints"),
    ("s31",  "assets/results/s31_ctrl_gt_skel_1px/*/checkpoints"),
    ("1pix", "assets/results/ctrl_fame_1px_v1/*/checkpoints"),
    ("s32b", "assets/results/s32b_repa_strong/*/checkpoints"),
    ("s32c", "assets/results/s32c_chain/*/checkpoints"),
]
print("=== eval step<=30000 ===")
total = 0
for name, g in exps:
    cands = []
    for ck in glob.glob(os.path.join(BASE, g)):
        for sub in ("eval_samples", "eval_samples_ctrl"):
            cands += glob.glob(os.path.join(ck, sub, "step*"))
    best, bs = None, -1
    for d in cands:
        try:
            s = int(os.path.basename(d).replace("step", ""))
        except ValueError:
            continue
        if s <= 30000 and s > bs:
            best, bs = d, s
    if best is None and cands:
        best = sorted(cands)[-1]
    if best:
        pngs = glob.glob(os.path.join(best, "**", "*.png"), recursive=True)
        sz = sum(os.path.getsize(f) for f in pngs)
        total += sz
        print(f"{name:5s}: {len(pngs):4d} png  {sz/1e6:7.1f} MB  {best}")
    else:
        print(f"{name:5s}: NO step dir")
print(f"TOTAL eval: {total/1e6:.1f} MB")

# ---- 2) 清洗/预处理脚本 (DiT 仓库内) ----
print("\n=== cleanse/prep scripts in DiT ===")
kw = re.compile(r"clean|preprocess|prepare|resize|verify_dataset|update_cfg|dataset|mccd", re.I)
seen = set()
for dp, dn, fn in os.walk(BASE):
    if "/.git" in dp or "/node_modules" in dp or "/checkpoints" in dp or "/results/" in dp:
        continue
    for f in fn:
        if f.endswith(".py") and kw.search(f):
            rel = os.path.relpath(os.path.join(dp, f), BASE)
            if rel not in seen:
                seen.add(rel)
                print(rel)

# ---- 3) fame 索引文件 (csv/json/tsv) 全仓库 + 常见数据根 ----
print("\n=== fame/index csv|json|tsv search ===")
roots = ["/root/Workspace/xy", "/root", "/data", "/workspace"]
hits = []
for root in roots:
    if not os.path.isdir(root):
        continue
    r = subprocess.run(
        f"find {root} -maxdepth 6 -iname '*fame*' \\( -name '*.csv' -o -name '*.json' -o -name '*.jsonl' -o -name '*.tsv' \\) 2>/dev/null",
        shell=True, capture_output=True, text=True)
    hits += [x for x in r.stdout.splitlines() if x]
for h in hits[:80]:
    print(h)
print("fame hits:", len(hits))

# 仓库内任何 csv/json 索引
print("\n=== any csv in DiT repo ===")
r = subprocess.run("find /root/Workspace/xy/DiT -maxdepth 4 -name '*.csv' 2>/dev/null", shell=True, capture_output=True, text=True)
print(r.stdout.strip() or "(none)")
