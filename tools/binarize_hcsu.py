"""HCSU 三源 (wild / tie / bei) 的**独立二值化**步骤。

设计原则（按用户要求）:
  1. **只做二值化**（wild 额外需要极性归一化, 见下），不裁切、不缩放、不改几何。
     归一化到 256x256 是**后续独立一步**（tie/bei 已是 256x256, 只有 wild 需要）。
  2. **原始数据只读保留**，产物写到独立目录 `_bin/`，随时可删可重跑。

极性处理:
  - tie/bei 实测 100% 白底 -> 无需处理
  - wild  实测 41.3% 白底 / 24.3% 灰底 / 34.4% 黑底 -> 黑底那批是**拓片式反相**,
    必须 `255 - a` 反转。判定用边框均值 < 0.5。
    ⚠ 现有管线 `build_dataset_px60.py` 遇到反相图是**直接丢弃**
      (文件头注释: UniCalli 被整源剔除的原因之一 "另有黑底拓片"),
      从没写过极性处理。所以这一步是新增的, 不是复用。

二值化阈值: **128**（与 `build_dataset_px60.py:65` `a < 128`、
`build_std_skel_px60.py:87` `skeletonize(a < 127)` 保持一致）。

书体范围: 只保留 楷/行/隶（用户决定; 草/篆没有对应字体渲染不出 std 骨架）。
"""
import argparse, collections, json, os, sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HCSU = "/root/Workspace/xy/HCSU"
KEEP_SCRIPTS = {"楷", "行", "隶"}
THRESH = 128
POLARITY_BORDER = 0.5      # 边框均值 < 0.5 判为反相
BLANK_FRAC = 0.002         # 二值化后墨点占比 < 0.2% 判为空白/无效


def process(task):
    src, dst = task
    try:
        im = Image.open(src)
        a = np.asarray(im.convert("L"))
    except Exception as e:
        return ("readfail", src, str(e)[:60], None, None)

    h, w = a.shape
    g = a.astype(np.float32) / 255.0
    border = float(np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]]).mean())
    # 极性判据: **笔画相对背景是深还是浅**（墨水是少数派）。
    # 比"边框均值<0.5"稳: 实测两判据一致率 98.3%, 不一致的 42/2500 全是边框
    # 落在 0.46~0.50 的临界样本 —— 旧判据在那批上会翻车。
    # 若两侧都几乎没有极端像素(极稀疏字), 退回边框均值判据。
    n_dark = int((g < border - 0.25).sum())
    n_light = int((g > border + 0.25).sum())
    if n_dark + n_light < 0.001 * g.size:
        inverted = border < POLARITY_BORDER
    else:
        inverted = n_light > n_dark
    if inverted:
        a = 255 - a

    b = (a < THRESH)
    ink = float(b.mean())
    if ink < BLANK_FRAC:
        return ("blank", src, f"ink={ink:.5f}", float(border), inverted)
    if ink > 0.90:
        return ("toosolid", src, f"ink={ink:.5f}", float(border), inverted)

    out = np.where(b, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    Image.fromarray(out, mode="L").save(dst, optimize=True)
    return ("ok", src, "", float(border), inverted)


def collect(root, tag, keep_scripts=KEEP_SCRIPTS, only_scripts=True):
    tasks, meta = [], []
    for dn in sorted(os.listdir(root)):
        dp = os.path.join(root, dn)
        if not os.path.isdir(dp):
            continue
        c, _, s = dn.rpartition("-")
        if only_scripts and s not in keep_scripts:
            continue
        for fn in sorted(os.listdir(dp)):
            if not fn.lower().endswith(".png"):
                continue
            tasks.append((os.path.join(dp, fn),
                          os.path.join(f"{HCSU}/_bin/{tag}", dn, fn)))
            meta.append((c, s, os.path.splitext(fn)[0]))
    return tasks, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="wild,tie,bei",
                    help="逗号分隔; wild/tie/bei 分别对应 wild_extract/tie_extract/bei_extract")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-root", default=f"{HCSU}/_bin")
    ap.add_argument("--all-scripts", action="store_true",
                    help="不限制书体(默认只留 楷/行/隶)")
    args = ap.parse_args()

    roots = {"wild": f"{HCSU}/wild_extract",
             "tie": f"{HCSU}/tie_extract",
             "bei": f"{HCSU}/bei_extract"}

    report = {}
    for tag in args.sources.split(","):
        tag = tag.strip()
        root = roots[tag]
        if not os.path.isdir(root):
            print(f"[skip] {tag}: {root} 不存在"); continue
        tasks, meta = collect(root, tag, only_scripts=not args.all_scripts)
        print(f"[{tag}] 待处理 {len(tasks)} 张 (书体={'全部' if args.all_scripts else sorted(KEEP_SCRIPTS)})")

        stats = collections.Counter()
        inv = collections.Counter()
        borders = []
        with Pool(args.workers) as pool:
            for i, (st, src, msg, bd, iv) in enumerate(
                    pool.imap_unordered(process, tasks, chunksize=64)):
                stats[st] += 1
                if bd is not None:
                    borders.append(bd)
                    inv[(st, iv)] += 1
                if st != "ok" and stats[st] <= 3:
                    print(f"   [{st}] {os.path.relpath(src, root)}  {msg}")
                if (i + 1) % 5000 == 0:
                    print(f"   ...{i+1}/{len(tasks)}")

        per_callig = collections.Counter()
        per_script = collections.Counter()
        for st, (c, s, ch) in zip(["ok"] * len(meta), meta):
            per_callig[c] += 1
            per_script[s] += 1
        report[tag] = dict(
            total=len(tasks), ok=stats["ok"], blank=stats["blank"],
            toosolid=stats["toosolid"], readfail=stats["readfail"],
            inverted=sum(v for (s, iv), v in inv.items() if iv),
            n_calligrapher=len(per_callig), per_script=dict(per_script),
            border_mean=float(np.mean(borders)) if borders else None)
        r = report[tag]
        print(f"   -> ok={r['ok']}  blank={r['blank']}  toosolid={r['toosolid']}  "
              f"readfail={r['readfail']}  **反相={r['inverted']}**")

    os.makedirs(f"{HCSU}/_bin", exist_ok=True)
    with open(f"{HCSU}/_bin/_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== 汇总 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
