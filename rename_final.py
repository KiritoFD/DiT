# -*- coding: utf-8 -*-
"""
重命名：把官方图复制到平铺结构 final/{split}/{img_id}.png。
源：MCCD/MCCD/MCCD_Character/trainset_dataset/{orig_path}
用 shutil.copy2 复制（保留官方原始数据），多进程并行。
用法：python rename_final.py --workers 8
"""
import os, sys, json, argparse, shutil
from concurrent.futures import ProcessPoolExecutor
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_ROOT = "MCCD/MCCD/MCCD_Character/trainset_dataset"
OUT_ROOT = "final"


def copy_one(rec):
    src = os.path.join(SRC_ROOT, rec["orig_path"])
    dst = os.path.join(OUT_ROOT, rec["final_split"], f"{rec['img_id']}.png")
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return rec["img_id"], True
    except Exception as e:
        return rec["img_id"], f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", default="final_manifest_split.json")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    m = json.load(open(args.manifest, encoding="utf-8"))
    if args.limit > 0:
        m = m[:args.limit]
    print("total to copy:", len(m))

    # 用 copy2 最快无需校验，但可加 --verify 选项略过
    ok = fail = 0
    fails = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (rid, status) in enumerate(ex.map(copy_one, m)):
            if status is True:
                ok += 1
            else:
                fail += 1
                fails.append((rid, status))
            if (i + 1) % 20000 == 0:
                print(f"progress {i+1}/{len(m)} ok={ok} fail={fail}", flush=True)

    print(f"Done ok={ok} fail={fail}")
    if fails:
        print("sample fails:", fails[:10])
        with open("_rename_fails.json", "w", encoding="utf-8") as f:
            json.dump(fails[:1000], f, ensure_ascii=False)


if __name__ == "__main__":
    main()
