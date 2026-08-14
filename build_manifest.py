# -*- coding: utf-8 -*-
"""
生成官方图 manifest：
遍历 MCCD_Character/trainset_dataset/{train,test} 下所有图片，
从文件名解析 (字, 字体, 书家/出处, 样本id)，用映射表转成数字 id，
分配全局唯一数字 id，输出 manifest。

manifest 每项：
{
  "img_id": 0..329714,          # 全局唯一图片 id
  "split": "train"/"test",      # 官方原始 split
  "char_id": int, "script_id": int, "calli_id": int,
  "orig_path": "train/楚/楚-...-66275.png",   # 相对 trainset_dataset 的原始路径
  "orig_char": "楚", "orig_script": "篆", "orig_calli": "集古文韵上声韵第三",
  "orig_seq": 66275             # 官方 parts[-1]（同字内序号）
}
输出 final_manifest.json（329715 条）。
"""
import os, sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = "MCCD/MCCD/MCCD_Character/trainset_dataset"
OTHERS = "others"


def clean_calligrapher(cal):
    s = cal.strip()
    if s in ("", "null", "None", "nan") or s.isdigit():
        return OTHERS
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--maps", default="_id_maps.json")
    ap.add_argument("--out", default="final_manifest.json")
    ap.add_argument("--start-id", type=int, default=0)
    args = ap.parse_args()

    maps = json.load(open(args.maps, encoding="utf-8"))
    cid = maps["character"]
    sid = maps["script"]
    kal = maps["calligrapher"]

    manifest = []
    img_id = args.start_id
    skipped = 0
    for split in ("train", "test"):
        sd = os.path.join(args.root, split)
        if not os.path.isdir(sd):
            continue
        for ch in sorted(os.listdir(sd)):
            cd = os.path.join(sd, ch)
            if not os.path.isdir(cd):
                continue
            for fn in sorted(os.listdir(cd)):
                if not fn.endswith(".png"):
                    continue
                base = fn[:-4]
                parts = base.split("-")
                if len(parts) == 5:
                    char, script, cali, seq = parts[0], parts[1], parts[3], parts[4]
                elif len(parts) == 6:
                    # 字-字体-朝代-书家-碑帖-id，碑帖忽略
                    char, script, cali, seq = parts[0], parts[1], parts[3], parts[5]
                else:
                    skipped += 1
                    continue
                cali_raw = cali                       # 原始 parts[3]（用于匹配远程目录名）
                cali = clean_calligrapher(cali)       # 清洗后的（用于 id）
                rec = {
                    "img_id": img_id,
                    "split": split,
                    "char_id": cid[char],
                    "script_id": sid[script],
                    "calli_id": kal[cali],
                    "orig_path": os.path.join(split, ch, fn).replace("\\", "/"),
                    "orig_char": char,
                    "orig_script": script,
                    "orig_calli": cali,
                    "orig_calli_raw": cali_raw,
                    "orig_seq": seq,
                }
                manifest.append(rec)
                img_id += 1

    manifest.sort(key=lambda r: r["img_id"])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print("total:", len(manifest), "skipped:", skipped)
    print("written", args.out)

    # 汇总
    from collections import Counter
    c = Counter(r["split"] for r in manifest)
    print("by split:", dict(c))
    print("img_id range:", manifest[0]["img_id"], "-", manifest[-1]["img_id"])


if __name__ == "__main__":
    main()
