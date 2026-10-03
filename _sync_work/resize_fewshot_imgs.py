"""把 few-shot 的 wild 图统一 resize 到 256x256，写副本，并把 CSV 指向副本。

⚠ 为什么: 评测读 GT 图算 ssim，而 wild 原始图尺寸不一(实测有 288x256)，
   直接读会 broadcast 失败。编码 latent 时用的是 transforms.Resize((256,256))，
   所以这里用同样的 resize —— 副本与 latent **一致**。
"""
import csv
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
Image.MAX_IMAGE_PIXELS = None
SIZE = 256
CALS = ("怀素", "伊秉绶", "徐渭", "沈周")


def resize_one(args):
    src, dst = args
    if os.path.exists(dst):
        return True
    try:
        im = Image.open(src).convert("RGB")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.LANCZOS)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.save(dst)
        return True
    except Exception as e:
        print(f"  ✗ {src}: {e}")
        return False


for c in CALS:
    out_dir = f"data/50k/fs_imgs_{c}"
    jobs = []
    for name in ("train", "eval"):
        p = f"assets/fs_{c}_{name}.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        for r in rows:
            src = r["image_path"]
            if not os.path.isabs(src):
                src = os.path.join(ROOT, src)
            # 目标文件名沿用原名（字符），不同字不会撞
            base = os.path.basename(src)
            dst = os.path.join(ROOT, out_dir, base)
            jobs.append((src, dst))
    with ThreadPoolExecutor(max_workers=16) as ex:
        ok = sum(ex.map(resize_one, jobs))
    print(f"  {c}: {ok}/{len(jobs)} 张已 resize -> {out_dir}")

    # CSV 指向副本
    for name in ("train", "eval"):
        p = f"assets/fs_{c}_{name}.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        cols = list(rows[0].keys())
        for r in rows:
            base = os.path.basename(r["image_path"])
            r["image_path"] = os.path.join(out_dir, base)
            r["src_image_path"] = os.path.join(out_dir, base)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print(f"    CSV 已更新")
