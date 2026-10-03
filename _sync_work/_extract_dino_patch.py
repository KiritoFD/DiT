# -*- coding: utf-8 -*-
"""提取 glyph 级 DINO patch 特征（采样），用于外形一致性研究。

策略
----
- 从 train_fame.csv 采样：每个 glyph (script_id, character_id) 最多取 n 张图。
- 用 DINOv2-base 提取 patch tokens (16x16x768)，按 glyph 平均。
- 输出: _sync_work/dino_patch_glyph_mean.npy  (glyphs x 768)  （patch 平均池化）
         _sync_work/dino_patch_glyph_full.npy  (glyphs x 256 x 768)（可选, 空间保留）
         _sync_work/dino_patch_index.json      [(sid,cid)]
"""
import os, sys, csv, json, time, io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from collections import defaultdict

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
DEV = "cuda"
_t0 = time.time()
def log(*a): print(" ".join(str(x) for x in a), flush=True)

CSV = "assets/train_fame.csv"
OUT_DIR = "_sync_work"
PER_GLYPH = 2        # 每 glyph 最多取 2 张（平衡覆盖与成本）
BATCH = 256
DINO_SIZE = 256

def load_model():
    from transformers import AutoModel
    log("Loading facebook/dinov2-base (fp16)...")
    model = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.float16)
    model.eval().to(DEV)
    return model

def read_img(path):
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def batch_to_tensor(imgs):
    arr = np.stack(imgs)
    t = torch.from_numpy(arr).to(DEV, dtype=torch.float16).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(DINO_SIZE, DINO_SIZE), mode="bilinear", align_corners=False) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV, dtype=torch.float16).view(1, 3, 1, 1)
    return (t - mean) / std

def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    # 每 glyph 采样
    picked = []
    cnt = defaultdict(int)
    for r in rows:
        g = (int(r["script_id"]), int(r["character_id"]))
        if cnt[g] >= PER_GLYPH:
            continue
        p = os.path.join(ROOT, r["image_path"].replace("final_images", "data/imgs/final_imgs_256"))
        if not os.path.isfile(p):
            p = os.path.join(ROOT, r["image_path"])
        if not os.path.isfile(p):
            continue
        cnt[g] += 1
        picked.append((p, g[0], g[1]))
    log(f"picked {len(picked)} images across {len(cnt)} glyphs (PER_GLYPH={PER_GLYPH})")

    model = load_model()

    # 按 glyph 累计 patch (mean)
    glyph_list = sorted(cnt.keys())
    g2i = {g: i for i, g in enumerate(glyph_list)}
    acc = np.zeros((len(glyph_list), 768), dtype=np.float64)
    ccount = np.zeros(len(glyph_list), dtype=np.int32)

    total = len(picked)
    for i in range(0, total, BATCH):
        batch = picked[i:i+BATCH]
        imgs = [read_img(p) for p, _, _ in batch]
        ok = [(j, im) for j, im in enumerate(imgs) if im is not None]
        if not ok:
            continue
        tensors = batch_to_tensor([im for _, im in ok])
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(tensors)
        patch = out.last_hidden_state[:, 1:, :].float().cpu().numpy()  # (B,256,768)
        for k, (j, _) in enumerate(ok):
            sid, cid = batch[j][1], batch[j][2]
            gi = g2i[(sid, cid)]
            acc[gi] += patch[k].mean(axis=0)   # patch mean -> 768
            ccount[gi] += 1
        done = min(i + BATCH, total)
        el = time.time() - _t0
        log(f"  {done}/{total}  {done/el:.0f}/s  gpu={torch.cuda.memory_allocated()/1e9:.1f}GB")

    acc /= np.maximum(ccount[:, None], 1)
    acc = acc.astype(np.float32)
    # 归一化
    acc = acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-8)

    np.save(os.path.join(OUT_DIR, "dino_patch_glyph_mean.npy"), acc)
    with open(os.path.join(OUT_DIR, "dino_patch_index.json"), "w", encoding="utf-8") as f:
        json.dump({"glyphs": [[s, c] for s, c in glyph_list], "count": len(glyph_list)}, f)
    log(f"Saved dino_patch_glyph_mean.npy {acc.shape}  glyphs={len(glyph_list)}")
    log(f"[{time.time()-_t0:.1f}s] done")

if __name__ == "__main__":
    main()
