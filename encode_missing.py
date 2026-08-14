# -*- coding: utf-8 -*-
"""对缺失的官方图 VAE encode，生成 latent，存 latent_missing/{img_id}.npy (float16)。"""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, cv2, torch
from diffusers.models import AutoencoderKL
from concurrent.futures import ThreadPoolExecutor

SRC_ROOT = "MCCD/MCCD/MCCD_Character/trainset_dataset"
OUT_DIR = "latent_missing"
SIZE = 256
LATENT_SCALE = 0.18215
VAE_PATH = "pretrained_models/sd-vae-ft-ema"

def read_image(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def main():
    missing = json.load(open("latent_missing.json", encoding="utf-8"))
    print("missing to encode:", len(missing))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    os.makedirs(OUT_DIR, exist_ok=True)

    BATCH = 20
    pool = ThreadPoolExecutor(max_workers=8)
    done = fail = 0
    start = time.time()
    for i in range(0, len(missing), BATCH):
        chunk = missing[i:i+BATCH]
        futs = {r["img_id"]: pool.submit(read_image, os.path.join(SRC_ROOT, r["orig_path"])) for r in chunk}
        imgs, ids = [], []
        for rid, f in futs.items():
            img = f.result()
            if img is None:
                fail += 1
                continue
            imgs.append(img); ids.append(rid)
        if not imgs:
            continue
        tensors = []
        for img in imgs:
            r = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            tensors.append(torch.from_numpy(rgb).permute(2,0,1))
        x = torch.stack(tensors).to(device)*2.0-1.0
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.mean.float().mul_(LATENT_SCALE).cpu().numpy()
        for rid, l in zip(ids, lat):
            np.save(os.path.join(OUT_DIR, f"{rid}.npy"), l.astype(np.float16))
        done += len(ids)
        if (i//BATCH+1) % 20 == 0:
            el = time.time()-start
            print(f"  {done}/{len(missing)} {done/el:.1f}/s fail={fail}", flush=True)
    el = time.time()-start
    print(f"Done encode={done} fail={fail} in {el:.0f}s", flush=True)

if __name__ == "__main__":
    main()
