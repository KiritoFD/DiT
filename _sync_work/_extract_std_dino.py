# -*- coding: utf-8 -*-
"""提取标准字形(kai/li 骨架图) 的 DINO cls + patch 特征。
标准字形 = 干净印刷体字形，消除书法真迹的书体混淆，用于测"外形一致性"真值。
"""
import os, sys, glob, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
DEV = "cuda"
_t0 = time.time()
def log(*a): print(" ".join(str(x) for x in a), flush=True)

BATCH = 128
DINO_SIZE = 256

def load_model():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").half().to(DEV).eval()
    return proc, model

def main():
    proc, model = load_model()
    for script in ["kai", "li"]:
        d = os.path.join("data/skel/std_skeleton_d3", script)
        files = sorted(glob.glob(os.path.join(d, "U+*.png")))
        cp_list = [int(os.path.basename(f)[2:].replace(".png", ""), 16) for f in files]
        log(f"[{script}] {len(files)} 字")
        cls_all = np.zeros((len(files), 768), dtype=np.float32)
        # patch: 只存 mean (768) 和 保留 8x8 降采样可选
        for i in range(0, len(files), BATCH):
            b_files = files[i:i+BATCH]
            imgs = [Image.open(f).convert("RGB") for f in b_files]
            inp = proc(images=imgs, return_tensors="pt").to(DEV)
            inp = {k: (v.half() if v.dtype == torch.float else v) for k, v in inp.items()}
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(**inp)
            cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()   # (B,768)
            patch = out.last_hidden_state[:, 1:, :].float().cpu().numpy()  # (B,256,768)
            pmean = patch.mean(axis=1)                                     # (B,768)
            cls_all[i:i+BATCH] = cls
            if i == 0:
                patch_mean_all = np.zeros((len(files), 768), dtype=np.float32)
            patch_mean_all[i:i+BATCH] = pmean
            log(f"  {min(i+BATCH,len(files))}/{len(files)}")
        # 归一化
        cls_all /= (np.linalg.norm(cls_all, axis=1, keepdims=True) + 1e-8)
        patch_mean_all /= (np.linalg.norm(patch_mean_all, axis=1, keepdims=True) + 1e-8)
        np.save(f"_sync_work/std_{script}_cls.npy", cls_all)
        np.save(f"_sync_work/std_{script}_patch_mean.npy", patch_mean_all)
        with open(f"_sync_work/std_{script}_index.json", "w", encoding="utf-8") as f:
            json.dump({"codepoints": cp_list, "count": len(cp_list)}, f)
        log(f"  saved std_{script}_cls {cls_all.shape} patch {patch_mean_all.shape}")
    log(f"[{time.time()-_t0:.1f}s] done")

if __name__ == "__main__":
    main()
