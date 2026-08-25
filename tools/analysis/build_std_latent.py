#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地 GPU VAE encode 全套标准字形 → latent 缓存(batch8 + try 降级逐张)。
进度 flush 到 progress.log。输出 kai/li 子目录 + manifest。"""
import os, sys, glob, json, time
import numpy as np
import torch
from PIL import Image

HERE=os.path.dirname(os.path.abspath(__file__))
IMG=os.path.join(HERE,"std_glyph_img"); OUT=os.path.join(HERE,"std_glyph_latent")
VAE_PATH=r"G:\GitHub\DiT\pretrained_models\sd-vae-ft-ema"
PROGRESS=os.path.join(OUT,"progress.log")
BATCH=8

def log(msg):
    with open(PROGRESS,"a",encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

def main():
    os.makedirs(OUT,exist_ok=True)
    from diffusers.models import AutoencoderKL
    torch.manual_seed(0)
    vae=AutoencoderKL.from_pretrained(VAE_PATH).to("cuda").eval()
    for p in vae.parameters(): p.requires_grad=False
    manifest={}
    for book in ["kai","li"]:
        img_dir=os.path.join(IMG,book); out_dir=os.path.join(OUT,book)
        os.makedirs(out_dir,exist_ok=True)
        files=sorted(glob.glob(os.path.join(img_dir,"U+*.png")))
        todo=[f for f in files if not os.path.exists(os.path.join(out_dir,os.path.splitext(os.path.basename(f))[0]+".npy"))]
        log(f"[{book}] 共{len(files)}, 待encode {len(todo)}")
        arr_images=[]; arr_names=[]
        def flush_batch():
            if not arr_images: return
            t=torch.stack(arr_images).to("cuda")
            with torch.no_grad():
                z=vae.encode(t).latent_dist.sample().mul_(0.18215).cpu().float()
            for nm,zz in zip(arr_names,z):
                np.save(os.path.join(out_dir,f"{nm}.npy"), zz.numpy())
            arr_images.clear(); arr_names.clear()
        done=0
        for fp in todo:
            name=os.path.splitext(os.path.basename(fp))[0]
            try:
                a=np.asarray(Image.open(fp).convert("RGB")).astype(np.float32)/127.5-1.0
            except Exception as e:
                log(f"  读图失败 {name}: {e}"); continue
            arr_images.append(torch.from_numpy(a.transpose(2,0,1)))
            arr_names.append(name)
            if len(arr_images)>=BATCH:
                try:
                    flush_batch()
                except Exception as e:
                    # 整批失败(可能是某张卡), 降级逐张
                    log(f"  批失败({e}), 降级逐张 {len(arr_names)}张")
                    singles=arr_images.copy(); snms=arr_names.copy()
                    arr_images.clear(); arr_names.clear()
                    for st,sn in zip(singles,snms):
                        try:
                            with torch.no_grad():
                                zz=vae.encode(st.unsqueeze(0).to("cuda")).latent_dist.sample().mul_(0.18215).cpu().float()
                            np.save(os.path.join(out_dir,f"{sn}.npy"),zz.numpy())
                        except Exception as e2:
                            log(f"    [skip] {sn}: {e2}")
                            continue
                done += BATCH
                if done % 200 == 0:
                    log(f"  {book} ... {done}/{len(todo)}")
            torch.cuda.empty_cache()
        flush_batch()
        done = len(os.listdir(out_dir))
        log(f"[{book}] 完成, 共 {done} 张")
        for name in [os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith('.npy')]:
            manifest[f"{book}/{name}"]=f"std_glyph_latent/{book}/{name}.npy"
    with open(os.path.join(OUT,"manifest.json"),"w",encoding="utf-8") as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2)
    log(f"manifest {len(manifest)} 条")
    return 0

if __name__=="__main__":
    sys.exit(main())
