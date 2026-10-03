#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== 训练日志 ==="
tail -12 /tmp/pix64_train.log 2>/dev/null
echo
echo "=== GT 骨架各宽度墨量 (同一批 id) ==="
/opt/conda/envs/cu121/bin/python - <<'PY'
import glob, os
import numpy as np
from PIL import Image
ids = sorted(os.path.basename(f)[:-4] for f in glob.glob("data/top10_style23/gt_skel_png/*.png"))[:200]
for d in ["gt_skel_png", "gt_skel_png_w1", "gt_skel_png_w3", "gt_skel_png_w7"]:
    fs = [f"data/top10_style23/{d}/{i}.png" for i in ids]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        print(f"  {d:>18}: 无文件")
        continue
    v = [ (np.asarray(Image.open(f).convert("L")) < 128).mean() for f in fs[:100] ]
    print(f"  {d:>18}: n={len(fs)} ink={np.mean(v):.4f}")
PY
echo
echo "=== v26 训练用的 g (aux_skel3) 解码墨量 ==="
/opt/conda/envs/cu121/bin/python - <<'PY'
import glob, numpy as np, torch as th, os
os.chdir("/root/Workspace/xy/DiT")
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("data/pretrained/pretrained_models/sd-vae-ft-ema").cuda().eval()
for d in ["data/top10_style23/shards_aux_skel3", "data/top10_style23/shards_std"]:
    fs = sorted(glob.glob(d + "/shard_*.npz"))[:1]
    with np.load(fs[0]) as z:
        lat = z["latents"][:32]
    t = th.from_numpy(np.asarray(lat, np.float32)).cuda()
    with th.no_grad():
        b = (vae.decode(t/0.18215).sample.mean(1) < 0).float().mean().item()
    print(f"  {d:>40}: 解码墨量 {b:.4f}")
PY
