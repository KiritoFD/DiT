#!/bin/bash
# prep_mode_shards.sh -- 为 item 1（目标 sample -> mode）准备 mode 版 Calli latent shards
#   与 v72 的 sample 版【逐项同规格】: 同 CSV / 同 VAE / 同 shard-size / 同 img_root
#   唯一差异: --encode-mode mode
set -eu
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python

CSV=exp-std/csv/train_top10_aug_sym.csv
OUT=exp-std/data/shards_img_aug_calli_kl1e6_s12500_mode
VAE=experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv
IMG_ROOT=data/top10_style23/imgs

echo "===== 0. 前置检查 ====="
( cd "$VAE" && md5sum -c MD5SUMS.txt ) | sed 's/^/  /'
[ -f "$CSV" ] && echo "  CSV rows = $(wc -l < "$CSV")" || { echo "  CSV 缺失"; exit 1; }
echo "  GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

echo
echo "===== 1. 编码 (encode-mode = mode) ====="
PYTHONPATH=. $PY tools/encode_aug_latents.py \
  --csv "$CSV" --out "$OUT" --vae "$VAE" --encode-mode mode \
  --img-root . --batch 64 --shard-size 5120 --workers 8

echo
echo "===== 2. 自检 + 溯源 ====="
PYTHONPATH=. $PY - "$OUT" "$VAE" "$IMG_ROOT" <<'PYEOF'
import json, os, sys, glob
import numpy as np, torch
from PIL import Image
from diffusers.models import AutoencoderKL

out, vae_dir, img_root = sys.argv[1], sys.argv[2], sys.argv[3]
sf = 0.18215
shards = sorted(glob.glob(os.path.join(out, "shard_*.npz")))
names = sorted(glob.glob(os.path.join(out, "names_*.txt")))
n_rows = 0
lats = []
for sp in shards[:3]:
    d = np.load(sp)
    key = "latents" if "latents" in d else list(d.keys())[0]
    lats.append(d[key])
    n_rows += d[key].shape[0]
allat = np.concatenate(lats, 0).astype(np.float32)
print(f"  n_shards = {len(shards)}   抽样行 = {n_rows}")
print(f"  latent std (masked) = {allat.std():.4f}   mean = {allat.mean():+.4f}")
print(f"    参考: v72(sample) 实测 std = 0.1955 ; 预期 mode 会更小(去掉后验内方差)")

dev = torch.device("cuda:0")
vae = AutoencoderKL.from_pretrained(vae_dir).to(dev).eval()
def L1(lat, img):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rec = vae.decode(lat.to(dev) / sf).sample.float()
    r = ((rec.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
    return float(np.abs(r - img).mean())

# 取第一个 shard 的前 4 行，找回原图做 L1 自检
d0 = np.load(shards[0])
key = "latents" if "latents" in d0 else list(d0.keys())[0]
lat4 = torch.from_numpy(d0[key][:4].astype(np.float32))
# 原图路径：从 CSV 顺序取（编码器按 CSV 顺序写 shard）
import csv as _csv
rows = list(_csv.DictReader(open("exp-std/csv/train_top10_aug_sym.csv", encoding="utf-8")))
imgs = []
for r in rows[:4]:
    p = r.get("image_path", "")
    p = p if os.path.isabs(p) else os.path.join(".", p)
    im = Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)
    imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
imgs = np.stack(imgs)
print(f"  selfcheck decode(lat/sf) L1 = {L1(lat4, imgs):.4f}   (v72 sample 版为 0.0072)")
prov = {
    "artifact": "DiT latent shards (v74 / item-1 : mode target)",
    "shards_dir": out, "n_shards": len(shards),
    "vae_dir": vae_dir, "vae_step": 12500, "vae_run": "calli_vae_dino_stdconv_kl1e6",
    "encode_convention": "z_stored = posterior.mode() * scaling_factor",
    "decode_convention": "x = vae.decode(z_stored / scaling_factor)",
    "scaling_factor": sf, "encode_mode": "mode (确定性目标; 与 v72 的 sample 唯一差异)",
    "latent_std_measured": round(float(allat.std()), 4),
    "source_csv": "exp-std/csv/train_top10_aug_sym.csv", "img_root": img_root,
    "note": "★ 与 shards_img_aug_calli_kl1e6_s12500_rsample (sample*sf) 只差 encode-mode, 勿混用",
    "created": "2026-10-10", "ok": True,
}
json.dump(prov, open(os.path.join(out, "PROVENANCE.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"  PROVENANCE -> {out}/PROVENANCE.json")
PYEOF

echo
echo "===== 3. 产物 ====="
du -sh "$OUT"; ls "$OUT" | head -5
echo "  文件数: $(ls "$OUT" | wc -l)"
