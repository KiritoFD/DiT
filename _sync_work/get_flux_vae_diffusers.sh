#!/bin/bash
# 取 diffusers 格式的 FLUX VAE (black-forest-labs/FLUX.1-schnell 的 vae/)
cd /root/Workspace/xy/DiT || exit 1
OUT=data/pretrained/flux_vae_diffusers
mkdir -p "$OUT"
for F in vae/config.json vae/diffusion_pytorch_model.safetensors; do
  B=$(basename "$F")
  curl -L -sS -m 1800 -o "$OUT/$B" \
    "https://www.modelscope.cn/api/v1/models/black-forest-labs/FLUX.1-schnell/repo?Revision=master&FilePath=${F}" \
    || { echo "$F 失败"; exit 3; }
  printf '  %-34s %s bytes\n' "$B" "$(stat -c%s "$OUT/$B")"
done
echo "--- config.json ---"; cat "$OUT/config.json"; echo
echo "--- safetensors 键概览 ---"
/opt/conda/envs/cu121/bin/python - <<'PY'
from safetensors import safe_open
from collections import Counter
p = "data/pretrained/flux_vae_diffusers/diffusion_pytorch_model.safetensors"
with safe_open(p, framework="pt") as f:
    ks = list(f.keys())
print("  键数 =", len(ks))
print("  前缀:", dict(Counter(k.split('.')[0] for k in ks)))
print("  含 quant_conv:", any("quant_conv" in k for k in ks),
      "| 含 post_quant_conv:", any("post_quant_conv" in k for k in ks))
for w in ("encoder.conv_out.weight", "decoder.conv_in.weight"):
    if w in ks:
        print(f"  {w} -> {tuple(f.get_slice(w).get_shape())}")
PY
