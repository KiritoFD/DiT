#!/bin/bash
# 探: 远端能否访问 HF; flux2_vae 仓库的文件清单/大小/格式
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. 网络 ##########"
curl -sS -m 15 -o /dev/null -w "huggingface.co HTTP=%{http_code} time=%{time_total}s\n" \
     https://huggingface.co 2>&1 | tail -2
curl -sS -m 20 "https://huggingface.co/api/models/ai-toolkit/flux2_vae" 2>&1 | head -c 1200
echo

echo
echo "########## 2. 磁盘 ##########"
df -h /root | tail -1

echo
echo "########## 3. HF 环境 ##########"
/opt/conda/envs/cu121/bin/python - <<'PY'
import importlib, os
for m in ("huggingface_hub", "safetensors", "diffusers", "transformers"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:18s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:18s} MISSING ({e!r})")
print("  HF_HOME =", os.environ.get("HF_HOME", "(default ~/.cache/huggingface)"))
print("  HF_ENDPOINT =", os.environ.get("HF_ENDPOINT", "(default)"))
p = os.path.expanduser("~/.cache/huggingface/hub")
print("  hub cache exists:", os.path.isdir(p), (os.listdir(p)[:8] if os.path.isdir(p) else ""))
PY

echo
echo "########## 4. 现有 SD VAE (对照组) ##########"
ls -la data/pretrained/pretrained_models/sd-vae-ft-ema/ 2>/dev/null | head -8
/opt/conda/envs/cu121/bin/python - <<'PY'
import json, os
p = "data/pretrained/pretrained_models/sd-vae-ft-ema/config.json"
if os.path.exists(p):
    c = json.load(open(p))
    for k in ("in_channels", "out_channels", "latent_channels", "block_out_channels",
              "down_block_types", "scaling_factor", "sample_size", "norm_num_groups"):
        if k in c:
            print(f"  {k:22s} = {c[k]}")
PY

echo
echo "########## 5. 代理/镜像变量 ##########"
env | grep -iE 'proxy|hf_|huggingface' | sed 's/=.*KEY.*/=***/' | head -10
