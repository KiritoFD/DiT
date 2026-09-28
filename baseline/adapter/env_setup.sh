#!/bin/bash
# Install baseline deps into the cloned `baseline` conda env (torch 2.5.1+cu121).
# Idempotent. Run: bash /root/Workspace/xy/DiT/baseline/adapter/env_setup.sh
set -e
PIP=/opt/conda/envs/baseline/bin/pip
MIRROR=${MIRROR:-}   # set MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple" if pypi is slow

$PIP install $MIRROR "diffusers==0.22.0" "transformers==4.33.1" "accelerate==0.23.0" \
    info-nce-pytorch pygame

$PIP install $MIRROR lmdb "sconf>=0.2.5" kornia lpips pytorch-fid \
    opencv-python-headless scikit-image scikit-learn tensorboardX tensorboard \
    pandas scipy tqdm pillow pyyaml fonttools

$PIP install $MIRROR albumentations==1.1.0 easydict --no-deps || true

/opt/conda/envs/baseline/bin/python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
for m in ["diffusers", "transformers", "accelerate", "lmdb", "sconf", "kornia",
          "lpips", "cv2", "skimage", "PIL", "fontTools", "tensorboardX"]:
    __import__(m)
    print("ok", m)
EOF
echo ENV_SETUP_DONE
