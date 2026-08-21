#!/bin/bash
set -x
cd /root/Workspace/xy/DiT
PY=/opt/conda/bin/python
echo '=== load DiT checkpoint (metadata only) ==='
$PY - <<'EOF'
import torch
ck = torch.load('pretrained_models/DiT-XL-2-256x256.pt', map_location='cpu')
state = ck.get('model', ck) if isinstance(ck, dict) else ck
if isinstance(state, dict):
    keys = list(state.keys())
    print('checkpoint dict keys:', keys[:5] if keys else 'N/A')
    print('num tensors:', len(keys))
    # get a sample shape
    for k in keys[:3]:
        print(f'  {k}: {tuple(state[k].shape)} dl_toch')
else:
    print('checkpoint is:', type(ck))
print('torch', torch.__version__)
EOF
echo '=== dataset canny count ==='
ls /root/Workspace/xy/DiT/dataset/canny/ | wc -l
echo '=== sample canny subdir ==='
ls /root/Workspace/xy/DiT/dataset/canny/ | head -5
echo '=== dataset root ==='
ls -la /root/Workspace/xy/DiT/dataset/