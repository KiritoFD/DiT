#!/bin/bash
# dl_unicalli.sh - download UniCalli dataset via hf-mirror (token via env HF_TOKEN)
cd /root/Workspace/xy/DiT
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN}"
/opt/conda/envs/cu121/bin/python - <<'EOF'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="TSXu/UniCalli_dataset", repo_type="dataset",
                      local_dir="data/unicalli",
                      token=os.environ.get("HF_TOKEN"))
print("OK:", p)
EOF
echo DONE
