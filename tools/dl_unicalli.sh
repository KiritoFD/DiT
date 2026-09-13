#!/bin/bash
# dl_unicalli.sh - download UniCalli dataset via hf-mirror
cd /root/Workspace/xy/DiT
export HF_ENDPOINT=https://hf-mirror.com
/opt/conda/envs/cu121/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="TSXu/UniCalli_dataset", repo_type="dataset",
                      local_dir="data/unicalli")
print(p)
EOF
echo DONE
