#!/bin/bash
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
$PY - <<'EOF'
import json, re
short = {
 "src/train/configs/v31_stage1_skel.json":
   "v31-stage1-skel: stage1 = std skel + style -> GT skel(w3). 架构照搬 v25_stdskel"
   " (DiT-2Cond-S/2 + adaln4), 唯一改动 latent_shards_dir = shards_gtskel_w3."
   " batch384 / 80k / early_stop. 详见 docs/experiments/2026-09-30-stage1-skel.md",
 "src/train/configs/v32_stage2_img.json":
   "v32-stage2-img: stage2 = GT skel + style -> img, 从 v26@30k 续训, 新增整块抹白"
   " 增强 glyph_mask_*. batch384 / 80k / early_stop."
   " 详见 docs/experiments/2026-09-30-stage1-skel.md",
}
for p, txt in short.items():
    s = open(p, encoding="utf-8").read()
    # 把整个 _comment 值(可能含真实换行)替换成单行短句
    s2 = re.sub(r'"_comment"\s*:\s*".*?",\s*\n',
                '"_comment": "%s",\n' % txt, s, count=1, flags=re.S)
    if s2 != s:
        open(p, "w", encoding="utf-8").write(s2)
    try:
        c = json.load(open(p, encoding="utf-8"))
        print("JSON OK ", p, "| batch =", c["global_batch_size"],
              "| max_steps =", c["max_steps"], "| early =", c["early_stop"])
    except Exception as e:
        print("JSON FAIL", p, e)
        raise SystemExit(1)
EOF
