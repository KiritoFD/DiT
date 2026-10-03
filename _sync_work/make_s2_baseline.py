#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 v17_s2 阶梯的「阶梯 0 基线」配置。

现有 7 个 v17_s2_s2*.json **全部**启用了 hier_style=1（S2 三层表），
缺一个**纯基线**（hier_style=0、无任何新通路），用于验证"历史失效可复现"。

基线 = v15c 的口径（hier_style=0, script_film=0, pair_residual=0,
                     spatial_film=0, local_ca=0），其余超参完全照抄 s2a。
"""
import json
import os
import copy

SRC = "/root/Workspace/xy/DiT/src/train/configs/v17_s2_s2a_scriptfilm.json"
DST = "/root/Workspace/xy/DiT/src/train/configs/v17_s2_s2z_baseline.json"

c = json.load(open(SRC, encoding="utf-8"))
c = copy.deepcopy(c)

# —— 关掉所有新通路 + 关掉 S2 三层表 ——
c["experiment_name"] = "v17-s2-s2z-baseline"
c["results_dir"] = "assets/results/v17_s2_s2z_baseline"
c["hier_style"] = 0
c["pair_residual"] = 0
c["script_film"] = False
c["script_embed_dim"] = None
c["num_scripts"] = 0
c["num_pairs"] = 0
c["pair_init"] = "zero"
c.pop("spatial_film_rank", None)
c.pop("local_ca_at", None)
c.pop("local_ca_q", None)
c.pop("local_ca_rank", None)
c.pop("local_ca_heads", None)
c.pop("local_ca_layers", None)

json.dump(c, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("wrote:", DST)
for k in ["experiment_name", "results_dir", "hier_style", "pair_residual",
          "script_film", "num_pairs", "num_scripts", "lr", "max_steps",
          "freeze_callig_table", "callig_emb_pretrained", "data_csv",
          "global_batch_size", "cond_drop_all_prob", "cond_drop_one_prob",
          "in_mem_eval", "in_mem_eval_sets"]:
    print("   %-24s = %s" % (k, c.get(k, "<none>")))
