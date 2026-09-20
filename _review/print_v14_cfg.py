# -*- coding: utf-8 -*-
"""打印 v14_style87 两个 stage 配置的关键字段 (远程跑)。"""
import json

for name in ("v14_style87_stage2", "v14_style87_stage3"):
    d = json.load(open(f"src/train/configs/{name}.json", encoding="utf-8"))
    print("=" * 30, name)
    for k in ("experiment_name", "results_dir", "model", "max_steps",
              "ckpt_every", "epoch_steps", "global_batch_size", "weight_decay",
              "glyph_drop_prob", "cond_drop_all_prob", "cond_drop_one_prob",
              "glyph_inject_mode", "glyph_inject_layers", "style_token_n",
              "callig_style_attn", "callig_n_style", "freeze_callig_table",
              "num_calligraphers", "data_csv", "skel_as_glyph_cond",
              "glyph_vec_cond", "condition_fusion", "w_repa",
              "in_mem_eval_sets", "resume_lr", "lr", "callig_emb_pretrained",
              "callig_id_map"):
        print(f"  {k} = {d.get(k, '<ABSENT>')}")
