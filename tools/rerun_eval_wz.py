# -*- coding: utf-8 -*-
"""rerun_eval_wz.py — 用修复后的 decode 回填已有 ckpt 的 in-mem eval.

背景 (2026-09-14 事故)
----------------------
`aux_zero_white` 从未在 train.py 的 argparse 注册, config 里的 `true` 被
**静默丢弃**(main_from_cli 只把 config 键套到已注册 action 上), 于是所有
decode 路径都没把白底加回。白底归零把空白区 latent 变成 0, 而 SD VAE 把
零向量解成灰黄棕 (实测 RGB≈[129,110,89]) -> 生成图整体发黄/发黑, poster
看起来"全黑"。训练本身完全正常(目标用的是正确的 wz latent)。

修复后(见 src/eval/inference.py::maybe_add_white), 用本脚本回填历史 step:
  * 覆盖 eval_samples_ctrl/step{step}/{set}/{g,gt}{i}.png
  * 追加/覆盖 eval_stdskel_{summary,batch}.csv 的同 step 行
  * 全量重画 posters/{set}_poster.png + {set}_struct.png
  * `{set}_input_g/` 是幂等的(g 条件 latent 未变), 如需重生成请先删目录

用法
----
    python tools/rerun_eval_wz.py \
        --results-dir assets/results/v11_pretrain_Sp2_base_wz \
        --config src/train/configs/v11_pretrain_Sp2_base_wz.json

    # 只回填部分 step
    python tools/rerun_eval_wz.py ... --steps 1000,2000,3000
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def build_model(a, ck, device):
    """按 ckpt 里存的训练参数重建主模型 (与 cpu_eval_worker._run_pretrain_g 同构)。"""
    from src.model import DiT_2Cond_models
    arch = dict(norm_type=a.get("norm_type", "rms"), mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)), rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)),
                attn_impl=a.get("attn_impl", "sdpa"))
    _aux = [x for x in (a.get("aux_latent_shards_dirs") or "").split(",") if x.strip()]
    in_ch = int(a.get("in_channels") or (4 + 4 * len(_aux)))
    use_g = bool(a.get("w_glyph_cond", False) or a.get("skel_as_glyph_cond", False))
    model = DiT_2Cond_models[a.get("model") or "DiT-2Cond-S/2"](
        in_channels=in_ch,
        num_calligraphers=int(a.get("num_calligraphers") or 1013),
        num_characters=int(a.get("num_characters") or 35130),
        condition_fusion=a.get("condition_fusion") or "factorized_add",
        callig_embed_dim=int(a.get("callig_embed_dim") or 128),
        char_embed_dim=int(a.get("char_embed_dim") or 384),
        char_proj_mode=a.get("char_proj_mode") or "mlp",
        freeze_char_table=bool(a.get("freeze_char_table", True)),
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25, cond_drop_which_glyph_prob=0.5,
        use_checkpoint=False, learn_sigma=False,
        use_glyph_cond=use_g,
        use_char_cond=not bool(a.get("no_char_cond", False)),
        use_std_dino_char_embedder=bool(a.get("use_std_dino_char_embedder", False)),
        std_dino_table_path=a.get("std_dino_table_path"),
        glyph_scale_init=float(a.get("glyph_scale_init", 0.4)),
        glyph_drop_prob=float(a.get("glyph_drop_prob", 0.0)),
        glyph_embedder_depth=int(a.get("glyph_embedder_depth", 0)),
        glyph_inject_layers=int(a.get("glyph_inject_layers", 0)),
        callig_style_attn=bool(a.get("callig_style_attn", False)),
        callig_n_style=int(a.get("callig_n_style", 8)),
        style_token_n=int(a.get("style_token_n", 0)),
        style_role_init=float(a.get("style_role_init", 0.02)),
        glyph_inject_mode=a.get("glyph_inject_mode", "adaln"), **arch)
    if a.get("freeze_callig_table"):
        model.y_callig_embedder.freeze_table()
    sd = _strip(ck.get("ema") or ck.get("model") or ck)
    # ★ strict=True: strict=False 会让形状不匹配的层静默用随机权重
    model.load_state_dict(sd, strict=True)
    assert not unexp, f"unexpected keys={sorted(unexp)[:8]}"
    return model.to(device).eval(), in_ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="实验根目录 (含 posters/ 与 eval_samples_ctrl/)")
    ap.add_argument("--config", required=True,
                    help="训练 config (取 aux_zero_white / in_mem_eval_sets 等)")
    ap.add_argument("--steps", default="", help="逗号分隔; 空=全部 ckpt (升序)")
    ap.add_argument("--run-dir", default="", help="ckpt 所在 run 目录; 空=自动找最新的")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    run_dir = args.run_dir
    if not run_dir:
        cands = sorted(glob.glob(os.path.join(args.results_dir, "*")),
                       key=os.path.getmtime)
        # 只认**真的含 ckpt** 的目录 (中断/取消的启动会留下空 checkpoints/)
        cands = [c for c in cands
                 if glob.glob(os.path.join(c, "checkpoints", "[0-9]*.pt"))]
        assert cands, f"no run dir with ckpt under {args.results_dir}"
        run_dir = cands[-1]
    ck_dir = os.path.join(run_dir, "checkpoints")
    ckpts = sorted(glob.glob(os.path.join(ck_dir, "[0-9]*.pt")))
    if args.steps:
        want = {int(s) for s in args.steps.split(",") if s.strip()}
        ckpts = [p for p in ckpts
                 if int(re.findall(r"(\d+)", os.path.basename(p))[-1]) in want]
    assert ckpts, f"no ckpt found in {ck_dir}"
    print(f"[rerun] run={run_dir}\n[rerun] {len(ckpts)} ckpt(s): "
          f"{[os.path.basename(p) for p in ckpts]}", flush=True)

    from src.eval.in_mem_eval import run_in_mem_eval
    dev = torch.device(args.device)

    # args = ckpt 里存的训练参数 ∪ config 里显式给的键
    # (ckpt 的 args 是事故前存的, 没有 aux_zero_white -> 必须由 config 补上)
    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    ra = first.get("args", {}) or {}
    if isinstance(ra, argparse.Namespace):
        ra = vars(ra)
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        ra[k] = v
    ra["aux_zero_white"] = bool(cfg.get("aux_zero_white", ra.get("aux_zero_white", False)))
    ra["results_dir"] = args.results_dir
    print(f"[rerun] aux_zero_white={ra['aux_zero_white']} "
          f"cfg={ra.get('eval_cfg')} steps={ra.get('eval_steps')} "
          f"sets={ra.get('in_mem_eval_sets')}", flush=True)

    model, in_ch = build_model(ra, first, dev)
    print(f"[rerun] model built in_channels={in_ch}", flush=True)
    ns = argparse.Namespace(**ra)

    t_all = time.time()
    for p in ckpts:
        step = int(re.findall(r"(\d+)", os.path.basename(p))[-1])
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = _strip(ck.get("ema") or ck.get("model") or ck)
        # ★ strict=True: strict=False 会让形状不匹配的层静默用随机权重
        model.load_state_dict(sd, strict=True)
        t0 = time.time()
        res = run_in_mem_eval(model, ns, step, dev, args.results_dir)
        print(f"[rerun] step={step} done in {time.time()-t0:.0f}s -> "
              + " | ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
        del ck
    print(f"[rerun] ALL DONE in {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
