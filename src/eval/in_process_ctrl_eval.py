# -*- coding: utf-8 -*-
"""
in_process_ctrl_eval — 薄壳: ControlNet 训练内的 in-process GPU eval.

实际实现 (采样/decode/落盘/指标) 在 src.eval.inference; 这里保留旧函数签名
(prepare_ctrl_eval_cache / run_ctrl_pair_eval) 供 train_controlnet.py 调用。
"""
from src.eval.inference import (
    make_eval_cache as prepare_ctrl_eval_cache,
    run_pair_eval,
    write_pending_metrics_marker,
)


def run_ctrl_pair_eval(ctrl, vae, diffusion, cache, device, step, checkpoint_dir,
                       ddim_steps=50, cfg_scale=4.0, dit_batch=16, vae_batch=16):
    """Run ctrl (GT skel) + base (no skel) GPU evals and write the pending marker.

    Signature preserved for train_controlnet.py; delegates to inference core.
    """
    from datetime import datetime

    def _log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [gpu-ctrl-eval] {msg}", flush=True)

    n_ctrl, e_ctrl = run_pair_eval(
        ctrl, vae, diffusion, cache, device, step, checkpoint_dir,
        ddim_steps=ddim_steps, cfg_scale=cfg_scale,
        dit_batch=dit_batch, vae_batch=vae_batch, with_skel=True, tag="ctrl")
    n_base, e_base = run_pair_eval(
        ctrl, vae, diffusion, cache, device, step, checkpoint_dir,
        ddim_steps=ddim_steps, cfg_scale=cfg_scale,
        dit_batch=dit_batch, vae_batch=vae_batch, with_skel=False, tag="base")
    marker = write_pending_metrics_marker(
        checkpoint_dir, step, n_base, n_ctrl, e_base, e_ctrl,
        ddim_steps, cfg_scale)
    _log(f"ctrl(base) n={n_base}@{e_base:.0f}s, ctrl(skel) n={n_ctrl}@{e_ctrl:.0f}s -> {marker}")
    return n_ctrl, e_ctrl