#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_joint_stage1_stage2.py —— 第三阶段: 联合训练

    std 骨架 + 风格 --[stage1 采样, 梯度穿过]--> g_pred --[stage2 flow]--> 图像
                                                                  |
                            L_img (stage2 的 flow loss) 反传 ────────┘
                            (+ λ·L_skel, 可选, **默认关闭**)

端到端 = 生成器只吃「最终图像的 loss」的梯度。GT 骨架**不参与**:
  它既不是最终目标, 实测也已证明它不是好目标 (w7 的 latent 对齐度 0.40 比 w3 的
  0.51 低, 下游反而更好; 且 GT 骨架对 3px png 的 mse 0.094 比模型输出 0.052 还差)。
  拿它做 λ 项 = 拿代理指标拖住生成器, 且会**牵连数据管线**(多加载一套 shard、
  内存/预载变慢、行集被它的存在与否筛选) —— 不允许。
  `--lam-skel > 0` 时才会加载 shards_gt, 其余情况与它完全无关。

与 tools/train_joint_g2img.py 的区别:
  那个脚本的生成器是**旧 bridge 设计**(g 起步 + hide-g), 采样写死; 本脚本的 stage1
  是 **噪声起步 + 骨架当条件**(skel_as_glyph_cond)。两者不能混用。
  模型一律用 src/eval/model_io 严格复刻 train.py 的构造(ckpt 的 args 自带全部架构字段)。

用法 (3a: 冻结 stage2, 只训 stage1):
  python tools/train_joint_stage1_stage2.py \
      --gen-ckpt assets/results/v33_stage1_xs/<run>/checkpoints/0030000.pt \
      --bak-ckpt assets/results/v32_stage2_img/<run>/checkpoints/0080000.pt \
      --train-bak 0 --lam-skel 0 --max-steps 20000 --batch 24
3b (解冻 stage2, 小 10x lr): 加 --train-bak 1 --bak-lr 1e-6
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TIME_SCALE = 1000.0          # 与 src/loss/flow_matching.py 的约定一致


def strip_orig(sd):
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}


def load_idx(d):
    idx = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                idx[int(i)] = (f, j)
    return idx


_SHARD_CACHE = {}


def get_lat(idx, i):
    """取一条 latent。★ 按分片缓存整包 —— savez_compressed 每次 np.load 都要
    整包解压(40MB), 逐条读会把 I/O 变成瓶颈(实测 6.7s/step)。"""
    f, j = idx[i]
    if f not in _SHARD_CACHE:
        with np.load(f) as z:
            _SHARD_CACHE[f] = np.asarray(z["latents"], np.float32)
    return _SHARD_CACHE[f][j]


def gen_sample(gen, g_std, y, steps, dev):
    """条件式采样 (噪声起步 + g 当条件), 带梯度 —— 这是 stage1 唯一被训的路径。"""
    b = g_std.shape[0]
    z = th.randn(b, g_std.shape[1], g_std.shape[2], g_std.shape[3], device=dev)
    ts = th.linspace(1.0, 0.0, steps + 1, device=dev)
    yc = th.zeros_like(y)
    for k in range(steps):
        t = th.full((b,), float(ts[k]) * TIME_SCALE, device=dev)
        v = gen(z, t, y_callig=y, y_char=yc, g=g_std)
        if isinstance(v, tuple):
            v = v[0]
        z = z + (ts[k + 1] - ts[k]) * v
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", required=True, help="stage1 ckpt (噪声起步 + 骨架当条件)")
    ap.add_argument("--resume-gen", default="",
                    help="★ 从联合训练的 ckpt 继续 (取其中 gen_ema 作为起点并重置 EMA)。\n"
                         "  注意: 评测用的是 EMA 权重, 所以续训必须从 gen_ema 接, 不是 gen。")
    ap.add_argument("--bak-ckpt", required=True, help="stage2 ckpt")
    ap.add_argument("--train-bak", type=int, default=0, help="1=也训 stage2 (3b)")
    ap.add_argument("--lam-skel", type=float, default=0.0,
                    help="★ 可选: 骨架辅助损失权重。**默认 0 = 纯端到端**, 生成器只吃\n"
                         "  最终图 loss 的梯度。>0 时才加载 --shards-gt (探针用)。")
    ap.add_argument("--csv", default="assets/train_top10_style23_minusval.csv")
    ap.add_argument("--shards-img", default="data/top10_style23/shards_img")
    ap.add_argument("--shards-std", default="data/top10_style23/shards_std")
    ap.add_argument("--shards-gt", default="data/top10_style23/shards_gtskel_w3",
                    help="骨架级监督的 GT (λ 项用)")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--out-dir", default="assets/results/v34_joint")
    ap.add_argument("--batch", type=int, default=16,
                    help="梯度要穿过 stage1 的采样链, 显存吃紧 -> 远小于单训 stage2 的 384")
    ap.add_argument("--gen-steps", type=int, default=8, help="训练期 stage1 采样步数")
    ap.add_argument("--lr", type=float, default=1e-5, help="stage1 的 lr")
    ap.add_argument("--bak-lr", type=float, default=0.0, help="stage2 的 lr (3b 用, 建议 1e-6)")
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1000,
                    help="每 N 步落一次 predskel shards (训练后统一评下游 ssim)")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--compile-mode", default="",
                    help="torch.compile(gen) 的模式 ('' = 关)。'reduce-overhead' 是本仓库\n"
                         "  在 train.py 里实测选定的省显存+提速档。★ 收益主要在 8 步采样链:\n"
                         "  同一个模块被调 8 次 + 小张量 -> 典型的 kernel-launch 受限。")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    from src.eval import model_io
    from src.loss import create_diffusion_or_flow, flow_kwargs_from
    from src.utils.callig_script_map import map_callig_script

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[1] stage1 (被训)  {a.gen_ckpt}", flush=True)
    gen, ga = model_io.load_model_from_ckpt(a.gen_ckpt, device=dev, use_ema=True)
    assert getattr(ga, "skel_as_glyph_cond", False), "stage1 必须是 skel_as_glyph_cond 模型"
    print(f"[1] stage2 (冻结)  {a.bak_ckpt}", flush=True)
    bak, ba = model_io.load_model_from_ckpt(a.bak_ckpt, device=dev, use_ema=True)
    bak_diff = create_diffusion_or_flow(
        timestep_respacing="", diffusion_type=getattr(ba, "diffusion_type", "flow"),
        **flow_kwargs_from(ba))
    print(f"[2] stage2 flow: {bak_diff}", flush=True)

    # ★ compile 之后, state_dict / load_state_dict / train / eval 一律走**原始模块**,
    #   否则 OptimizedModule 的 `_orig_mod.` 前缀会让 EMA 载入与保存的键名对不上。
    gen_raw = gen
    if a.compile_mode:
        print(f"[1b] torch.compile(gen, mode={a.compile_mode})", flush=True)
        gen = th.compile(gen_raw, mode=a.compile_mode)

    if a.resume_gen:
        _ck = th.load(a.resume_gen, map_location="cpu", weights_only=False)
        _sd = strip_orig(_ck.get("gen_ema") or _ck.get("gen") or {})
        _ms, _us = gen_raw.load_state_dict(_sd, strict=False)
        print(f"[1c] 续训: {a.resume_gen} (step={_ck.get('step')}) "
              f"载入 EMA 权重 missing={len(_ms)} unexpected={len(_us)}", flush=True)

    for p in gen_raw.parameters():
        p.requires_grad_(True)
    for p in bak.parameters():
        p.requires_grad_(bool(a.train_bak))
    gen_p = [p for p in gen_raw.parameters() if p.requires_grad]
    bak_p = [p for p in bak.parameters() if p.requires_grad]
    print(f"[2] 可训 stage1 {sum(p.numel() for p in gen_p):,} | "
          f"stage2 {sum(p.numel() for p in bak_p):,} "
          f"({'解冻' if a.train_bak else '冻结'}) | λ_skel={a.lam_skel}", flush=True)

    groups = [{"params": gen_p, "lr": a.lr}]
    if bak_p:
        groups.append({"params": bak_p, "lr": a.bak_lr or a.lr * 0.1})
    opt = th.optim.AdamW(groups, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (min((s + 1) / max(a.warmup, 1), 1.0) *
                        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(a.max_steps, 1)))))))
    ema = {k: v.detach().clone() for k, v in gen_raw.state_dict().items()}

    # ── 数据 ──
    #   单一数据源: csv + shards_img(目标) + shards_std(条件)。行集 = csv 全部行,
    #   不被任何辅助产物筛选。GT 骨架仅在 --lam-skel>0 (可选探针) 时才碰。
    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    gt_i = load_idx(a.shards_gt) if (a.lam_skel > 0 and a.shards_gt) else {}

    # ★ 复用仓库既有的数据集 (与 train.py / train_skelnet_dit 同机制, 字段一体返回)。
    #   ⚠ 反面教材: 自己手搓"每步 np.load 分片" -> savez_compressed 每次整包解压(40MB),
    #     16 样本 x 3 种 = 48 次/步 -> 实测 6.7 s/step, GPU 全程空等。
    from src.utils.latent_dataset import MCCDLatentDataset
    ds = MCCDLatentDataset(
        csv_file=a.csv, latent_shards_dir=a.shards_img, img_root="",
        image_size=256, is_train=True, preload=True, load_image=False,
        num_preload_workers=16,
        skel_latent_shards_dir=a.shards_std,
        aux_latent_shards_dirs=([a.shards_gt] if a.lam_skel > 0 else None),
        callig_id_map=None, callig_script_map=csmap)
    print(f"[3] 数据集 {len(ds)} 条 | 字段: latent(目标图) + skel_latent(std 条件) + y_callig"
          f"{' + aux(GT骨架, λ探针)' if a.lam_skel > 0 else ''} | 预载完成", flush=True)

    # ── 评测: 落 predskel shards (训练后统一用 run_skel_calibration 评), 顺带打骨架 L1 ──
    EVAL_SETS = [("seen20", "assets/eval_top10_seen_20.csv",
                  "data/top10_style23/shards_std"),
                 ("strict84", "assets/eval_top10_strict_subset84.csv",
                  "data/50k_v2_glyph15k/shards_std")]

    @th.no_grad()
    def eval_dump(step):
        ck = {k: v.detach().clone() for k, v in gen_raw.state_dict().items()}
        gen_raw.load_state_dict({k: v.to(dev) for k, v in ema.items()})
        gen_raw.eval()
        # ★ 把"本次评测实际使用的那份权重"原样存一份 —— 否则事后无从对账
        #   (之前只存训练 ckpt 的 gen/gen_ema, 无法证明与评测时一致)。
        _ep = os.path.join(a.out_dir, f"evalgen_step{step:06d}.pt")
        th.save({"gen_eval": {k: v.detach().cpu()
                              for k, v in gen_raw.state_dict().items()},
                 "step": step, "src": "eval_dump/ema"}, _ep)
        print(f"    [eval] 评测权重快照 -> {_ep}", flush=True)
        for nm, csvp, sdir in EVAL_SETS:
            si, gi = load_idx(sdir), gt_i   # gt_i 仅在 λ>0 时非空
            ee, gg, yy = [], [], []
            for r in csv.DictReader(open(csvp, encoding="utf-8")):
                m = re.search(r"(\d+)\.png$", r.get("image_path", ""))
                if not m:
                    continue
                i = int(m.group(1))
                if i not in si or (gi and i not in gi):
                    continue
                ee.append(i)
                gg.append(get_lat(si, i))
                yy.append(int(map_callig_script(int(r["calligrapher_id"]),
                                                int(r["script_id"]), csmap)))
            outs = []
            for s in range(0, len(ee), 16):
                g = th.from_numpy(np.stack(gg[s:s + 16])).to(dev)
                y = th.tensor(yy[s:s + 16], dtype=th.long, device=dev)
                # ★ 用 gen_raw (与权重/eval() 同模块), 不走编译路径 —— 原来传给的是
                #   编译后的 gen, 与 eval()/权重所在模块不是同一个实例。
                z = gen_sample(gen_raw, g, y, 50, dev)
                outs.append(z.float().cpu().numpy())
            o = np.concatenate(outs).astype(np.float16)
            # ★ 无条件解码校验: 生成骨架必须有墨。λ=0 时原来整段 L1 检查被跳过,
            #   白图能一路跑到训练结束而无人发现 (2026-10-01 踩过)。
            _ink = float("nan")
            # ★ 懒加载 VAE (训练器原本没有; 用仓库的 _get_vae), 且只解码前 8 张 ——
            #   84 张一次性 decode 的激活 ~8GiB, 会把训练挤 OOM (in_mem_eval 踩过)。
            global _EVAL_VAE
            if "_EVAL_VAE" not in globals():
                from src.eval.in_mem_eval import _get_vae
                _EVAL_VAE = _get_vae(
                    dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
            with th.no_grad():
                _k = min(8, o.shape[0])
                _dec = _EVAL_VAE.decode(
                    th.from_numpy(o[:_k].astype(np.float32)).to(dev) / 0.18215).sample
            _ink = float((_dec.float().mean(1) < 0).float().mean())
            del _dec
            th.cuda.empty_cache()
            if _ink <= 0.001:
                print(f"    [eval] ✗✗ 生成骨架无墨(ink={_ink:.4f}) —— "
                      f"该步输出是白图, ckpt 不要用!", flush=True)
            d = os.path.join(a.out_dir, f"predskel_step{step:06d}", nm)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "shard_00000.npz"),
                                latents=o, img_ids=np.array(ee, dtype=np.int64))
            # 落盘 predskel shards 才是关键 (下游 run_skel_calibration 判分);
            # 骨架 L1 只是 λ 探针的顺手指示, 默认关闭时不算。
            msg = (f"    [eval] step {step} {nm} n={len(ee)} "
                   f"gen_ink={_ink:.4f}")
            if gi:
                l1 = float(np.mean([np.abs(o[t].astype(np.float32)
                                           - get_lat(gi, ee[t])) for t in range(len(ee))]))
                msg += f" 骨架L1={l1:.4f}"
            print(msg + f" -> {d}", flush=True)
        gen_raw.load_state_dict(ck)
        gen_raw.train()

    t0, run_img, run_skel, run_gn, cnt = time.time(), 0.0, 0.0, 0.0, 0
    for step in range(1, a.max_steps + 1):
        k = np.random.randint(0, len(ds), a.batch)
        bs = [ds[int(j)] for j in k]
        x0 = th.stack([b["latent"].float() for b in bs]).to(dev)
        g_std = th.stack([b["skel_latent"].float() for b in bs]).to(dev)
        y = th.tensor([int(b["y_callig"]) for b in bs], dtype=th.long, device=dev)
        # GT 骨架只在 λ 探针时取, 默认路径完全不碰
        g_gt = (th.stack([b["aux_latents"].float() for b in bs]).to(dev)
                if a.lam_skel > 0 else None)

        # ★ bf16 采样: 8 步采样的前向+回传是算力大头, 原来跑在 fp32 (没裹 autocast)。
        with th.autocast("cuda", dtype=th.bfloat16):
            g_pred = gen_sample(gen, g_std, y, a.gen_steps, dev)
        t = bak_diff.sample_t(x0.shape[0], dev)
        with th.autocast("cuda", dtype=th.bfloat16):
            ld = bak_diff.training_losses(
                bak, x0, t,
                {"y_callig": y, "y_char": th.zeros_like(y), "g": g_pred})
            l_img = ld["loss"].mean()
        # ★ 端到端: 唯一的监督信号就是 L_img (真迹图像的 flow loss) 反传到 stage1。
        loss = l_img
        l_skel = th.zeros((), device=dev)
        if a.lam_skel > 0 and g_gt is not None:
            l_skel = th.nn.functional.mse_loss(g_pred.float(), g_gt)
            loss = loss + a.lam_skel * l_skel

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # ★ 端到端正确性的守卫: stage1 的梯度范数。compile 若把采样链的图切断,
        #   L_img 照样正常下降, 但生成器一个梯度都收不到 —— 只有这个数是证据。
        gn_gen = float(th.nn.utils.clip_grad_norm_(gen_p, 1.0)) if gen_p else 0.0
        if bak_p:
            th.nn.utils.clip_grad_norm_(bak_p, 1.0)
        run_gn += gn_gen
        opt.step()
        sched.step()
        with th.no_grad():
            for kk, v in gen_raw.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[kk].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                else:
                    ema[kk].copy_(v)
        run_img += float(l_img)
        run_skel += float(l_skel)
        cnt += 1
        if step % a.log_every == 0:
            _sk = (f"L_skel {run_skel / cnt:.4f}" if a.lam_skel > 0
                   else "端到端(仅 L_img)")
            print(f"    step {step:6d}  L_img {run_img / cnt:.4f}  {_sk}  "
                  f"|grad_gen| {run_gn / cnt:.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{time.time() - t0:.0f}s  "
                  f"peak {th.cuda.max_memory_allocated() / 2**30:.2f}G",
                  flush=True)
            run_img = run_skel = run_gn = 0.0
            cnt = 0
        if a.eval_every > 0 and step % a.eval_every == 0:
            eval_dump(step)
        if a.ckpt_every > 0 and step % a.ckpt_every == 0:
            p = os.path.join(a.out_dir, f"joint_{step:06d}.pt")
            th.save({"gen": strip_orig(gen_raw.state_dict()), "gen_ema": strip_orig(ema),
                     "bak": strip_orig(bak.state_dict()),
                     "step": step, "args": vars(a)}, p)
            print(f"    [ckpt] {p}", flush=True)
    print(f"[4] DONE -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
