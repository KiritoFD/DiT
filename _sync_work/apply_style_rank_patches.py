#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""往 train.py / cli.py 打补丁（幂等）。

P1: train.py init 区建 StyleRankLoss（全冻结, added=0）
P3: 训练循环 loss 组装 + pred_xstart 携带图
P2: cli.py 加 --w-style-rank 等 flag
"""
TP = '/root/Workspace/xy/DiT/src/train/train.py'
CP = '/root/Workspace/xy/DiT/src/train/cli.py'

# ---------- P1 ----------
s = open(TP, encoding='utf-8').read()
if 'StyleRankLoss' not in s:
    anchor = "    # ---- 实例骨架结构 loss: 冻结 probe (新增可训练参数 0) ----"
    p1 = (
        "    # ---- 冻结风格排序头 loss (2026-09-25 冒烟): 全冻结, added=0 ----\n"
        "    # loss = 1 - cos(f(pred_xstart), centroid[y_pair]); 只在 t∈[t_min,t_max] 生效。\n"
        "    _style_rank_loss_fn = None\n"
        "    if getattr(args, 'w_style_rank', 0.0) > 0:\n"
        "        import importlib\n"
        "        _srm = importlib.import_module('src.loss.style_rank_module')\n"
        "        _style_rank_loss_fn = _srm.StyleRankLoss(\n"
        "            ckpt=getattr(args, 'style_rank_ckpt', 'assets/style_enc_latent.pt'),\n"
        "            cent_npy=getattr(args, 'style_rank_cent', 'assets/rank_cent87.npy'),\n"
        "            t_min=float(getattr(args, 'style_rank_t_min', 0.05)),\n"
        "            t_max=float(getattr(args, 'style_rank_t_max', 0.25)))\n"
        "        _style_rank_loss_fn.to(device)\n"
        "        logger.info(\"[style-rank] enabled: w=%.4f t=[%.2f,%.2f] cent=%s added=0\"\n"
        "                    % (args.w_style_rank, _style_rank_loss_fn.t_min,\n"
        "                       _style_rank_loss_fn.t_max, tuple(_style_rank_loss_fn.cent.shape)))\n\n"
    )
    assert anchor in s, 'P1 anchor not found'
    s = s.replace(anchor, p1 + anchor, 1)
    open(TP, 'w', encoding='utf-8').write(s)
    print('P1 applied')
else:
    print('P1 already present')

# ---------- P3: loss 组装 + pred_xstart 携带图 ----------
s = open(TP, encoding='utf-8').read()
if 'loss_style_rank' not in s:
    anchor3 = (
        "                loss = (loss_diff\n"
        "                        + loss_repa  # 统一 REPA: w × (1 - cos) 已在 RepaModule.forward 内含 warmup\n"
        "                        + getattr(args, 'w_std_mid', 0.0) * loss_std_mid\n"
        "                        + getattr(args, 'w_latent_skel', 0.0) * loss_skel_struct)")
    p3 = anchor3 + "\n\n" + (
        "                # ---- style-rank loss (t 门控在模块内) ----\n"
        "                loss_style_rank = torch.tensor(0.0, device=device)\n"
        "                if _style_rank_loss_fn is not None and getattr(args, 'w_style_rank', 0.0) > 0:\n"
        "                    if pred_xstart_latent is not None:\n"
        "                        _yp = batch['y_pair'].to(device, non_blocking=True)\n"
        "                        loss_style_rank, _n_style_rank = _style_rank_loss_fn(\n"
        "                            pred_xstart_latent, _yp, t.to(device))\n"
        "                        loss = loss + getattr(args, 'w_style_rank', 0.0) * loss_style_rank\n")
    assert anchor3 in s, 'P3 anchor not found'
    s = s.replace(anchor3, p3, 1)
open(TP, 'w', encoding='utf-8').write(s)
print('P3 done')

# P3b: 让 pred_xstart 携带梯度图
s = open(TP, encoding='utf-8').read()
old = ("                _need_x0_grad = (getattr(args, 'w_std_mid', 0.0) > 0\n"
       "                                 or getattr(args, 'w_latent_skel', 0.0) > 0)")
new = ("                _need_x0_grad = (getattr(args, 'w_std_mid', 0.0) > 0\n"
       "                                 or getattr(args, 'w_latent_skel', 0.0) > 0\n"
       "                                 or getattr(args, 'w_style_rank', 0.0) > 0)")
if old in s and "w_style_rank', 0.0) > 0\n" not in s.split('_need_x0_grad')[1][:200]:
    s = s.replace(old, new, 1)
old2 = ("                _need_x0 = (getattr(args, 'w_std_mid', 0.0) > 0\n"
        "                            or getattr(args, 'w_latent_skel', 0.0) > 0)")
new2 = ("                _need_x0 = (getattr(args, 'w_std_mid', 0.0) > 0\n"
        "                            or getattr(args, 'w_latent_skel', 0.0) > 0\n"
        "                            or getattr(args, 'w_style_rank', 0.0) > 0)")
if old2 in s:
    s = s.replace(old2, new2, 1)
open(TP, 'w', encoding='utf-8').write(s)
print('P3b done')

# ---------- P2: cli.py ----------
c = open(CP, encoding='utf-8').read()
if 'w-style-rank' not in c:
    idx = c.index('    parser.add_argument("--w-latent-skel"')
    p2 = (
        "    parser.add_argument(\"--w-style-rank\", type=float, default=0.0, dest=\"w_style_rank\",\n"
        "                        help=\"frozen style-ranker loss (1-cos(f(pred_x0), pair centroid)), \"\n"
        "                             \"active only inside t-gate. smoke: 0.005. 0=off.\")\n"
        "    parser.add_argument(\"--style-rank-ckpt\", type=str, default=\"assets/style_enc_latent.pt\",\n"
        "                        dest=\"style_rank_ckpt\")\n"
        "    parser.add_argument(\"--style-rank-cent\", type=str, default=\"assets/rank_cent87.npy\",\n"
        "                        dest=\"style_rank_cent\")\n"
        "    parser.add_argument(\"--style-rank-t-min\", type=float, default=0.05, dest=\"style_rank_t_min\")\n"
        "    parser.add_argument(\"--style-rank-t-max\", type=float, default=0.25, dest=\"style_rank_t_max\")\n"
    )
    c = c[:idx] + p2 + c[idx:]
    open(CP, 'w', encoding='utf-8').write(c)
    print('P2 applied')
else:
    print('P2 already present')

print('ALL DONE')