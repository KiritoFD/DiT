#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/eval_union_ckpt.py — 独立评测 **v30-union 的某个 ckpt**（不训练）。

为什么要独立脚本: `tools/train_joint_g2img.py` 没有 --eval-only，而它的 run_dir 是
带时间戳的（`<results-dir>/<ts>-<name>`），resume 又只在**新建的** run_dir 里找 ckpt，
所以没法直接 resume 到旧 ckpt 再触发 eval。这里把 main() 里的 `make_eval_args` /
`regen_pred_shards` / `run_eval` 三段原样复刻出来。

用法:
  python tools/eval_union_ckpt.py \
      --ckpt assets/results/v30_union_LEAKED_20260930/20260930-182328-v30-union/checkpoints/0002500.pt \
      --out-dir assets/results/v30_union_test --tag TEST
"""
import os
import re
import sys
import csv
import glob
import json
import copy
import time
import argparse
import numpy as np
import torch as th
from types import SimpleNamespace

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '/root/Workspace/xy/DiT')

from tools.train_joint_g2img import build_models, gen_sample, _strip  # noqa: E402

ROOT = '/root/Workspace/xy/DiT'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='',
                    help='v30-union ckpt (含 gen_ema/bak_ema)。留空则用 --gen-ckpt/--bak-ckpt 两阶段权重。')
    ap.add_argument('--out-dir', default='assets/results/v30_union_test')
    ap.add_argument('--tag', default='TEST')
    ap.add_argument('--bak-ckpt',
                    default='assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt')
    ap.add_argument('--bak-config',
                    default='assets/results/v26_gtskel/20260929-103927-v26-gtskel/resolved_config.json')
    ap.add_argument('--gen-ckpt', default='assets/skelnet_dit_H_bridge_nog_w7.pt.best')
    ap.add_argument('--strict-csv', default='assets/eval_v13_strict_fixed.csv')
    ap.add_argument('--seen-csv', default='assets/eval_top10_seen_20.csv')
    ap.add_argument('--strict-shards-gt', default='data/top10_style23/gt_skel_eval_strict84')
    ap.add_argument('--pred-shards-out', default='data/top10_style23/predskel_eval_strict84_TEST')
    ap.add_argument('--std84-cache', default='data/top10_style23/std84_latents.npz')
    ap.add_argument('--vae', default='data/pretrained/pretrained_models/sd-vae-ft-ema')
    ap.add_argument('--style-emb', default='assets/callig_script_emb_top10.pt')
    ap.add_argument('--callig-map', default='assets/callig_script_id_map_top10.json')
    ap.add_argument('--gen-steps-eval', type=int, default=25)
    ap.add_argument('--std-shards', default='data/50k_v2_glyph15k/shards_std',
                    help='★ 标准骨架 latent 的正确来源。regen_pred_shards 原本是**重新编码 PNG**'
                         '(v=1-2*img, 墨=+1), 与训练分布(arr/127.5-1, 墨=-1)极性相反 -> 生成器收到'
                         '域外输入 -> pred 全垃圾。这里改成直接查 shards_std。')
    ap.add_argument('--beta', type=float, default=0.634,
                    help='★ 幅度校准 (gen_predskel_dit.py 同口径): z=(1-β)*g_std+β*z_gen')
    ap.add_argument('--renorm', action='store_true',
                    help='★ 去曝光: decode -> 二值化 -> skeletonize -> 3px 膨胀 -> 重编码, '
                         '把条件拉回 v26 训练分布(3px GT 骨架)。与 gen_predskel_dit.py --renorm 同口径。')
    ap.add_argument('--fix-std', action='store_true', default=True,
                    help='用 shards_std 替换 std84 缓存 (默认开)')
    ap.add_argument('--no-pred', action='store_true', help='跳过生成器重生成(只用现成 pred shards)')
    ap.add_argument('--std-as-pred', action='store_true',
                    help='★ 把**标准骨架**本身当作 pred 写出 -> 得到「跳过 SkelNet」的基线')
    ap.add_argument('--pred-shards-std', default='data/top10_style23/predskel_eval_strict84_STD')
    a = ap.parse_args()

    dev = th.device('cuda')
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.pred_shards_out, exist_ok=True)

    print(f'[1] 建模型 (shell 用原始 ckpt, 随后被 union ckpt 覆盖)', flush=True)
    a.gen_ckpt = a.gen_ckpt
    a.bak_ckpt = a.bak_ckpt
    a.bak_config = a.bak_config
    gen, bak = build_models(a, dev)
    gen_eval = copy.deepcopy(gen).eval()
    bak_eval = copy.deepcopy(bak).eval()
    for p in list(gen_eval.parameters()) + list(bak_eval.parameters()):
        p.requires_grad_(False)

    if not a.ckpt:
        print('[2] 两阶段模式: 直接用 --gen-ckpt / --bak-ckpt (build_models 已载入)', flush=True)
        gen_ema = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        bak_ema = {k: v.detach().clone() for k, v in bak.state_dict().items()}
        step = 0
        gen.load_state_dict(gen_ema); bak.load_state_dict(bak_ema)
        return _run(a, dev, gen, bak, gen_eval, bak_eval, gen_ema, bak_ema, step)
    print(f'[2] 载入 union ckpt: {a.ckpt}', flush=True)
    d = th.load(a.ckpt, map_location='cpu', weights_only=False)
    print(f'    键: {list(d.keys())}  step={d.get("step")}', flush=True)
    gen_ema = _strip(d['gen_ema'])
    bak_ema = _strip(d['bak_ema'])
    step = int(d.get('step', 0))
    m1, u1 = gen.load_state_dict(gen_ema, strict=False)
    m2, u2 = bak.load_state_dict(bak_ema, strict=False)
    print(f'    gen miss={len(m1)} unexp={len(u1)} | bak miss={len(m2)} unexp={len(u2)}',
          flush=True)

    return _run(a, dev, gen, bak, gen_eval, bak_eval, gen_ema, bak_ema, step)


def _run(a, dev, gen, bak, gen_eval, bak_eval, gen_ema, bak_ema, step):
    from src.eval.in_mem_eval import run_in_mem_eval, _get_vae
    rcfg = json.load(open(a.bak_config, encoding='utf-8'))

    def make_eval_args():
        ev = SimpleNamespace(**{k: v for k, v in rcfg.items() if not k.startswith('_')})
        ev.eval_csv = os.path.join(os.path.dirname(a.strict_csv),
                                   'eval_v13_strict84_aligned.csv')
        ev.eval_skel_latent_shards_dir = a.strict_shards_gt
        ev.eval_skel_latent_shards_dir_pred = (a.pred_shards_std if a.std_as_pred
                                               else a.pred_shards_out)
        ev.eval_blend_alpha = 0.0
        ev.eval_cfg = 1.0
        ev.eval_steps = 50
        ev.eval_self_cond = False
        ev.img_root = None
        return ev

    if not a.no_pred:
        print('[3] 重生成 pred skel shards (生成器 EMA)', flush=True)
        gen_eval.load_state_dict(gen_ema)
        gen_eval.eval()
        vae = _get_vae(dev, a.vae)
        rows = list(csv.DictReader(open(a.strict_csv, encoding='utf-8')))
        gt_ids = set()
        for sp in glob.glob(os.path.join(a.strict_shards_gt, 'shard_*.npz')):
            with np.load(sp) as z:
                gt_ids |= {int(i) for i in z['img_ids']}
        sel = []
        for r in rows:
            m = re.search(r'(\d+)\.png$', r.get('image_path', ''))
            if m and int(m.group(1)) in gt_ids:
                sel.append(r)
        ids84 = [int(re.search(r'(\d+)\.png$', r['image_path']).group(1)) for r in sel]
        print(f'    strict84: {len(sel)}/{len(rows)} 行与 GT shards 对齐', flush=True)
        align_csv = os.path.join(os.path.dirname(a.strict_csv),
                                 'eval_v13_strict84_aligned.csv')
        with open(align_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(sel[0].keys()))
            w.writeheader()
            w.writerows(sel)
        # ★ 直接从 shards_std 查 (正确极性); 不再用 std84_latents.npz (那是反的)
        _sm = {}
        for _f in glob.glob(os.path.join(a.std_shards, 'shard_*.npz')):
            with np.load(_f) as _z:
                for _j, _i in enumerate(_z['img_ids']):
                    if int(_i) in set(ids84):
                        _sm[int(_i)] = _z['latents'][_j].astype(np.float32)
        _miss = [i for i in ids84 if i not in _sm]
        print(f'    std shards 命中 {len(_sm)}/{len(ids84)} (缺 {len(_miss)})', flush=True)
        if _miss:
            # ★ 缺失的用 PNG 现编码 —— 但必须用**正确极性** v = 2*img-1 (白=+1, 墨=-1),
            #   与 src/data/vae_io.py:_tf_gray 的 arr/127.5-1 一致。
            #   原 regen_pred_shards 写的是 v = 1-2*img -> 极性相反 -> 生成器域外输入 -> 垃圾。
            from PIL import Image
            _byid = {int(re.search(r'(\d+)\.png$', r['image_path']).group(1)): r for r in sel}
            for i in _miss:
                _p = _byid[i]['std_path']
                _p = _p if os.path.isabs(_p) else os.path.join(ROOT, _p)
                _im = np.asarray(Image.open(_p).convert('L'), np.float32) / 255.0
                _v = 2.0 * _im - 1.0                     # ★ 正确极性
                _x = th.from_numpy(_v)[None, None].repeat(1, 3, 1, 1).to(dev)
                _sm[i] = (vae.encode(_x).latent_dist.mode() * 0.18215).float()[0].cpu().numpy()
            print(f'    补编码 {len(_miss)} 条 (v=2*img-1, 正确极性)', flush=True)
        lat84 = th.stack([th.from_numpy(_sm[i]) for i in ids84])
        _zs = np.load(a.std84_cache) if os.path.exists(a.std84_cache) else None
        if _zs is not None:
            print(f'    [对照] std84 缓存 mean={_zs["latents"].astype(np.float32).mean():+.4f} '
                  f'vs shards_std mean={lat84.numpy().mean():+.4f}', flush=True)
        lat84 = lat84.to(dev)
        from src.utils.callig_script_map import map_callig_script
        csmap = json.load(open(a.callig_map, encoding='utf-8'))
        outs = th.zeros_like(lat84)
        for s in range(0, len(sel), 16):
            rs = sel[s:s + 16]
            y = th.tensor([int(map_callig_script(int(r['calligrapher_id']),
                                                 int(r['script_id']), csmap))
                           for r in rs], device=dev)
            with th.no_grad():
                _g = lat84[s:s + 16]
                _z = gen_sample(gen_eval, _g, y, a.gen_steps_eval)
                # ★ 幅度校准 (与 gen_predskel_dit.py --beta 同口径)
                _z = (1 - a.beta) * _g + a.beta * _z
                if a.renorm:
                    # ★ 去曝光: 拉回 v26 训练分布 (3px GT 骨架配方), 与 gen_predskel_dit 同口径
                    from skimage.morphology import skeletonize as _sk
                    from scipy.ndimage import binary_dilation as _bd
                    _dec = vae.decode(_z / 0.18215).sample.mean(1)
                    _ink = (_dec < 0).cpu().numpy()
                    _proc = np.empty_like(_ink)
                    for _i2 in range(_ink.shape[0]):
                        _proc[_i2] = _bd(_sk(_ink[_i2]),
                                         structure=np.ones((3, 3), bool), iterations=1)
                    _im = th.from_numpy((1.0 - _proc.astype(np.float32)) * 2 - 1)
                    _im = _im.unsqueeze(1).repeat(1, 3, 1, 1).to(dev)
                    with th.no_grad():
                        _z = vae.encode(_im).latent_dist.mode() * 0.18215
                outs[s:s + 16] = _z
        np.savez_compressed(os.path.join(a.pred_shards_out, 'shard_00000.npz'),
                            img_ids=np.array(ids84),
                            latents=outs.cpu().numpy().astype(np.float16))
        if a.std_as_pred:
            import shutil as _sh
            _sh.rmtree(a.pred_shards_std, ignore_errors=True)
            os.makedirs(a.pred_shards_std, exist_ok=True)
            np.savez_compressed(os.path.join(a.pred_shards_std, 'shard_00000.npz'),
                                img_ids=np.array(ids84),
                                latents=lat84.cpu().numpy().astype(np.float16))
            print(f'    -> 另存 std 骨架作为 pred: {a.pred_shards_std} ({len(ids84)} 条)', flush=True)
        print(f'    -> {a.pred_shards_out}/shard_00000.npz ({len(ids84)} 条)', flush=True)

    print('[4] 评测 (v26 主干 + 联合生成器)', flush=True)
    bak_eval.load_state_dict(bak_ema)
    bak_eval.eval()
    ev_args = make_eval_args()
    align_csv = os.path.join(os.path.dirname(a.strict_csv), 'eval_v13_strict84_aligned.csv')
    sets = [('strict84', align_csv, 84),
            ('strict84_pred', align_csv, 84),
            ('seen20', a.seen_csv, 20)]
    res = run_in_mem_eval(bak_eval, ev_args, step, dev, a.out_dir, sets=sets)
    print('[eval] ' + '  '.join(f'{k}={v:.4f}' for k, v in res.items()), flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
