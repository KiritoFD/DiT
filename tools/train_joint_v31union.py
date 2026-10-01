# -*- coding: utf-8 -*-
"""train_joint_v36.py — v36 联训 (矩阵结论后的正确配方):

  冻结 stage2 (v32@50k, 矩阵最优相性) + 解冻 stage1 (v33@30k)
  loss = L_img(端到端, 唯一监督) + λ1·L_skel_flow(stage1 原本任务: 噪声起步
         flow matching on GT-w3 骨架, 防主任务退化) + λ2·MSE(g_pred, g_gt)(防塌锚)

  与 v35 (train_joint_stage1_stage2.py) 的差异:
  1. gen_sample 传**真实 y_char (glyph_id)** —— v33 是 train.py 训的带字条件,
     v35 传 zeros 是错误字条件 (matrix 里 v33 直接 eval dice 0.59 正常即证)
  2. 挂 stage1 原本 flow loss (v35 只有可选 MSE 锚, 纯 L_img 实测会把生成器
     推向空白解 —— v35 5000 步 gen_ink 0.0026)
  3. eval 用 JointSkel2Img 统一模型 + in_mem_eval 双口径 (e2e / oracle)
  4. 40k steps, 起点 = 相性最优组合 v33@30k x v32@50k (矩阵 0.5487/frag 1.56)

用法:
  python -u tools/train_joint_v36.py   # 全默认
"""
import os, sys, json, glob, csv, re, time, math, argparse
import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
TIME_SCALE = 1000.0


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default="assets/results/v33_stage1_xs/20261001-113156-v33-stage1-xs/checkpoints/0030000.pt")
    ap.add_argument("--bak-ckpt", default="assets/results/v32_stage2_img/20261001-062933-v32-stage2-img/checkpoints/0050000.pt")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--csv", default="assets/train_top10_style23_minusval_clean84.csv")
    ap.add_argument("--shards-img", default="data/top10_style23/shards_img")
    ap.add_argument("--shards-std", default="data/top10_style23/shards_std")
    ap.add_argument("--shards-gt", default="data/top10_style23/shards_gtskel_w3")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--results-dir", default="assets/results/v31union")
    ap.add_argument("--experiment-name", default="v31union")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gen-lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--gen-steps-train", type=int, default=8)
    ap.add_argument("--gen-steps-eval", type=int, default=25)
    ap.add_argument("--lam-skel-flow", type=float, default=1.0,
                    help="stage1 原本 flow loss 权重 (噪声起步 + GT-w3 目标)")
    ap.add_argument("--lam-skel-mse", type=float, default=0.3,
                    help="g_pred vs GT 骨架 latent 的直接锚 (防空白塌缩)")
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--ema-interval", type=int, default=4)
    # ── 条件「整块抹白」增强 (与 src/train/train.py 同名同义) ──
    ap.add_argument("--glyph-mask-prob", type=float, default=0.0,
                    help="连续区域抹白: 施加的样本比例 (0=关)")
    ap.add_argument("--glyph-mask-size", type=int, default=4,
                    help="方块边长 (latent 格数; 1 格=8px -> 4 格=32x32 px)")
    ap.add_argument("--glyph-mask-jitter", type=int, default=1, help="边长随机 ±")
    ap.add_argument("--glyph-mask-n", type=int, default=3, help="每条样本抹几块")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--strict84-csv", default="assets/eval_v13_strict84_aligned.csv")
    ap.add_argument("--seen20-csv", default="assets/eval_top10_seen_20.csv")
    ap.add_argument("--shards-std-eval", default="data/top10_style23/predskel_std84_e2e")
    ap.add_argument("--overfit", action="store_true",
                    help="overfit 测试: 训练集 = strict84 csv (84 组合), eval 同源")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default="",
                    help="JSON 实验配置 (与 train.py 同风格的 resolved_config): "
                         "里面的键作为默认值, CLI 显式给的参数优先。")
    a = ap.parse_args()
    if a.config and os.path.isfile(a.config):
        _cfg = json.load(open(a.config, encoding="utf-8"))
        _def = {k: ap.get_default(k) for k in vars(a)}
        _applied = 0
        for k, v in _cfg.items():
            if k not in vars(a):
                continue
            if getattr(a, k) == _def.get(k):      # CLI 没显式改 -> 用 config
                setattr(a, k, v)
                _applied += 1
        print(f"[config] {a.config}: 应用 {_applied} 个键 (CLI 显式参数优先)", flush=True)
    return a


def load_idx(d):
    idx = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            for j, iid in enumerate(z["img_ids"]):
                idx[int(iid)] = (sp, j)
    return idx


def get_lat(idx, iid):
    sp, j = idx[int(iid)]
    with np.load(sp) as z:
        return np.array(z["latents"][j], copy=True).astype(np.float32)


def build_models(a, dev):
    from src.eval import model_io
    gen, ga = model_io.load_model_from_ckpt(a.gen_ckpt, device=dev, use_ema=True)
    bak, ba = model_io.load_model_from_ckpt(a.bak_ckpt, device=dev, use_ema=True)
    gen.train(), bak.train()
    for p in bak.parameters():
        p.requires_grad_(False)   # ★ 冻结 stage2
    return gen, bak, ga, ba


def main():
    a = parse_args()
    th.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = th.device("cuda")
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(a.results_dir, f"{ts}-{a.experiment_name}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(run_dir, "resolved_config.json"), "w",
                            encoding="utf-8"), indent=1, default=str)

    from src.utils.callig_script_map import load_callig_script_map
    csmap = load_callig_script_map(a.callig_map)
    gen, bak, ga, ba = build_models(a, dev)
    gen.train(), bak.eval()

    # ---- 数据: img(x0) + std(g 条件) + GT-w3(骨架 loss 目标), y_char=glyph_id ----
    from src.utils.latent_dataset import MCCDLatentDataset
    if a.overfit:
        aux = ['data/top10_style23/gt_skel_eval_strict84']
    else:
        aux = [a.shards_gt] if a.lam_skel_mse > 0 or a.lam_skel_flow > 0 else None
    _train_csv = a.strict84_csv if a.overfit else a.csv
    _shards_img = ('data/top10_style23/shards_img_eval84' if a.overfit else a.shards_img)
    ds = MCCDLatentDataset(
        csv_file=_train_csv, latent_shards_dir=_shards_img, img_root="",
        image_size=256, is_train=True, preload=True, load_image=False,
        skel_latent_shards_dir=(a.shards_std_eval if a.overfit else a.shards_std),
        aux_latent_shards_dirs=aux,
        callig_id_map=None, callig_script_map=csmap)
    print(f"[data] rows={len(ds)} (aux={'GT骨架' if aux else '无'})", flush=True)

    gen_p = [p for p in gen.parameters() if p.requires_grad]
    opt = th.optim.AdamW(gen_p, lr=a.gen_lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1),
                           a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 *
                           (1 + math.cos(math.pi * min(1.0, s / max(a.max_steps, 1))))))
    ema = {k: v.detach().clone() for k, v in gen.state_dict().items()}

    def gen_sample(g_std, y, y_char, steps, dev, grad=False):
        """噪声起步 + (std骨架, style, glyph) 条件; y_char=真实 glyph_id."""
        b = g_std.shape[0]
        z = th.randn_like(g_std)
        tsv = np.linspace(1.0, 0.0, steps + 1)
        ctx = th.enable_grad() if grad else th.no_grad()
        with ctx:
            for k in range(steps):
                t = th.full((b,), float(tsv[k]) * TIME_SCALE, device=dev)
                v = gen(z, t, y_callig=y, y_char=y_char, g=g_std)
                if isinstance(v, tuple):
                    v = v[0]
                z = z + (float(tsv[k + 1]) - float(tsv[k])) * v.float()
        return z

    # ---- eval: JointSkel2Img 统一模型 + in_mem_eval 双口径 ----
    from src.eval.in_mem_eval import run_in_mem_eval, _get_vae
    from src.model.joint_skel2img import JointSkel2Img
    from types import SimpleNamespace
    rcfg = json.load(open(glob.glob(
        "assets/results/v32_stage2_img/20261001-062933*/resolved_config.json")[0], encoding="utf-8"))
    vae = _get_vae(dev, a.vae).float()

    def make_eval_args():
        ev = SimpleNamespace(**{k: v for k, v in rcfg.items() if not k.startswith("_")})
        ev.eval_blend_alpha, ev.eval_cfg, ev.eval_steps = 0.0, 1.0, 50
        ev.eval_self_cond, ev.img_root = False, None
        # e2e 口径: g 条件 = std84 骨架 shards (wrapper 内部 stage1 生成)
        ev.eval_skel_latent_shards_dir = a.shards_std_eval
        ev.eval_skel_latent_shards_dir_pred = ""   # 不用 _pred 口径 (e2e 直接生成)
        return ev

    def eval_ckpt(step, gen_sd):
        gen.load_state_dict(_strip(gen_sd))
        gen.eval()
        wrapper = JointSkel2Img(gen, bak, gen_steps=a.gen_steps_eval).to(dev).eval()
        ev_args = make_eval_args()
        # e2e 口径的 g 条件 = std 骨架 (与训练的 g_std 同分布), 由 wrapper 内部生成 g_pred
        _ev_csv = a.strict84_csv if a.overfit else a.strict84_csv
        res = run_in_mem_eval(wrapper, ev_args, step, dev, run_dir,
                              sets=[("strict84_e2e", _ev_csv, 84),
                                    ("seen20_e2e", a.seen20_csv, 20)])
        # oracle 参照: stage2 直接吃 GT 骨架 (不随训练变, 每次仅 strict84)
        ev_o = make_eval_args()
        ev_o.eval_skel_latent_shards_dir = "data/top10_style23/gt_skel_eval_strict84"
        res_o = run_in_mem_eval(bak, ev_o, step, dev, run_dir,
                                sets=[("strict84_oracle", a.strict84_csv, 84)])
        gen.train()
        _m = {**res, **res_o}
        _m = {k: (v if not isinstance(v, dict) else v.get("ssim_mean", v)) for k, v in _m.items()}
        _m = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in _m.items()}
        print(f"[eval@{step}] " + json.dumps(_m, ensure_ascii=False, default=str), flush=True)
        return res

    @th.no_grad()
    def write_std84_shards():
        """e2e 口径的 g 条件 shards: strict84 的 std 骨架 latent.
        直接从 csv std_path PNG encode (训练 shards_std 不覆盖全部 eval id)."""
        out_f = os.path.join(a.shards_std_eval, "shard_00000.npz")
        if os.path.exists(out_f):
            return
        os.makedirs(a.shards_std_eval, exist_ok=True)
        lats, ids = [], []
        for r in csv.DictReader(open(a.strict84_csv, encoding="utf-8")):
            p = r["std_path"] if os.path.isabs(r["std_path"]) else os.path.join("/root/Workspace/xy/DiT", r["std_path"])
            img = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
            x = th.from_numpy(1.0 - 2.0 * img)[None, None].repeat(1, 3, 1, 1).to(dev)
            with th.autocast("cuda", dtype=th.bfloat16):
                lats.append((vae.encode(x).latent_dist.mode() * 0.18215).float()[0].cpu())
            ids.append(int(re.search(r"(\d+)\.png", r["image_path"]).group(1)))
        # seen20 的 id 在训练 shards_std 里 (历史 seen 口径用训练骨架), 合并进来
        idx_std = load_idx(a.shards_std)
        seen_rows = list(csv.DictReader(open(a.seen20_csv, encoding="utf-8")))
        for r in seen_rows:
            iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
            if iid not in ids:
                lats.append(th.from_numpy(get_lat(idx_std, iid)).cpu())
                ids.append(iid)
        np.savez_compressed(out_f, latents=th.stack(lats).numpy().astype(np.float16),
                            img_ids=np.array(ids))
        print(f"[eval] std84+seen shards -> {a.shards_std_eval} ({len(ids)})", flush=True)

    # ---- 训练循环 ----
    t0 = time.time()
    r_img, r_sflow, r_smse, r_gn, cnt = 0.0, 0.0, 0.0, 0.0, 0
    step = 0
    print(f"[train] gen={sum(p.numel() for p in gen_p):,} (stage2 冻结) "
          f"λ_flow={a.lam_skel_flow} λ_mse={a.lam_skel_mse} batch={a.batch}", flush=True)
    while step < a.max_steps:
        k = np.random.randint(0, len(ds), a.batch)
        bs = [ds[int(j)] for j in k]
        x0 = th.stack([b["latent"].float() for b in bs]).to(dev)
        g_std = th.stack([b["skel_latent"].float() for b in bs]).to(dev)
        y = th.tensor([int(b["y_callig"]) for b in bs], dtype=th.long, device=dev)
        y_char = th.tensor([int(b["y_char"]) for b in bs], dtype=th.long, device=dev)
        g_gt = (th.stack([b["aux_latents"].float() for b in bs]).to(dev)
                if aux else None)

        # 1) 端到端采样 (带梯度, 真实 glyph_id)
        # ★ 采样链必须 eval 态: train 态的 cond_drop(10%/步) 在 25 步中让 93% 样本
        #   至少丢一次条件 -> g_pred 全坏 (实测 eval态 vs train态 max diff 2.47)
        gen.eval()
        g_pred = gen_sample(g_std, y, y_char, a.gen_steps_train, dev, grad=True)
        gen.train()
        # 2) L_img (stage2 冻结, 梯度只更新 stage1 经 g_pred)
        # ★ 条件「整块抹白」增强 (2026-10-01, 与 src/train/train.py 同名同义):
        #   推理时喂给 stage2 的是 stage1 **预测**的骨架, 它的失效模式不是"加点噪",
        #   而是"少一笔 / 糊一块"。这里对 g_pred 做**连续区域**抹白 (Z_BG_VEC):
        #   (a) 让 stage2 学会在条件残缺时靠风格条件出字 -> 更鲁棒;
        #   (b) 堵住"生成器靠降低条件信息量讨好 loss"的退化路径。
        _g_in = g_pred
        _gm_prob = float(getattr(a, "glyph_mask_prob", 0.0))
        if _gm_prob > 0:
            from src.utils.deform_aug import Z_BG_VEC
            _z_bg = Z_BG_VEC.to(device=dev, dtype=_g_in.dtype)
            _B, _C, _H, _W = _g_in.shape
            _hit = th.rand(_B, device=dev) < _gm_prob
            _sz = int(getattr(a, "glyph_mask_size", 4))
            _jit = int(getattr(a, "glyph_mask_jitter", 1))
            _nm = int(getattr(a, "glyph_mask_n", 3))
            _lo, _hi = max(2, _sz - _jit), max(2, _sz + _jit)
            _msk = th.zeros(_B, 1, _H, _W, device=dev, dtype=th.bool)
            for _b in range(_B):
                if not bool(_hit[_b]):
                    continue
                for _ in range(_nm):
                    _h = int(th.randint(_lo, _hi + 1, (1,)).item())
                    _w = int(th.randint(_lo, _hi + 1, (1,)).item())
                    _yy = int(th.randint(0, _H - _h + 1, (1,)).item())
                    _xx = int(th.randint(0, _W - _w + 1, (1,)).item())
                    _msk[_b, 0, _yy:_yy + _h, _xx:_xx + _w] = True
            _g_in = th.where(_msk, _z_bg.expand_as(_g_in), _g_in)
        bak_diff_t = th.sigmoid(th.randn(x0.shape[0], device=dev))
        noise = th.randn_like(x0)
        x_t = (1 - bak_diff_t[:, None, None, None]) * x0 + bak_diff_t[:, None, None, None] * noise
        with th.autocast("cuda", dtype=th.bfloat16):
            # ★ y_char 用**真实 glyph_id** (与 eval/oracle 口径一致)。原来传 zeros
            #   等于给 stage2 一个错误字条件 —— 与 stage1 那次同类错误。
            out = bak(x_t, bak_diff_t * TIME_SCALE, y_callig=y,
                      y_char=y_char, g=_g_in)
            if isinstance(out, tuple):
                out = out[0]
            l_img = th.nn.functional.mse_loss(out.float(), noise - x0)
        loss = l_img
        # 3) stage1 原本 flow loss (噪声起步 + GT-w3 目标; 真实 glyph_id)
        l_sflow = th.zeros((), device=dev)
        if a.lam_skel_flow > 0 and g_gt is not None:
            t1 = th.sigmoid(th.randn(x0.shape[0], device=dev))
            z_t = (1 - t1[:, None, None, None]) * g_gt + t1[:, None, None, None] * noise
            v_t = noise - g_gt
            with th.autocast("cuda", dtype=th.bfloat16):
                o1 = gen(z_t, t1 * TIME_SCALE, y_callig=y, y_char=y_char, g=g_std)
                if isinstance(o1, tuple):
                    o1 = o1[0]
                l_sflow = th.nn.functional.mse_loss(o1.float(), v_t)
            loss = loss + a.lam_skel_flow * l_sflow
        # 4) 直接锚 (g_pred vs GT, 防空白塌缩)
        l_smse = th.zeros((), device=dev)
        if a.lam_skel_mse > 0 and g_gt is not None:
            l_smse = th.nn.functional.mse_loss(g_pred.float(), g_gt)
            loss = loss + a.lam_skel_mse * l_smse

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = th.nn.utils.clip_grad_norm_(gen_p, 1.0)
        opt.step()
        sched.step()
        step += 1
        with th.no_grad():
            if step % a.ema_interval == 0:
                for kk, v in gen.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[kk].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                    else:
                        ema[kk].copy_(v)
        r_img += float(l_img)
        r_sflow += float(l_sflow)
        r_smse += float(l_smse)
        r_gn += float(gn)
        cnt += 1
        if step % a.log_every == 0:
            print(f"    step {step:6d}  L_img {r_img/cnt:.4f}  L_skel_flow {r_sflow/cnt:.4f}  "
                  f"L_anchor {r_smse/cnt:.4f}  |grad_gen| {r_gn/cnt:.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.0f}s  "
                  f"peak {th.cuda.max_memory_allocated()/2**30:.2f}G", flush=True)
            r_img = r_sflow = r_smse = r_gn = 0.0
            cnt = 0
            t0 = time.time()

        if step % a.ckpt_every == 0:
            # ★ 共享 ckpt 口径 (与 src/train/train.py 一致): 文件名 `{step:07d}.pt`,
            #   键 `model`(在线) / `ema` / `args` / `step` —— 这样 model_io /
            #   eval_union_ckpt / gen_predskel 等既有工具可直接读。
            #   额外字段供 union 重建用 (stage2 来源 + eval 采样步数)。
            _g_sd = _strip(gen.state_dict())
            _g_ema = _strip(ema)
            th.save({"model": _g_sd, "ema": _g_ema,
                     "gen": _g_sd, "gen_ema": _g_ema,        # 兼容联合脚本口径
                     "gen_ckpt": a.gen_ckpt, "bak_ckpt": a.bak_ckpt,
                     "gen_steps": a.gen_steps_eval, "union": True,
                     "step": step, "args": vars(a)},
                    os.path.join(ckpt_dir, f"{step:07d}.pt"))

        if step % a.eval_every == 0:
            write_std84_shards()
            eval_ckpt(step, ema)
            t0 = time.time()
        if a.smoke and step >= 6:
            break

    print(f"[4] DONE -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
