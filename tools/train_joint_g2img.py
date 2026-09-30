# -*- coding: utf-8 -*-
"""train_joint_g2img.py — 端到端联合训练: 骨架生成器 (latent w7) + v26 主干.

动机 (doc 06): 两阶段断裂在接口 —— 生成器单训时 loss 不知道 decode 后笔画断没断,
主干冻结时没有人优化 "g_pred 是否可被主干使用"。本脚本把两者拼成一个可微链路:
    g_std --[生成器 bridge Euler k 步, hide-g]--> g_pred --[v26 主干 g 条件]--> 图像
只用最终图像的 flow loss, 全解冻 (双参数组), 梯度从图像一路回传到生成器。
无 char 条件架构保证 g 是字形唯一来源, 断笔画必然推高 loss -> 必然被修复。

评测与 doc 06 完全同尺 (复用 in_mem_eval 三口径, summary CSV 同列):
  strict84      g=GT 骨架   (oracle 上界, v26 冻结时 0.8163)
  strict84_pred g=EMA 生成器在线产出的 g_pred (两阶段 w7raw 0.6427)
  seen20        重构口径 (v26 冻结时 0.6517)
"""
import os, sys, json, time, math, argparse, glob, csv, re, copy
import numpy as np
import torch as th
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
TIME_SCALE = 1000.0


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default="assets/skelnet_dit_H_bridge_nog_w7.pt.best")
    ap.add_argument("--bak-ckpt",
                    default="assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt")
    ap.add_argument("--bak-config",
                    default="assets/results/v26_gtskel/20260929-103927-v26-gtskel/resolved_config.json")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--csv", default="assets/train_top10_style23_minusval.csv")
    ap.add_argument("--shards-img", default="data/top10_style23/shards_img")
    ap.add_argument("--shards-std", default="data/top10_style23/shards_std")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--results-dir", default="assets/results/v30_union")
    ap.add_argument("--experiment-name", default="v30-union")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bak-lr", type=float, default=5e-4)
    ap.add_argument("--gen-lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=2500)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--gen-steps-train", type=int, default=8)
    ap.add_argument("--gen-steps-eval", type=int, default=25)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--ema-interval", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--strict-csv", default="assets/eval_v13_strict_fixed.csv")
    ap.add_argument("--seen-csv", default="assets/eval_top10_seen_20.csv")
    ap.add_argument("--strict-shards-gt", default="data/top10_style23/gt_skel_eval_strict84")
    ap.add_argument("--pred-shards-out", default="data/top10_style23/predskel_eval_strict84_joint")
    ap.add_argument("--std84-cache", default="data/top10_style23/std84_latents.npz")
    ap.add_argument("--glyph-noise-scale", type=float, default=0.3)
    ap.add_argument("--glyph-noise-prob", type=float, default=0.5)
    ap.add_argument("--glyph-patch-drop", type=float, default=0.05)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build_models(a, dev):
    from src.model.dit import DiT_2Cond, DiT_2Cond_models
    r = json.load(open(a.bak_config, encoding="utf-8"))

    bak = DiT_2Cond_models[r.get("model", "DiT-2Cond-S/2")](
        input_size=32, num_calligraphers=int(r["num_calligraphers"]),
        num_characters=int(r["num_characters"]), use_checkpoint=False,
        learn_sigma=False, condition_fusion=r["condition_fusion"],
        cond_fusion_norm=r.get("cond_fusion_norm", "split"),
        callig_embed_dim=int(r["callig_embed_dim"]),
        char_embed_dim=int(r.get("char_embed_dim") or 384),
        glyph_vec_cond=bool(r.get("glyph_vec_cond", True)),
        glyph_vec_dim=int(r.get("glyph_vec_dim", 128)),
        glyph_vec_pool=r.get("glyph_vec_pool", "mean"),
        cond_drop_all_prob=float(r["cond_drop_all_prob"]),
        cond_drop_one_prob=float(r.get("cond_drop_one_prob", 0.0)),
        cond_drop_which_glyph_prob=float(r.get("cond_drop_which_glyph_prob", 0.85)),
        use_glyph_cond=True, use_char_cond=False,
        glyph_scale_init=float(r.get("glyph_scale_init", 0.6)),
        glyph_drop_prob=float(r.get("glyph_drop_prob", 0.0)),
        glyph_inject_layers=int(r.get("glyph_inject_layers", 4)),
        glyph_inject_mode=r.get("glyph_inject_mode", "adaln"),
        glyph_embedder_depth=int(r.get("glyph_embedder_depth", 2)),
        glyph_embedder_sep=bool(r.get("glyph_embedder_sep", False)),
        glyph_in_channels=4, in_channels=4,
        char_proj_mode=r.get("char_proj_mode", "full"),
        attn_impl=r.get("attn_impl", "sdpa"),
        norm_type=r.get("norm_type", "rms"), mlp_type=r.get("mlp_type", "swiglu"),
        qk_norm=bool(r.get("qk_norm", 1)), rope=bool(r.get("rope", 1)),
        rope_theta=float(r.get("rope_theta", 100.0))).to(dev)

    # 生成器: 逐字对齐 train_skelnet_dit.py 的构造 (默认 arch, 保权重逐位一致)
    n_slots = int(r["num_calligraphers"])
    gen = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4, out_channels=4,
        depth=6, hidden_size=256, num_heads=4,
        num_calligraphers=n_slots, num_characters=1,
        use_char_cond=False, use_glyph_cond=True, glyph_in_channels=4,
        glyph_inject_layers=2, glyph_inject_mode="adaln",
        glyph_scale_init=0.6, glyph_drop_prob=0.0, glyph_embedder_depth=2,
        condition_fusion="factorized_cat", callig_embed_dim=128,
        glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=0.1, cond_drop_one_prob=0.0,
        learn_sigma=False).to(dev)

    from src.utils.channel_expand import materialize_lazy_params
    dg = th.load(a.gen_ckpt, map_location="cpu", weights_only=False)
    gs = _strip(dg.get("ema") or dg)
    lazy_g = materialize_lazy_params(gen, gs)
    m1, u1 = gen.load_state_dict(gs, strict=False)
    assert len(u1) == 0, f"gen unexpected {u1[:4]}"
    db = th.load(a.bak_ckpt, map_location="cpu", weights_only=False)
    bs = _strip(db.get("ema") or db.get("model") or db)
    lazy_b = materialize_lazy_params(bak, bs)
    bak.to(dev)   # 懒参数 (null_embed) 创建在 CPU, 搬上去
    m2, u2 = bak.load_state_dict(bs, strict=False)
    assert len(u2) == 0, f"bak unexpected {u2[:4]}"
    # v26 语义: callig 表冻结, null_embed (CFG uncond) 可训 —— materialize 后保持可训
    for p in bak.y_callig_embedder.parameters():
        if p is getattr(bak.y_callig_embedder, "null_embed", None):
            p.requires_grad = True
        else:
            p.requires_grad = False
    print(f"[model] gen miss={len(m1)} lazy={lazy_g} | bak miss={len(m2)} lazy={lazy_b}",
          flush=True)
    return gen, bak


def gen_sample(gen, g_std, y, steps):
    """bridge Euler (hide-g, g 输入恒零): z=g_std -> g_pred. y: (b,) slot id.
    ⚠ 无 no_grad: 联合训练需要梯度穿过采样链回传生成器 (eval 侧调用处自行包)."""
    b = g_std.shape[0]
    z = g_std.clone()
    ss = np.linspace(1.0, 0.0, steps + 1)
    zero_g = th.zeros_like(g_std)
    yc = th.zeros((b,), dtype=th.long, device=g_std.device)
    for k in range(steps):
        t_i = th.full((b,), float(ss[k]), device=g_std.device)
        with th.autocast("cuda", dtype=th.bfloat16):
            out = gen(z, t_i * TIME_SCALE, y_callig=y, y_char=yc, g=zero_g)
        if isinstance(out, tuple):
            out = out[0]
        z = z + (float(ss[k + 1]) - float(ss[k])) * out.float()
    return z


class _Tee:
    """stdout 双写: 终端 + run 目录 log.txt (对齐 train.py 的 FileHandler 约定)."""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.s = sys.stdout

    def write(self, m):
        self.s.write(m)
        self.f.write(m)

    def flush(self):
        self.s.flush()
        self.f.flush()


def main():
    a = parse_args()
    th.manual_seed(a.seed)
    dev = th.device("cuda")
    # 标准 run 目录: assets/results/v30_union/<ts>-v30-union/
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(a.results_dir, f"{ts}-{a.experiment_name}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    sys.stdout = _Tee(os.path.join(run_dir, "log.txt"))
    sys.stderr = sys.stdout
    json.dump({k: v for k, v in vars(a).items()},
              open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"[run_dir] {run_dir}", flush=True)
    resume_from = None
    _rck = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "0*.pt")))
    if _rck:
        resume_from = _rck[-1]
        print(f"[resume] 发现 ckpt {os.path.basename(resume_from)}", flush=True)

    from src.utils.callig_script_map import map_callig_script
    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    gen, bak = build_models(a, dev)  # regen_pred_shards 的槽位映射用 csmap
    gen.train(), bak.train()

    # EMA 评测副本 (不覆盖训练权重)
    gen_eval = copy.deepcopy(gen).eval()
    bak_eval = copy.deepcopy(bak).eval()
    for p in list(gen_eval.parameters()) + list(bak_eval.parameters()):
        p.requires_grad_(False)

    # ---- 数据: MCCDLatentDataset 多进程预载 (与 train_skelnet_dit 同机制) ----
    from src.utils.latent_dataset import MCCDLatentDataset
    from src.utils.callig_script_map import load_callig_script_map
    csmap_obj = load_callig_script_map(a.callig_map)

    def make_ds(csv_path, train):
        return MCCDLatentDataset(
            csv_file=csv_path, latent_shards_dir=a.shards_img, img_root="",
            image_size=256, is_train=train, preload=True, load_image=False,
            skel_latent_shards_dir=a.shards_std,
            callig_id_map=None, callig_script_map=csmap_obj)

    ds = make_ds(a.csv, True)
    print(f"[data] rows={len(ds)} (预载完成)", flush=True)

    # ---- 优化器: 双参数组 ----
    gen_p = [p for p in gen.parameters() if p.requires_grad]
    bak_p = [p for p in bak.parameters() if p.requires_grad]
    opt = th.optim.AdamW([
        {"params": bak_p, "lr": a.bak_lr},
        {"params": gen_p, "lr": a.gen_lr},
    ], weight_decay=a.wd)

    def lr_scale(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        prog = (s - a.warmup) / max(a.max_steps - a.warmup, 1)
        return a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = th.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    ema_gen = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    ema_bak = {k: v.detach().clone() for k, v in bak.state_dict().items()}

    def ema_update(d, sd):
        with th.no_grad():
            for k, v in sd.items():
                if v.dtype.is_floating_point:
                    d[k].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                else:
                    d[k].copy_(v)

    # ---- 评测 (复用 in_mem_eval; EMA 载入评测副本, 训练权重不动) ----
    from src.eval.in_mem_eval import run_in_mem_eval, _get_vae
    from types import SimpleNamespace
    rcfg = json.load(open(a.bak_config, encoding="utf-8"))

    def make_eval_args():
        ev = SimpleNamespace(**{k: v for k, v in rcfg.items() if not k.startswith("_")})
        ev.eval_csv = getattr(a, "strict84_aligned_csv",
                              os.path.join(os.path.dirname(a.strict_csv),
                                           "eval_v13_strict84_aligned.csv"))
        ev.eval_skel_latent_shards_dir = a.strict_shards_gt
        ev.eval_skel_latent_shards_dir_pred = a.pred_shards_out
        ev.eval_blend_alpha = 0.0
        ev.eval_cfg = 1.0
        ev.eval_steps = 50
        ev.eval_self_cond = False
        ev.img_root = None
        return ev

    @th.no_grad()
    def regen_pred_shards():
        gen_eval.load_state_dict(ema_gen)   # eval 用 EMA 副本 + 显式 no_grad
        gen_eval.eval()
        vae = _get_vae(dev, a.vae)
        os.makedirs(a.pred_shards_out, exist_ok=True)
        rows84 = list(csv.DictReader(open(a.strict_csv, encoding="utf-8")))
        # id = image_path 数字 (csv 无 img_id 列); 只取 GT shards 覆盖的 84 个 (对齐 oracle 口径)
        gt_ids = set()
        for sp in glob.glob(os.path.join(a.strict_shards_gt, "shard_*.npz")):
            with np.load(sp) as z:
                gt_ids |= {int(i) for i in z["img_ids"]}
        sel = []
        for r in rows84:
            m = re.search(r"(\d+)\.png$", r.get("image_path", ""))
            if m and int(m.group(1)) in gt_ids:
                sel.append(r)
        ids84 = [int(re.search(r"(\d+)\.png$", r["image_path"]).group(1)) for r in sel]
        print(f"[eval] strict84: {len(sel)}/{len(rows84)} 行与 GT shards 对齐", flush=True)
        # 对齐子集 csv (in_mem_eval 按顺序取前 n 行 -> 必须喂过滤后的顺序, 否则 g 错位)
        align_csv = os.path.join(os.path.dirname(a.strict_csv), "eval_v13_strict84_aligned.csv")
        with open(align_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sel[0].keys()))
            w.writeheader()
            w.writerows(sel)
        a.strict84_aligned_csv = align_csv
        if os.path.exists(a.std84_cache):
            z = np.load(a.std84_cache)
            cached = {int(i): v for i, v in zip(z["img_ids"], z["latents"])}
            lat84 = th.stack([th.from_numpy(cached[i].astype(np.float32)) for i in ids84])
        else:
            lats = []
            for r in sel:
                p = r["std_path"] if os.path.isabs(r["std_path"]) else os.path.join(ROOT, r["std_path"])
                img = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
                v = 1.0 - 2.0 * img
                x = th.from_numpy(v)[None, None].repeat(1, 3, 1, 1).to(dev)
                lat = (vae.encode(x).latent_dist.mode() * 0.18215).float()[0].cpu()
                lats.append(lat)
            lat84 = th.stack(lats)
            np.savez_compressed(a.std84_cache, img_ids=np.array(ids84),
                                latents=lat84.numpy().astype(np.float16))
        lat84 = lat84.to(dev)
        outs = th.zeros_like(lat84)
        B = 16
        for s in range(0, len(sel), B):
            rs = sel[s:s + B]
            y = th.tensor([int(map_callig_script(int(r["calligrapher_id"]),
                                                 int(r["script_id"]), csmap))
                           for r in rs], device=dev)
            outs[s:s + B] = gen_sample(gen_eval, lat84[s:s + B], y, a.gen_steps_eval)
        np.savez_compressed(os.path.join(a.pred_shards_out, "shard_00000.npz"),
                            img_ids=np.array(ids84),
                            latents=outs.cpu().numpy().astype(np.float16))

    def run_eval(step):
        regen_pred_shards()
        bak_eval.load_state_dict(ema_bak)
        bak_eval.eval()
        ev_args = make_eval_args()
        _al = getattr(a, "strict84_aligned_csv", None) or a.strict_csv
        sets = [("strict84", _al, 84),
                ("strict84_pred", _al, 84),
                ("seen20", a.seen_csv, 20)]
        try:
            res = run_in_mem_eval(bak_eval, ev_args, step, dev, run_dir, sets=sets)
            print(f"[eval@{step}] " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()),
                  flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
        bak.train(), gen.train()

    if a.smoke:
        batch = [ds[i] for i in range(4)]
        x0 = th.stack([b["latent"] for b in batch]).to(dev)
        g = th.stack([b["skel_latent"] for b in batch]).to(dev).float()
        y = th.stack([b["y_callig"] for b in batch]).to(dev).long().view(-1)
        g_pred = gen_sample(gen, g, y, 4)
        g_pred.retain_grad()
        print(f"[smoke] g_pred {tuple(g_pred.shape)} requires_grad={g_pred.requires_grad} "
              f"|Δ(g_std)|={(g_pred - g).abs().mean():.4f}", flush=True)
        t = th.sigmoid(th.randn(4, device=dev))
        noise = th.randn_like(x0)
        x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * noise
        out = bak(x_t, t * TIME_SCALE, y_callig=y, y_char=th.zeros_like(y), g=g_pred)
        out = out[0] if isinstance(out, tuple) else out
        # 诊断 1: 直接对 g_pred 求梯度的通路 (绕过 diff loss)
        probe = (g_pred * th.randn_like(g_pred)).sum()
        probe.backward(retain_graph=True)
        pg = sum(p.grad.norm().item() ** 2 for p in gen.parameters()
                 if p.grad is not None) ** 0.5
        print(f"[smoke] probe g_pred->gen gradNorm={pg:.4f} "
              f"g_pred.grad norm={g_pred.grad.norm().item():.4f}", flush=True)
        gen.zero_grad(set_to_none=True)
        # 诊断 2: 完整 diff loss
        loss = th.nn.functional.mse_loss(out, noise - x0)
        loss.backward()
        pg2 = sum(p.grad.norm().item() ** 2 for p in gen.parameters()
                  if p.grad is not None) ** 0.5
        bgrad = sum(p.grad.norm().item() ** 2 for p in bak.parameters()
                    if p.grad is not None) ** 0.5
        gemb = sum(p.grad.norm().item() ** 2 for p in bak.glyph_embedder.parameters()
                   if p.grad is not None) ** 0.5
        print(f"[smoke] loss={float(loss):.4f} gen_gradNorm={pg2:.4f} "
              f"bak_gradNorm={bgrad:.3f} bak_glyphEmbedder_gradNorm={gemb:.4f}", flush=True)
        return

    # ---- 训练循环 ----
    start_step = 0
    if resume_from:
        d = th.load(resume_from, map_location="cpu", weights_only=False)
        gen.load_state_dict(_strip(d["gen_ema"]))
        bak.load_state_dict(_strip(d["bak_ema"]))
        ema_gen = {k: v.detach().clone() for k, v in _strip(d["gen_ema"]).items()}
        ema_bak = {k: v.detach().clone() for k, v in _strip(d["bak_ema"]).items()}
        start_step = int(d["step"])
        for _ in range(start_step):
            sched.step()
        print(f"[resume] 权重+EMA 恢复, step={start_step} (opt/sched 状态 ckpt 未存, 重走 cosine)", flush=True)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    print(f"[train] bak={sum(p.numel() for p in bak_p):,} "
          f"gen={sum(p.numel() for p in gen_p):,} ds={len(ds)}", flush=True)
    step = start_step
    while step < a.max_steps:
        sel = np.random.randint(0, len(ds), a.batch)
        batch = [ds[int(i)] for i in sel]
        x0 = th.stack([b["latent"] for b in batch]).to(dev)
        g = th.stack([b["skel_latent"] for b in batch]).to(dev).float()
        y = th.stack([b["y_callig"] for b in batch]).to(dev).long().view(-1)

        g_pred = gen_sample(gen, g, y, a.gen_steps_train)

        if np.random.rand() < a.glyph_noise_prob:
            g_pred = g_pred + th.randn_like(g_pred) * a.glyph_noise_scale
        if a.glyph_patch_drop > 0 and np.random.rand() < 0.5:
            ps = 8
            px, py = np.random.randint(0, 32 - ps), np.random.randint(0, 32 - ps)
            g_pred[:, :, px:px + ps, py:py + ps] = 0.0

        t = th.sigmoid(th.randn(a.batch, device=dev))
        noise = th.randn_like(x0)
        x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * noise
        v_t = noise - x0
        with th.autocast("cuda", dtype=th.bfloat16):
            out = bak(x_t, t * TIME_SCALE, y_callig=y, y_char=th.zeros_like(y), g=g_pred)
            if isinstance(out, tuple):
                out = out[0]
            loss = th.nn.functional.mse_loss(out.float(), v_t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = th.nn.utils.clip_grad_norm_(bak_p + gen_p, 1.0)
        opt.step()
        sched.step()
        step += 1
        run_loss += float(loss)
        run_n += 1
        if step % a.ema_interval == 0:
            ema_update(ema_gen, gen.state_dict())
            ema_update(ema_bak, bak.state_dict())

        if step % a.log_every == 0:
            sps = run_n / max(time.time() - t0, 1e-6)
            mem = th.cuda.memory_reserved() / 1024 ** 3
            print(f"({step:07d}) loss={run_loss / run_n:.4f} "
                  f"LR bak={opt.param_groups[0]['lr']:.2e} gen={opt.param_groups[1]['lr']:.2e} "
                  f"gN={float(gn):.2f} | {sps:.2f} sps | {mem:.1f}G", flush=True)
            run_loss, run_n, t0 = 0.0, 0, time.time()

        if step % a.ckpt_every == 0:
            th.save({"gen_ema": {k: v.cpu() for k, v in ema_gen.items()},
                     "bak_ema": {k: v.cpu() for k, v in ema_bak.items()},
                     "step": step, "args": vars(a)},
                    os.path.join(ckpt_dir, f"{step:07d}.pt"))

        if step % a.eval_every == 0 and not a.no_eval:
            run_eval(step)
            t0 = time.time()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
