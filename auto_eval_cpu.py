#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""auto_eval_cpu.py — 独立 CPU 评测进程（与 train.py 完全解耦）。

train.py 现在只训练 + 保存 ckpt（GPU 只训练，不做 eval）。本进程在 CPU 上轮询
当前实验的 ckpt 目录，发现新 checkpoint（*.pt 且带 .done 完成标记）即评测：

  1) 指标: eval100（eval_csv, n=eval_n）自由采样 DDIM 算 MSE/SSIM
            -> <ckpt_dir>/eval_auto_{step}.json
  2) 展示: show5（固定 unseen 5 样本）-> eval_latest.png + eval_samples/stepXXXXXX/
  3) 展示: seen5（训练集样本，不进入任何指标）-> seen_samples/stepXXXXXX/

权重加载优先 EMA（use_ema 时用 ckpt['ema']），否则用 ckpt['delta']。
模型/VAE/采样参数全部取自 ckpt 内保存的 args，确保与训练架构完全一致；
eval 的参数（--eval-n / --steps / --cfg / --batch）可在命令行覆盖，无需改训练。

用法:
  python auto_eval_cpu.py --results-dir 5script/results/<exp> [--interval 30]
                          [--seen5-csv 5script/seen5_top30.csv]
                          [--eval-n 100] [--steps 50] [--cfg 4.0] [--batch 16]
                          [--threads 96] [--once]
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import datetime
import multiprocessing as mp

import torch

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_SEEN5_CSV = "5script/seen5_top30.csv"


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 模型 / VAE 构建（复刻 train.py 的构建与加载顺序，保证架构一致）
# ---------------------------------------------------------------------------

def build_model(args, device="cpu"):
    latent_size = args.image_size // 8
    cond_mode = getattr(args, "cond_mode", "2cond")
    if cond_mode == "3cond":
        from models import DiT_3Cond_models
        if args.model not in DiT_3Cond_models:
            raise ValueError(f"cond_mode=3cond but model '{args.model}' not a 3Cond model")
        model = DiT_3Cond_models[args.model](
            input_size=latent_size,
            num_calligraphers=args.num_calligraphers,
            num_scripts=args.num_scripts,
            num_characters=args.num_characters,
            use_checkpoint=args.use_checkpoint,
            condition_fusion=args.condition_fusion,
            callig_embed_dim=args.callig_embed_dim,
            script_embed_dim=args.script_embed_dim,
            char_embed_dim=args.char_embed_dim,
            cond_drop_all_prob=args.cond_drop_all_prob,
            cond_drop_one_prob=args.cond_drop_one_prob,
        )
    else:
        from models import DiT_2Cond_models
        if args.model not in DiT_2Cond_models:
            raise ValueError(f"cond_mode=2cond but model '{args.model}' not a 2Cond model")
        model = DiT_2Cond_models[args.model](
            input_size=latent_size,
            num_calligraphers=args.num_calligraphers,
            num_characters=args.num_characters,
            use_checkpoint=args.use_checkpoint,
            condition_fusion=args.condition_fusion,
            callig_embed_dim=args.callig_embed_dim,
            char_embed_dim=args.char_embed_dim,
            cond_drop_all_prob=args.cond_drop_all_prob,
            cond_drop_one_prob=args.cond_drop_one_prob,
            skel_head_enabled=getattr(args, "w_skel_head", 0) > 0,
            use_glyph_cond=getattr(args, "w_glyph_cond", 0) > 0,
            glyph_scale_init=getattr(args, "glyph_scale_init", 0.4),
        )
    # 1) pretrained body（与 train.py 一致，过滤条件头键）
    if getattr(args, "pretrained", None):
        from download import find_model
        state_dict = find_model(args.pretrained)
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith(("y_embedder", "y_callig", "y_script",
                                          "y_char", "cond_fusion"))}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log(f"[model] loaded pretrained body {args.pretrained} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})")
    # 2) LoRA 注入（与训练一致；本实验 use_lora=false 时跳过）
    if getattr(args, "use_lora", True):
        from lora import inject_lora
        _r = getattr(args, "lora_r", 16)
        _alpha = getattr(args, "lora_alpha", None)
        if _alpha is None:
            _alpha = _r
        model = inject_lora(model, r=_r, lora_alpha=_alpha,
                            target=getattr(args, "lora_target", "all"))
        log(f"[model] injected LoRA r={_r}")
    model = model.to(device).eval()
    return model


def load_ckpt_weights(model, ckpt, ckpt_name):
    sd = ckpt.get("ema")
    src = "ema"
    if sd is None:
        sd = ckpt.get("delta", ckpt.get("model"))
        src = "delta"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"[model] loaded weights from {src} ({ckpt_name}): "
        f"missing={len(missing)}, unexpected={len(unexpected)}")
    return src


def load_vae(args, device="cpu"):
    from diffusers.models import AutoencoderKL
    path = getattr(args, "vae_path", None)
    if path and os.path.exists(path):
        log(f"[vae] loading {path}")
        return AutoencoderKL.from_pretrained(path).to(device).eval()
    log(f"[vae] loading stabilityai/sd-vae-ft-{args.vae}")
    return AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()


# ---------------------------------------------------------------------------
# 数据缓存
# ---------------------------------------------------------------------------

def build_caches(args, seen5_csv):
    from eval_auto import prepare_gen_cache
    from dataset import MCCDDataset

    def cache_for(csv_path, n):
        if not csv_path or not os.path.exists(csv_path):
            log(f"[cache] skip missing csv: {csv_path}")
            return None
        ds = MCCDDataset(csv_file=csv_path, root_dir=args.data_dir,
                         image_size=args.image_size, load_canny=False, load_skel=False,
                         use_glyph_cond=getattr(args, "w_glyph_cond", False))
        c = prepare_gen_cache(ds, n=n, cond_mode=args.cond_mode)
        log(f"[cache] {csv_path} -> {len(c['conds'])} samples")
        return c

    eval_cache = cache_for(getattr(args, "eval_csv", None), int(getattr(args, "eval_n", 100)))
    show5_cache = cache_for(getattr(args, "show5_csv", None), 100)
    seen5_cache = cache_for(seen5_csv, 5)
    return eval_cache, show5_cache, seen5_cache


# ---------------------------------------------------------------------------
# 常驻 worker 池：eval100 数据并行（data-parallel，进程间不共享计算）。
# 用 fork 继承父进程已加载的 model/vae/cache（写时复制，几乎不占额外内存），
# 每 worker 只跑自己那份样本、用少量线程——单进程 64 线程的 fork/join 开销
# 吃掉了 batch 并行收益，多进程才能真正用满这台机器的核。
#
# 重要坑（两次）：
#   1) step2000 卡死 7h：fork 发生在父进程已跑过 torch 多线程(OpenMP 池已建)
#      之后，子进程继承损坏的线程池而挂死。曾改成【每轮先 fork 再跑展示】。
#   2) step3000 仍慢(40min vs 9min)：第二次 fork 时父进程虽 set_num_threads(1)，
#      但已建好的 OpenMP 池缩不回去，worker 仍被拖慢。
#
# 根治方案：启动时【一次性 fork 常驻 worker】（此时父进程一定在单线程、无任何
# 多线程工作之后才建池，见 main：build_model/cache 期间线程=1），之后每轮只发
# 任务（worker 自己 torch.load 换权重 + 跑自己那份），【永不再 fork】。
# 任一 worker 失败/超时/死亡 -> 本轮回退单进程，且整池废弃（broken），不再复用，
# 绝不让整轮卡死。
# ---------------------------------------------------------------------------

_SH = {}

WORKER_TIMEOUT = 1800  # 单轮 eval100 最长秒数（100/8≈13 张 x 50 步，>15 分钟视为挂死）


class EvalPool:
    __slots__ = ("q_task", "q_res", "procs", "ranges", "broken")


def _pool_worker(q_task, q_res, start, end):
    try:
        torch.set_num_threads(_SH["worker_threads"])
        from eval_auto import eval_gen_in_memory
        while True:
            task = q_task.get()
            if task is None:
                break
            ckpt_path, base, step = task
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                load_ckpt_weights(_SH["model"], ckpt, base)
                cache = _SH["cache"]
                sub = {
                    "conds": cache["conds"][start:end],
                    "gts": cache["gts"][start:end],
                    "gs": cache.get("gs")[start:end] if cache.get("gs") is not None else None,
                }
                mse, ssim = eval_gen_in_memory(
                    _SH["model"], _SH["vae"], "cpu", sub,
                    n=end - start, steps=_SH["steps"], cfg=_SH["cfg"], seed=_SH["seed"],
                    batch=_SH["batch"], vis_out=None, vis_n=0,
                    cond_mode=_SH["cond_mode"], save_samples_dir=None, step=None,
                    glyph_init_mix=_SH["glyph_init_mix"])
                q_res.put(("ok", start, end, mse, ssim))
            except Exception as e:
                q_res.put(("err", start, end, str(e)))
    except Exception as e:
        try:
            q_res.put(("fatal", start, end, str(e)))
        except Exception:
            pass


def start_pool(model, vae, cache, cfg, workers, worker_threads):
    """启动时一次性 fork 常驻 worker（必须在任何 torch 多线程工作之前调用）。
    返回 EvalPool。"""
    n = len(cache["conds"])
    sizes = [n // workers + (1 if i < n % workers else 0) for i in range(workers)]
    ranges, s = [], 0
    for sz in sizes:
        ranges.append((s, s + sz))
        s += sz
    global _SH
    _SH.update(model=model, vae=vae, cache=cache,
               steps=int(cfg["steps"]), cfg=float(cfg["cfg"]),
               seed=int(cfg["seed"]), batch=int(cfg["batch"]),
               cond_mode=cfg["cond_mode"],
               glyph_init_mix=float(cfg["glyph_init_mix"]),
               worker_threads=worker_threads)
    ctx = mp.get_context("fork")
    q_task, q_res = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=_pool_worker, args=(q_task, q_res, st, en), daemon=True)
             for st, en in ranges]
    for p in procs:
        p.start()
    pool = EvalPool()
    pool.q_task, pool.q_res = q_task, q_res
    pool.procs, pool.ranges, pool.broken = procs, ranges, False
    log(f"[eval100] spawned persistent pool: {len(procs)} workers x {worker_threads} "
        f"threads (fork 一次, 常驻)")
    return pool


def pool_submit(pool, ckpt_path, base, step):
    """给每个 worker 发一份本轮任务（各跑自己那份样本）。
    返回是否已提交；池已 broken / 有 worker 死亡则返回 False。"""
    if pool is None or pool.broken:
        return False
    if not all(p.is_alive() for p in pool.procs):
        pool.broken = True
        log("[eval100] pool worker(s) dead -> 弃用, 回退单进程")
        return False
    task = (ckpt_path, base, step)
    for _ in pool.procs:
        pool.q_task.put(task)
    return True


def pool_collect(pool, timeout=WORKER_TIMEOUT):
    """等待本轮全部 worker 结果（带超时），汇总加权 MSE/SSIM。
    返回 (mse, ssim, ok)；失败时置 broken 并关池，调用方回退单进程。"""
    deadline = time.time() + timeout
    got, errs = {}, []
    for _ in range(len(pool.procs)):
        remaining = deadline - time.time()
        if remaining <= 0:
            errs.append("timeout")
            break
        try:
            parts = pool.q_res.get(timeout=remaining)
            tag, st, en = parts[0], parts[1], parts[2]
            if tag == "ok":
                got[(st, en)] = (parts[3], parts[4])
            else:
                errs.append(str(parts[3]) if len(parts) > 3 else parts)
        except Exception:
            errs.append("no-result")
    missing = [(st, en) for (st, en) in pool.ranges if (st, en) not in got]
    if missing or errs:
        log(f"[eval100] {len(errs)} err + {len(missing)} missing -> 回退单进程 "
            f"({errs[:1]})")
        pool.broken = True
        pool_shutdown(pool)
        return None, None, False
    t_mse = t_ssim = t_cnt = 0.0
    for (st, en), (mse, ssim) in got.items():
        c = en - st
        t_mse += mse * c
        t_ssim += ssim * c
        t_cnt += c
    return t_mse / t_cnt, t_ssim / t_cnt, True


def pool_shutdown(pool):
    if pool is None:
        return
    for _ in pool.procs:
        pool.q_task.put(None)
    for p in pool.procs:
        p.join(timeout=10)
    for p in pool.procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    log("[eval100] pool shut down")


# ---------------------------------------------------------------------------
# 单 ckpt 评测
# ---------------------------------------------------------------------------

def eval_one(model, vae, device, ckpt_dir, step, caches, cfg,
             pool=None, ckpt_path=None, base=None):
    from eval_auto import eval_gen_in_memory
    eval_cache, show5_cache, seen5_cache = caches

    steps = int(cfg["steps"])
    sample_cfg = float(cfg["cfg"])
    seed = int(cfg["seed"])
    metric_batch = int(cfg["batch"])
    workers = int(cfg.get("workers", 1))
    worker_threads = int(cfg.get("worker_threads", 8))

    result = {"step": step, "mse": None, "ssim": None}

    # 0) 向常驻池提交本轮 eval100 任务（worker 与父进程的展示并行跑）。
    #    池在启动时已一次性 fork，此后父进程随便用多线程，无再次 fork。
    submitted = False
    if pool is not None and eval_cache is not None and len(eval_cache["conds"]) >= workers:
        submitted = pool_submit(pool, ckpt_path, base, step)

    # 父进程线程数：展示用（不影响已 fork 的常驻池）
    torch.set_num_threads(max(1, min(worker_threads, 8)))

    # 1) 展示：show5（固定 unseen 5 样本）
    if show5_cache is not None:
        n = len(show5_cache["conds"])
        log(f"[eval] step {step}: show5 display (n={n}) ...")
        eval_gen_in_memory(
            model, vae, device, show5_cache,
            n=n, steps=steps, cfg=sample_cfg, seed=seed, batch=min(n, 16),
            vis_out=f"{ckpt_dir}/eval_latest.png", vis_n=n,
            cond_mode=cfg["cond_mode"],
            save_samples_dir=f"{ckpt_dir}/eval_samples", step=step,
            glyph_init_mix=float(cfg["glyph_init_mix"]))

    # 2) 展示：seen5（训练集样本，不进入任何指标）
    if seen5_cache is not None:
        n = len(seen5_cache["conds"])
        log(f"[eval] step {step}: seen5 display (n={n}) ...")
        eval_gen_in_memory(
            model, vae, device, seen5_cache,
            n=n, steps=steps, cfg=sample_cfg, seed=seed, batch=min(n, 16),
            vis_out=None, vis_n=n,
            cond_mode=cfg["cond_mode"],
            save_samples_dir=f"{ckpt_dir}/seen_samples", step=step,
            glyph_init_mix=float(cfg["glyph_init_mix"]))

    # 3) 指标：eval100（最慢；常驻池 worker 已在前台跑完，这里汇总；失败则回退单进程）
    if eval_cache is not None:
        n = len(eval_cache["conds"])
        t0 = time.time()
        mse = ssim = None
        if submitted:
            log(f"[eval] step {step}: free-sampling eval100 (n={n}, workers={workers}) ...")
            mse, ssim, used = pool_collect(pool)
            if not used:
                log(f"[eval] step {step}: 并行失败，回退单进程 eval100 ...")
        if mse is None:
            log(f"[eval] step {step}: free-sampling eval100 (n={n}, steps={steps}, cfg={sample_cfg}) ...")
            mse, ssim = eval_gen_in_memory(
                model, vae, device, eval_cache,
                n=n, steps=steps, cfg=sample_cfg, seed=seed, batch=metric_batch,
                vis_out=None, vis_n=5, cond_mode=cfg["cond_mode"],
                save_samples_dir=None, step=None,
                glyph_init_mix=float(cfg["glyph_init_mix"]))
        result.update(mse=mse, ssim=ssim)
        log(f"[eval] step {step}: free-sampling MSE={mse:.5f} SSIM={ssim:.4f} "
            f"({time.time()-t0:.0f}s)")
        with open(f"{ckpt_dir}/eval_auto_{step:07d}.json", "w") as _ef:
            json.dump(result, _ef)

    return result


# ---------------------------------------------------------------------------
# 轮询主循环
# ---------------------------------------------------------------------------

def read_active_ckpt_dir(results_dir):
    marker = os.path.join(results_dir, "_active_ckpt_dir.txt")
    if not os.path.exists(marker):
        return None
    with open(marker, encoding="utf-8") as f:
        return f.read().strip() or None


def load_state(ckpt_dir):
    sp = os.path.join(ckpt_dir, "cpu_eval_state.json")
    try:
        with open(sp, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(ckpt_dir, state):
    sp = os.path.join(ckpt_dir, "cpu_eval_state.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None,
                    help="训练 results 目录（train.py 在此写 _active_ckpt_dir.txt）")
    ap.add_argument("--ckpt-dir", default=None,
                    help="直接指定 ckpt 目录（优先于轮询 _active_ckpt_dir.txt）")
    ap.add_argument("--interval", type=int, default=30, help="轮询间隔秒")
    ap.add_argument("--once", action="store_true", help="只处理当前所有新 ckpt 一次后退出")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="eval 设备（cuda 用于 GPU 批量评测，自动选 cuda）")
    ap.add_argument("--threads", type=int, default=0, help="torch 线程数（0=默认全核）")
    ap.add_argument("--workers", type=int, default=1, help="eval100 数据并行进程数（fork 继承，>1 启用）")
    ap.add_argument("--worker-threads", type=int, default=8, help="每 worker 的线程数")
    ap.add_argument("--seen5-csv", default=DEFAULT_SEEN5_CSV)
    ap.add_argument("--eval-n", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    if not args.ckpt_dir and not args.results_dir:
        ap.error("must provide --results-dir or --ckpt-dir")
    if args.device == "cuda" and not torch.cuda.is_available():
        log("[warn] --device cuda but CUDA unavailable, fallback to cpu")
        args.device = "cpu"
    if args.device == "cuda":
        log(f"using device cuda ({torch.cuda.get_device_name(0)})")

    if args.workers > 1:
        # 多进程模式：父进程必须单线程，保证 start_pool 的 fork 永远干净（覆盖 --threads）
        torch.set_num_threads(1)
        log("torch threads = 1 (multiprocess eval100; fork 需单线程父进程)")
    elif args.threads > 0:
        torch.set_num_threads(args.threads)
        log(f"torch threads set to {torch.get_num_threads()}")
    else:
        log(f"torch threads = {torch.get_num_threads()}")

    results_dir = None
    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
        os.makedirs(results_dir, exist_ok=True)

    model = vae = caches = None
    model_args = None
    last_ckpt_dir = None
    pool = None

    while True:
        ckpt_dir = args.ckpt_dir or read_active_ckpt_dir(results_dir)
        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            log(f"[wait] no active ckpt dir yet ({results_dir}) ...")
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        # 实验切换：重置模型/缓存/状态/池
        if ckpt_dir != last_ckpt_dir:
            log(f"[watch] active ckpt dir: {ckpt_dir}")
            if pool is not None:
                pool_shutdown(pool)
                pool = None
            model = vae = caches = None
            model_args = None
            last_ckpt_dir = ckpt_dir
            state = load_state(ckpt_dir)

        done_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        if not done_files:
            log(f"[wait] no checkpoints yet in {ckpt_dir}")
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        processed_any = False
        for pt in done_files:
            base = os.path.basename(pt)
            if base in state:
                continue
            if not os.path.exists(pt + ".done"):
                log(f"[skip] {base}: .done marker missing (still saving)")
                continue
            try:
                ckpt = torch.load(pt, map_location="cpu", weights_only=False)
            except Exception as e:
                log(f"[warn] load {base} failed: {e}")
                continue
            ckpt_args = ckpt.get("args")
            if ckpt_args is None:
                log(f"[warn] {base}: no args in ckpt, skip")
                state[base] = {"error": "no args"}
                continue
            step = int(ckpt.get("train_steps", 0) or 0)
            log(f"[eval] === processing {base} (step {step}) ===")

            cfg = {
                "steps": args.steps if args.steps else int(getattr(ckpt_args, "eval_steps", 50)),
                "cfg": args.cfg if args.cfg else float(getattr(ckpt_args, "eval_cfg", 4.0)),
                "seed": int(getattr(ckpt_args, "eval_seed", 0)),
                "batch": args.batch if args.batch else int(getattr(ckpt_args, "eval_batch", 16)),
                "cond_mode": getattr(ckpt_args, "cond_mode", "2cond"),
                "glyph_init_mix": float(getattr(ckpt_args, "glyph_init_mix", 0.0)),
                "workers": args.workers,
                "worker_threads": args.worker_threads,
            }

            if model is None or model_args is None or model_args != ckpt_args:
                log("[model] building ...")
                if pool is not None:
                    pool_shutdown(pool)
                    pool = None
                model = build_model(ckpt_args, device=args.device)
                model_args = ckpt_args
                log("[model] loading weights ...")
                load_ckpt_weights(model, ckpt, base)
                if vae is None:
                    vae = load_vae(ckpt_args, device=args.device)
                caches = build_caches(ckpt_args, args.seen5_csv)
                if vae is None:
                    log("[vae] VAE unavailable; skipping eval")
                    state[base] = {"error": "vae unavailable"}
                    continue
                # 常驻池：此刻父进程仍处单线程（main 启动时 threads=1），fork 干净，
                # 一次性 fork，之后所有轮次复用，不再 fork。
                if args.workers > 1 and caches[0] is not None \
                        and len(caches[0]["conds"]) >= args.workers:
                    pool = start_pool(model, vae, caches[0], cfg,
                                      args.workers, args.worker_threads)
            else:
                # 同一实验：只换权重，不重建模型
                load_ckpt_weights(model, ckpt, base)

            try:
                res = eval_one(model, vae, args.device, ckpt_dir, step, caches, cfg,
                               pool=pool, ckpt_path=pt, base=base)
                state[base] = {"step": step, "ok": True,
                               "mse": res["mse"], "ssim": res["ssim"],
                               "ts": datetime.datetime.now().isoformat()}
                save_state(ckpt_dir, state)
                processed_any = True
            except Exception as e:
                import traceback
                log(f"[error] eval {base} failed: {e}")
                traceback.print_exc()
                state[base] = {"step": step, "error": str(e)}
                save_state(ckpt_dir, state)

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())