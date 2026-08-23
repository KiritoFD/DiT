#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""auto_eval_cpu.py — 独立 CPU 评测进程（与 train.py 完全解耦）。

架构 (v2 — spawn + 动态任务队列):
  - 使用 ProcessPoolExecutor + spawn（绝不用 fork，杜绝 OpenMP/MKL 死锁）
  - 每个 Worker 独立建 model/VAE/caches，父子进程内存完全隔离（无 CoW 错觉）
  - eval100 拆成小 batch（默认 8），动态分发到 Worker 池，消灭长尾效应
  - show5/seen5/eval_batch 统一为 Task，通过同一个 executor 调度
  - 父进程是纯调度器（0% CPU），所有算力集中在 Worker 池
  - Worker 复用：同一实验的 ckpt 只换权重，不重建模型/缓存

train.py 只训练+保存 ckpt（GPU 只训练）。本进程轮询 ckpt 目录，发现新 checkpoint
（*.pt 且带 .done 标记）即评测：
  1) 指标: eval100 自由采样 DDIM → MSE/SSIM → eval_auto_{step}.json
  2) 展示: show5 → eval_latest.png + eval_samples/stepXXXXXXX/
  3) 展示: seen5 → seen_samples/stepXXXXXXX/

用法:
  python auto_eval_cpu.py --results-dir 5script/results/<exp>
                           --workers 8 --worker-threads 8
                           [--seen5-csv 5script/seen5_top30.csv]
                           [--show5-csv 5script/show5_top30.csv]
                           [--eval-n 100] [--steps 50] [--cfg 4.0] [--batch 8]
                           [--interval 30] [--once]
"""
import argparse
import glob
import json
import os
import sys
import time
import datetime
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_SEEN5_CSV = "5script/seen5_top30.csv"
# eval100 动态分批的默认每批样本数；None=自动按 ceil(eval_n / workers) 计算
EVAL_BATCH_SIZE = None


# ---------------------------------------------------------------------------
# CPU 亲和性: 10 Worker × 6 物理核 = 60 核（留 4 核给 train.py / OS / pull_monitor）
# ---------------------------------------------------------------------------

def _build_core_affinity_table():
    """为每个 Worker 预分配互不重叠的物理核列表。
    双路 EPYC 7542: 64 物理核 (0-63), SMT: cpu N 与 cpu N+64 配对。
    我们只用物理核 0-63（不用超线程 64-127），避免 FPU 抢占。
    10 Worker × 6 核 = 60 核, 剩余 4 核 (60-63) 留给系统+训练。
    """
    PHYSICAL_CORES = list(range(64))  # 物理核 0-63（64-127 是它们的超线程对）
    WORKERS = 10
    CORES_PER_WORKER = 6
    table = []
    for w in range(WORKERS):
        start = w * CORES_PER_WORKER
        end = start + CORES_PER_WORKER
        table.append(PHYSICAL_CORES[start:end])
    return table


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 模型 / VAE 构建（复刻 train.py 的构建与加载顺序，保证架构一致）
# ---------------------------------------------------------------------------

def build_model(args, device="cpu"):
    vae_downscale = getattr(args, "vae_downscale", 8)
    latent_size = args.image_size // vae_downscale
    cond_mode = getattr(args, "cond_mode", "2cond")
    _in_ch = getattr(args, "latent_channels", 4)
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
            in_channels=_in_ch,
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
            in_channels=_in_ch,
        )
    if getattr(args, "pretrained", None):
        from download import find_model
        state_dict = find_model(args.pretrained)
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith(("y_embedder", "y_callig", "y_script",
                                          "y_char", "cond_fusion"))}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log(f"[model] loaded pretrained body {args.pretrained} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})")
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
# Worker 进程：spawn 隔离，全局状态仅存活于子进程生命周期内
# ---------------------------------------------------------------------------

_W = {}  # Worker 全局状态（每个子进程独立）


def _init_worker(ckpt_args_dict, worker_threads, seen5_csv, show5_csv_override,
                 core_table):
    """每个 Worker 启动时仅执行一次（spawn 后）。
    独立建 model/VAE/caches，与父进程内存完全隔离。
    core_table: 全局核表（picklable list of lists），用 /tmp 文件做原子计数分配 worker_id。"""
    import ctypes, tempfile

    # ── 原子获取唯一 worker_id（文件锁方式，spawn 安全）──
    lock_path = "/tmp/_eval_worker_id.lock"
    counter_path = "/tmp/_eval_worker_id.counter"
    with open(lock_path, "w") as lf:
        import fcntl
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            try:
                with open(counter_path, "r") as cf:
                    worker_id = int(cf.read().strip())
            except (IOError, ValueError):
                worker_id = 0
            with open(counter_path, "w") as cf:
                cf.write(str(worker_id + 1))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    # ── 绑核: 用 os.sched_setaffinity 把本进程钉在指定物理核上 ──
    cpu_affinity = []
    if core_table and worker_id < len(core_table):
        cpu_affinity = set(core_table[worker_id])
    else:
        cpu_affinity = set(range(60, 64))  # fallback: 留给系统的核

    try:
        os.sched_setaffinity(0, cpu_affinity)
        log(f"[worker {worker_id} pid={os.getpid()}] "
            f"pinned to cores {sorted(cpu_affinity)}")
    except Exception as e:
        log(f"[worker {worker_id} pid={os.getpid()}] "
            f"affinity error: {e}")

    # ── OpenMP 线程绑核: 限制在已绑核范围内 ──
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"

    torch.set_num_threads(worker_threads)
    from argparse import Namespace
    a = Namespace(**ckpt_args_dict)
    if show5_csv_override:
        a.show5_csv = show5_csv_override
    _W["model"] = build_model(a, "cpu")
    _W["vae"] = load_vae(a, "cpu")
    _W["caches"] = build_caches(a, seen5_csv)
    _W["current_ckpt"] = None
    _W["cfg"] = {
        "steps": int(getattr(a, "eval_steps", 50)),
        "cfg": float(getattr(a, "eval_cfg", 4.0)),
        "seed": int(getattr(a, "eval_seed", 0)),
        "batch": int(getattr(a, "eval_batch", 16)),
        "cond_mode": getattr(a, "cond_mode", "2cond"),
        "glyph_init_mix": float(getattr(a, "glyph_init_mix", 0.0)),
        "latent_channels": int(getattr(a, "latent_channels", 4)),
        "latent_spatial": int(getattr(a, "image_size", 256)) // int(getattr(a, "vae_downscale", 8)),
        "scaling_factor": float(getattr(a, "vae_scaling_factor", 0.18215)),
    }
    log(f"[worker {worker_id} pid={os.getpid()}] initialized "
        f"(threads={worker_threads}, cores={len(cpu_affinity)})")


def _run_task(task):
    """执行单个任务。Worker 已有 model/VAE/caches，这里只换权重+跑推理。
    task 是一个 dict，包含所有必要参数。"""
    from eval_auto import eval_gen_in_memory
    ckpt_path = task["ckpt_path"]
    ckpt_name = task["ckpt_name"]

    # 换权重（仅当 ckpt 变化时）
    if ckpt_name != _W["current_ckpt"]:
        # 注意：PyTorch 1.13 不支持 mmap=True。但 OS Page Cache 天然共享：
        # 251GB RAM + 670MB ckpt → 第一个 worker 读完后文件已驻留 page cache，
        # 后续 15 个 worker 直接从内存读，无磁盘 I/O
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_ckpt_weights(_W["model"], ckpt, ckpt_name)
        _W["current_ckpt"] = ckpt_name
        del ckpt

    cfg = _W["cfg"]
    ttype = task["type"]
    result = {"type": ttype, "worker": os.getpid()}

    if ttype == "eval_batch":
        cache = _W["caches"][0]  # eval_cache
        s, e = task["start"], task["end"]
        sub = {
            "conds": cache["conds"][s:e],
            "gts": cache["gts"][s:e],
            "gs": cache["gs"][s:e] if cache.get("gs") is not None else None,
        }
        mse, ssim = eval_gen_in_memory(
            _W["model"], _W["vae"], "cpu", sub,
            n=e - s, steps=cfg["steps"], cfg=cfg["cfg"], seed=cfg["seed"],
            batch=min(e - s, cfg["batch"]), vis_out=None, vis_n=0,
            cond_mode=cfg["cond_mode"], save_samples_dir=None, step=None,
            glyph_init_mix=cfg["glyph_init_mix"],
            latent_channels=cfg["latent_channels"],
            latent_spatial=cfg["latent_spatial"],
            scaling_factor=cfg["scaling_factor"])
        result.update(mse=mse, ssim=ssim, count=e - s, start=s, end=e)

    elif ttype == "show5":
        cache = _W["caches"][1]  # show5_cache
        n = len(cache["conds"])
        eval_gen_in_memory(
            _W["model"], _W["vae"], "cpu", cache,
            n=n, steps=cfg["steps"], cfg=cfg["cfg"], seed=cfg["seed"],
            batch=min(n, 16),
            vis_out=task.get("vis_out"), vis_n=n,
            cond_mode=cfg["cond_mode"],
            save_samples_dir=task.get("save_dir"), step=task.get("step"),
            glyph_init_mix=cfg["glyph_init_mix"],
            latent_channels=cfg["latent_channels"],
            latent_spatial=cfg["latent_spatial"],
            scaling_factor=cfg["scaling_factor"])
        result.update(status="ok")

    elif ttype == "seen5":
        cache = _W["caches"][2]  # seen5_cache
        n = len(cache["conds"])
        eval_gen_in_memory(
            _W["model"], _W["vae"], "cpu", cache,
            n=n, steps=cfg["steps"], cfg=cfg["cfg"], seed=cfg["seed"],
            batch=min(n, 16),
            vis_out=None, vis_n=n,
            cond_mode=cfg["cond_mode"],
            save_samples_dir=task.get("save_dir"), step=task.get("step"),
            glyph_init_mix=cfg["glyph_init_mix"],
            latent_channels=cfg["latent_channels"],
            latent_spatial=cfg["latent_spatial"],
            scaling_factor=cfg["scaling_factor"])
        result.update(status="ok")

    return result


# ---------------------------------------------------------------------------
# 单 ckpt 评测：纯调度逻辑
# ---------------------------------------------------------------------------

def eval_one_ckpt(executor, ckpt_path, ckpt_name, step, ckpt_dir,
                  workers, eval_n, batch_sz=None):
    """提交一个 ckpt 的全部任务到 executor，等待结果，汇总指标。
    batch_sz: eval100 动态分批大小。None 时按 ceil(eval_n / workers) 自动计算，
    确保任务数 ≈ Worker 数，一波流带走（消灭长尾效应）。"""
    t0 = time.time()
    n_eval = eval_n
    if batch_sz is None or batch_sz <= 0:
        # 自动: 让 batch 数 ≈ worker 数，每个 worker 恰好分到 ~1 批
        batch_sz = max(1, (n_eval + workers - 1) // workers)
    eval_ranges = [(i, min(i + batch_sz, n_eval))
                   for i in range(0, n_eval, batch_sz)]

    futures = []

    # 1) show5（1 个任务）
    show5_task = {
        "type": "show5", "ckpt_path": ckpt_path, "ckpt_name": ckpt_name,
        "vis_out": f"{ckpt_dir}/eval_latest.png",
        "save_dir": f"{ckpt_dir}/eval_samples", "step": step,
    }
    futures.append(("show5", executor.submit(_run_task, show5_task)))

    # 2) seen5（1 个任务）
    seen5_task = {
        "type": "seen5", "ckpt_path": ckpt_path, "ckpt_name": ckpt_name,
        "save_dir": f"{ckpt_dir}/seen_samples", "step": step,
    }
    futures.append(("seen5", executor.submit(_run_task, seen5_task)))

    # 3) eval100（动态分批，约 13 个任务）
    for s, e in eval_ranges:
        task = {
            "type": "eval_batch", "ckpt_path": ckpt_path, "ckpt_name": ckpt_name,
            "start": s, "end": e,
        }
        futures.append(("eval", executor.submit(_run_task, task)))

    # 收集结果（动态完成，先完成先收集）
    total_mse = total_ssim = total_cnt = 0.0
    show5_ok = seen5_ok = False
    errors = []

    for label, future in futures:
        try:
            res = future.result(timeout=1800)
            if res["type"] == "eval_batch":
                total_mse += res["mse"] * res["count"]
                total_ssim += res["ssim"] * res["count"]
                total_cnt += res["count"]
            elif res["type"] == "show5":
                show5_ok = True
            elif res["type"] == "seen5":
                seen5_ok = True
        except Exception as e:
            errors.append(f"{label}: {e}")

    if errors:
        log(f"[eval] step {step}: {len(errors)} errors: {errors[:3]}")

    if total_cnt == 0:
        log(f"[eval] step {step}: eval100 全部失败")
        return None, None

    mse = total_mse / total_cnt
    ssim = total_ssim / total_cnt
    elapsed = time.time() - t0
    log(f"[eval] step {step}: MSE={mse:.5f} SSIM={ssim:.4f} "
        f"({elapsed:.0f}s, show5={'ok' if show5_ok else 'FAIL'}, "
        f"seen5={'ok' if seen5_ok else 'FAIL'})")

    # 写指标 JSON
    result = {"step": step, "mse": mse, "ssim": ssim}
    with open(f"{ckpt_dir}/eval_auto_{step:07d}.json", "w") as f:
        json.dump(result, f)

    return mse, ssim


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


def _ckpt_args_to_dict(ckpt_args):
    """把 argparse.Namespace 转成 dict（spawn pickling 友好）。"""
    return dict(vars(ckpt_args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--workers", type=int, default=10,
                    help="Worker 进程数（默认 10）")
    ap.add_argument("--worker-threads", type=int, default=6,
                    help="每 Worker 的 torch 线程数（默认 6，10×6=60 物理核）")
    ap.add_argument("--cores-per-worker", type=int, default=6,
                    help="每 Worker 绑定物理核数（默认 6，10×6=60，留 4 核给系统）")
    ap.add_argument("--no-affinity", action="store_true",
                    help="禁用 CPU 绑核（调试用）")
    ap.add_argument("--seen5-csv", default=DEFAULT_SEEN5_CSV)
    ap.add_argument("--show5-csv", default=None)
    ap.add_argument("--eval-n", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None,
                    help="eval100 动态分批大小（None=自动 ceil(eval_n/workers)）")
    args = ap.parse_args()

    if not args.ckpt_dir and not args.results_dir:
        ap.error("must provide --results-dir or --ckpt-dir")

    # 父进程是纯调度器，不需要任何计算线程。设 1 线程避免 OpenMP 干扰。
    torch.set_num_threads(1)
    log(f"[main] workers={args.workers}, worker_threads={args.worker_threads}, "
        f"cores_per_worker={args.cores_per_worker}, "
        f"batch={args.batch or 'auto'}, "
        f"affinity={'OFF' if args.no_affinity else 'ON'}")

    # 预分配 CPU 核心表
    core_table = None
    if not args.no_affinity:
        PHYSICAL_CORES = list(range(64))  # 双路 EPYC 7542 物理核 0-63
        n_w = args.workers
        cpw = args.cores_per_worker
        core_table = []
        for w in range(n_w):
            start = w * cpw
            end = min(start + cpw, 64)
            core_table.append(PHYSICAL_CORES[start:end])
        log(f"[main] core affinity table: " +
            " ".join(f"W{i}={core_table[i]}" for i in range(n_w)))

    results_dir = None
    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
        os.makedirs(results_dir, exist_ok=True)

    executor = None
    last_ckpt_dir = None
    ckpt_args_dict = None

    while True:
        ckpt_dir = args.ckpt_dir or read_active_ckpt_dir(results_dir)
        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            log(f"[wait] no active ckpt dir yet ({results_dir}) ...")
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        # 实验切换：销毁旧 pool，重建
        if ckpt_dir != last_ckpt_dir:
            log(f"[watch] active ckpt dir: {ckpt_dir}")
            if executor is not None:
                executor.shutdown(wait=False)
                executor = None
            last_ckpt_dir = ckpt_dir
            state = load_state(ckpt_dir)
            log(f"[watch] loaded state: {len(state)} entries")

        done_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        pending = [pt for pt in done_files
                   if os.path.basename(pt) not in state
                   and os.path.exists(pt + ".done")]
        if not pending:
            # 没有待评测的 ckpt，静默等待（只打一次日志，不刷屏）
            if not hasattr(main, '_last_scan_log') or time.time() - main._last_scan_log > 300:
                log(f"[scan] {len(done_files)} ckpts, {len(state)} done, 0 pending — waiting for new ckpt")
                main._last_scan_log = time.time()
            if args.once:
                return 0
            time.sleep(args.interval)
            continue
        log(f"[scan] {len(done_files)} ckpts total, {len(state)} done, {len(pending)} pending")

        for pt in pending:
            base = os.path.basename(pt)

            log(f"[scan] loading {base} ({os.path.getsize(pt)/1e6:.0f}MB)...")
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

            # show5_csv 覆盖
            if args.show5_csv:
                ckpt_args.show5_csv = args.show5_csv

            # 首次或实验切换：建 Worker 池
            if executor is None:
                ckpt_args_dict = _ckpt_args_to_dict(ckpt_args)
                log(f"[pool] creating ProcessPoolExecutor: "
                    f"{args.workers} workers (spawn, "
                    f"affinity={'ON' if core_table else 'OFF'}), "
                    f"args_dict={len(ckpt_args_dict)} keys")
                ctx = mp.get_context("spawn")

                # 重置 worker_id 计数器（文件锁方式，spawn 安全）
                try:
                    os.remove("/tmp/_eval_worker_id.counter")
                except OSError:
                    pass

                executor = ProcessPoolExecutor(
                    max_workers=args.workers,
                    mp_context=ctx,
                    initializer=_init_worker,
                    initargs=(ckpt_args_dict, args.worker_threads,
                              args.seen5_csv, args.show5_csv,
                              core_table),
                )
                log(f"[pool] ProcessPoolExecutor created, waiting for workers...")

            # 评测
            eval_n = int(getattr(ckpt_args, "eval_n", 100))
            if args.eval_n:
                eval_n = args.eval_n

            try:
                mse, ssim = eval_one_ckpt(
                    executor, pt, base, step, ckpt_dir,
                    args.workers, eval_n, batch_sz=args.batch)
                if mse is not None:
                    state[base] = {"step": step, "ok": True,
                                   "mse": mse, "ssim": ssim,
                                   "ts": datetime.datetime.now().isoformat()}
                else:
                    state[base] = {"step": step, "error": "eval failed"}
                save_state(ckpt_dir, state)
            except Exception as e:
                log(f"[error] eval {base} failed: {e}")
                traceback.print_exc()
                state[base] = {"step": step, "error": str(e)}
                save_state(ckpt_dir, state)

            del ckpt
            # 实时更新 state，让下一轮 scan 只看到未完成的
            state = load_state(ckpt_dir)

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
