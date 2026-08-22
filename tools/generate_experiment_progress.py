#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为每个实验生成 eval 汇总海报: 用该实验所有已存 eval(跨 run) 拼一张图,
每 ckpt 一行(img|canny|skel), 每行标注 step + MSE/SSIM。所有实验的海报放
tools/experiment_progress/(每实验一个 <exp>.png)。
数据源: 远程 5script/results/<exp>/*/checkpoints/eval_samples + eval_auto_*.json

下载是缓存优先: 本地 _exp_cache/<exp>/ 已有有效数据的实验直接跳过, 不再重新下载。
用 scp 加密钥环境变量(若有)可提高稳定性。
用法:
  python generate_experiment_progress.py --download-only   只下载(海报稍后)
  python generate_experiment_progress.py --posters         只用缓存生成海报
  python generate_experiment_progress.py                   下载+生成海报
  python generate_experiment_progress.py <exp> --download-only 只处理某个实验
"""
import os, sys, glob, json, subprocess, shutil, base64, argparse, time

import tarfile

# 统一 UTF-8 输出, 避免 GBK 控制台对中文/unicode 符号报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "experiment_progress")
CACHE = os.path.join(HERE, "_exp_cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
REMOTE = "root@10.176.54.17"
PORT = "36430"
BASE = "/root/Workspace/xy/DiT"
SSH_KEY = os.environ.get("PP_SSH_KEY")  # 可选: 指向 id_rsa 路径, 提升 scp 稳定性

# 实验列表: (展示名, 目录名)
EXPS = [
    ("S_kailishu_noloss_2factor", "s2_fromscratch_2factor"),
    ("S_kailishu_mid",            "s2_fromscratch_glyphmid"),
    ("S_top30_noloss",            "s5_2factor_top30"),
    ("S_top30_pixel_cs_post50k",  "s5_2factor_struct_post50k"),
    ("B_top30_latentC",           "s5_2factor_B_latentstruct"),
    ("B_top30_latentC_pixelsk_opt","s5_2factor_B_latentstruct_pixelsk_opt"),
    ("B_top30_cs_bf16",           "s5_2factor_B_canny05_pixelsk"),
    ("B_top30_cs_fp32",           "s5_2factor_B_pixelfp32"),
    ("v3b_XL_glyph_kailishu",     "v3b_xl_glyphcond"),
    ("v3c_XL_midstep_kailishu",   "v3c_xl_glyphcond_midstep"),
]


def _sopt():
    return (["-i", SSH_KEY] if SSH_KEY else []) + ["-o", "ConnectTimeout=20"]


def ssh(cmd, timeout=120, tries=3):
    for i in range(tries):
        try:
            r = subprocess.run(["ssh", *_sopt(), "-p", PORT, REMOTE, cmd],
                               capture_output=True, text=True, timeout=timeout)
            out = r.stdout or ""
            if "done" in out or r.returncode == 0:
                return out
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
    return ""


def scp(remote_src, local_dst, timeout=600, tries=4):
    last = None
    for i in range(tries):
        try:
            r = subprocess.run(["scp", *_sopt(), "-P", PORT, remote_src, local_dst],
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return True
            last = r.stderr
        except subprocess.TimeoutExpired:
            last = "timeout"
        time.sleep(5)
    return last


def count_steps(es_root):
    n = 0
    for root, dirs, files in os.walk(es_root):
        if os.path.basename(root).startswith("step") and "samples.json" in files:
            n += 1
    return n


def has_cache(exp):
    """该实验本地是否已有有效数据可跳过下载。"""
    d = os.path.join(CACHE, exp)
    es = os.path.join(d, "eval_samples")
    jd = os.path.join(d, "eval_jsons")
    if not os.path.isdir(es) or count_steps(es) == 0:
        return False
    jsons = (glob.glob(os.path.join(jd, "eval_auto_*.json"))
             if os.path.isdir(jd) else [])
    # 允许部分实验确实没有 eval_auto json(如 v3c 只有 show5); 但只要 json 目录为空且目录本身不新,
    # 都视为需要补一次 json 小拉取。这里对"已有 step"为主判据。
    return count_steps(es) > 0


def download(exp, force=False):
    """把该实验所有 run 的 eval_samples + eval_auto json 拉到本地缓存。
    已有有效缓存且非 force 时直接跳过。"""
    d = os.path.join(CACHE, exp)
    es = os.path.join(d, "eval_samples")
    jd = os.path.join(d, "eval_jsons")
    if not force and has_cache(exp):
        n = count_steps(es)
        print(f"[{exp}] 已缓存 {n} step, 跳过下载 ✔")
        return n

    os.makedirs(d, exist_ok=True)
    os.makedirs(es, exist_ok=True)
    os.makedirs(jd, exist_ok=True)

    # 远程: 把该实验所有 run 的 eval_samples 打包到 /tmp
    tar_cmd = ("cd %s/5script/results/%s && tar czf /tmp/_pp_%s.tgz "
               "*/checkpoints/eval_samples/*/sample*.png "
               "*/checkpoints/eval_samples/*/gt*.png "
               "*/checkpoints/eval_samples/*/samples.json "
               "2>/dev/null; echo done" % (BASE, exp, exp))
    if not ssh(tar_cmd, 180):
        print(f"[{exp}] tar 阶段远程无响应, 跳过"); return 0

    tgz = os.path.join(d, "pp.tgz")
    # scp 拉回, 失败重试(覆盖写)
    res = scp(f"{REMOTE}:/tmp/_pp_{exp}.tgz", tgz)
    if res is not True:
        print(f"[{exp}] scp 拉取失败: {str(res)[-120:]}"); return 0

    try:
        with tarfile.open(tgz) as t:
            members = t.getmembers()
            # 解压到临时子目录再扁平化, 避免脏目录残留
            xroot = os.path.join(d, "_x")
            shutil.rmtree(xroot, ignore_errors=True)
            os.makedirs(xroot)
            t.extractall(xroot)
    except Exception as e:
        print(f"[{exp}] 解压失败 {e}"); return 0

    # 扁平化: 所有含 samples.json 的 step 目录移到 es 根(同名覆盖, 深的先移)
    steps = [r for r, dn, fn in os.walk(xroot)
             if os.path.basename(r).startswith("step") and "samples.json" in fn]
    for sroot in sorted(steps, key=len, reverse=True):
        base = os.path.basename(sroot)
        dst = os.path.join(es, base)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(sroot, dst)
    shutil.rmtree(xroot, ignore_errors=True)
    if os.path.exists(tgz):
        os.remove(tgz)

    # eval_auto json 单独拉(小), 失败不致命
    scp(f"{REMOTE}:{BASE}/5script/results/{exp}/*/checkpoints/eval_auto_*.json",
        os.path.join(jd, "") + ".")

    n = count_steps(es)
    print(f"[{exp}] 拉到 {n} 个 step 目录")
    return n


def gen(exp):
    d = os.path.join(CACHE, exp)
    es = os.path.join(d, "eval_samples")
    jd = os.path.join(d, "eval_jsons")
    if not os.path.isdir(es) or count_steps(es) == 0:
        print(f"[{exp}] 无 eval_samples"); return False
    args = [sys.executable, os.path.join(HERE, "make_eval_poster.py"),
            "--show5-dir", es, "--gt-dir", os.path.join(HERE, "remote_gt"),
            "--show5-csv", os.path.join(HERE, "eval5_top30.csv"),
            "--eval-json-dir", jd, "--exp", exp, "-o", os.path.join(OUT, f"{exp}.png")]
    r = subprocess.run(args, capture_output=True, text=True, timeout=300)
    ok = os.path.exists(os.path.join(OUT, f"{exp}.png"))
    print(f"[{exp}] {'OK' if ok else 'FAIL'} {r.stdout[-80:] if r.stdout else ''}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="?", default=None, help="只处理该实验(目录名)")
    ap.add_argument("--download-only", action="store_true", help="只下载, 不生成海报")
    ap.add_argument("--posters", action="store_true", help="只用缓存生成海报, 不下载")
    ap.add_argument("--refresh", action="store_true",
                    help="强制重新下载(先清空该实验缓存), 再生成海报")
    args = ap.parse_args()

    for name, exp in EXPS:
        if args.exp and args.exp != exp:
            continue
        if args.refresh:
            d = os.path.join(CACHE, exp)
            if os.path.isdir(d):
                shutil.rmtree(d)
        if not args.posters:
            download(exp, force=args.refresh)
        if not args.download_only:
            gen(exp)


if __name__ == "__main__":
    main()
