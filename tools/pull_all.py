#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pull_all.py —— 把远程【所有实验】的日志与 eval 产物批量拉到本地，并解析成
每个实验一个 train_data.json，放到 tools/remote_pulled/<exp>/ 下。

每个 exp 只保留【最新一次 run】（按 log.txt 的 mtime 选择），避免 53 个 run
重复拉取、体积爆炸。

之后由 build_dashboards.py 为每个 exp 生成自包含的静态 html。

用法:
    python pull_all.py            # 拉全部实验（增量：仅 scp 不存在/更新的文件）
    python pull_all.py --exp s8_structv2_b8all   # 只拉某个实验
    python pull_all.py --list     # 只打印远程实验清单，不拉取
"""
import os
import sys
import json
import glob
import time
import subprocess

REMOTE_USER = "root"
REMOTE_HOST = "10.176.54.17"
REMOTE_PORT = "36430"
REMOTE_BASE = "/root/Workspace/xy/DiT"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "remote_pulled")
MANIFEST = os.path.join(HERE, "active_exps.json")


def load_manifest():
    """读取要同步的实验清单。格式: {"active": ["exp1", "exp2"], "updatedAt": "..."}
    若文件不存在，返回空列表（即不默认全量扫描）。"""
    if not os.path.isfile(MANIFEST):
        return []
    try:
        d = json.load(open(MANIFEST, encoding="utf-8"))
        return d.get("active", [])
    except Exception:
        return []


def save_manifest(active):
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"active": active, "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False, indent=2)

# 复用 pull_log 的解析正则与工具函数（避免重复实现）
from pull_log import LINE_RE, AUTOEVAL_RE, to_num

SSH = ["ssh", "-o", "ConnectTimeout=15", "-p", REMOTE_PORT,
       f"{REMOTE_USER}@{REMOTE_HOST}"]


def ssh_exec(cmd, timeout=120):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh 失败({r.returncode}): {r.stderr.strip()[:300]}")
    return r.stdout


def list_remote_runs():
    """返回 list[(exp, run_dir, mtime)]。run_dir 为远程绝对路径（不含 log.txt）。"""
    cmd = (
        "runs=$(find {base}/5script/results -name log.txt 2>/dev/null | sed 's#/log.txt##'); "
        "for d in $runs; do "
        "  exp=$(basename $(dirname \"$d\")); "
        "  echo \"RUN|$exp|$d|$(stat -c %Y \"$d/log.txt\" 2>/dev/null || echo 0)\"; "
        "done; "
        "for f in {base}/exp_*.log; do "
        "  [ -f \"$f\" ] && echo \"TOP|$(basename \"$f\" .log)|$f|$(stat -c %Y \"$f\")\"; "
        "done"
    ).format(base=REMOTE_BASE)
    out = ssh_exec(cmd)
    entries = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        kind, exp, path, mtime = parts[0], parts[1], parts[2], int(parts[3])
        # 日志文件路径
        log = os.path.join(path, "log.txt") if kind == "RUN" else path
        entries.append((exp, path, mtime, log))
    return entries


def pick_latest_per_exp(entries):
    """每个 exp 只保留 mtime 最大的 run。"""
    best = {}
    for exp, path, mtime, log in entries:
        if exp not in best or mtime > best[exp][2]:
            best[exp] = (exp, path, mtime, log)
    return list(best.values())


def scp_files(remote_dir, local_dir, files):
    """把远程目录下指定的文件/目录 scp 到本地目录。

    files 中支持远程 glob（如 'checkpoints/eval_auto_*.json'），
    通过先 ssh 展开再逐个 scp，避免 scp 多 glob 源时整体失败。
    """
    os.makedirs(local_dir, exist_ok=True)
    expanded = []
    for f in files:
        if "*" in f:
            try:
                out = ssh_exec(f"ls -d {remote_dir}/{f} 2>/dev/null")
                expanded += [p for p in out.splitlines() if p.strip()]
            except Exception:
                pass
        else:
            expanded.append(os.path.join(remote_dir, f))
    ok = True
    for src in expanded:
        dst = local_dir + "/"
        cmd = ["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT,
               f"{REMOTE_USER}@{REMOTE_HOST}:{src}", dst]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            ok = False
    return ok


def scp_single(remote_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    cmd = ["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT,
           f"{REMOTE_USER}@{REMOTE_HOST}:{remote_path}", local_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def parse_log_file(log_path):
    """解析单个本地 log 文件为 rows（兼容新旧日志格式）。"""
    rows = []
    if not os.path.isfile(log_path):
        return rows
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "total": to_num(m.group(2)),
                    "diff": to_num(m.group(3)),
                    "canny": to_num(m.group(4)),
                    "skel": to_num(m.group(5)),
                    "latc": to_num(m.group(6)),
                    "lats": to_num(m.group(7)),
                    "repa": to_num(m.group(8)),
                    "skelh": to_num(m.group(9)),
                    "stdmid": to_num(m.group(10)),
                    "x0lat": to_num(m.group(11)),
                    "stepsPerSec": to_num(m.group(12)),
                    "memCur": to_num(m.group(13)),
                    "memPeak": to_num(m.group(14)),
                    "mse": None, "ssim": None, "ts": None,
                })
                continue
            a = AUTOEVAL_RE.search(line)
            if a:
                rows.append({
                    "step": int(a.group(1)),
                    "total": None, "diff": None, "canny": None, "skel": None,
                    "latc": None, "lats": None, "repa": None, "skelh": None,
                    "stdmid": None, "x0lat": None, "stepsPerSec": None,
                    "memCur": None, "memPeak": None,
                    "mse": to_num(a.group(2)), "ssim": to_num(a.group(3)),
                    "ts": None,
                })
                continue
    return rows


def write_json(exp, source, rows):
    """去重 + 排序 + 写出 train_data.json。"""
    seen = {}
    for r in rows:
        seen[r["step"]] = r
    rows = [seen[k] for k in sorted(seen.keys())]
    out_json = os.path.join(OUT_ROOT, exp, "train_data.json")
    last = rows[-1] if rows else None
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "expName": exp,
            "source": source,
            "pulledAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(rows),
            "rows": rows,
        }, f, ensure_ascii=False)
    info = f"step={last['step']}" if last else "0 条"
    print(f"  [ok] {exp}: {len(rows)} 条 ({info}) -> {out_json}")


def merge_eval_jsons(rows, local_dir):
    """用本地 checkpoints/eval_auto_*.json 覆盖对应 step 的 mse/ssim。"""
    mapping = {r["step"]: r for r in rows}
    any_change = False
    for f in sorted(glob.glob(os.path.join(local_dir, "eval_auto_*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if "step" not in d:
            continue
        st = int(d["step"])
        if st in mapping:
            mapping[st]["mse"] = to_num(d.get("mse"))
            mapping[st]["ssim"] = to_num(d.get("ssim"))
            any_change = True
    return rows, any_change


def pull_one(exp, run_dir, log_remote, force=False, local_only=False):
    local_dir = os.path.join(OUT_ROOT, exp)
    local_log = os.path.join(local_dir, "log.txt")
    if local_only:
        # 数据已在本地（由 _pack_remote.py 全量打包而来），跳过 scp
        if not os.path.isfile(local_log):
            print(f"  [skip] 本地无 log.txt: {exp}")
            return
        rows = parse_log_file(local_log)
        rows, _ = merge_eval_jsons(rows, local_dir)
        write_json(exp, "local:" + run_dir, rows)
        return
    # 增量：用【文件字节数】判断（而非 mtime，因为 scp 后本地 mtime 会晚于远程，
    # 导致 mtime 比较永远认为已最新、漏掉持续写入的新日志）。
    need_pull = force
    if not need_pull and os.path.isfile(local_log):
        try:
            rsz = int(ssh_exec(f"stat -c %s '{log_remote}'"))
        except Exception:
            rsz = -1  # ssh 失败：无法比较，保守拉取
        lsz = os.path.getsize(local_log)
        # 远程更大（有新增日志）则拉；ssh 失败(rsz=-1)也拉，避免漏更新
        need_pull = (rsz == -1) or (rsz > lsz)
    else:
        need_pull = True

    if need_pull:
        ok = scp_files(run_dir, local_dir, [
            "log.txt", "resolved_config.json",
            "checkpoints/eval_latest.png", "checkpoints/eval_auto_*.json",
        ])
        if not ok:
            print(f"  [warn] scp 部分失败: {exp}")
    else:
        print(f"  [skip] 已是最新: {exp}")

    rows = parse_log_file(local_log)
    rows, _ = merge_eval_jsons(rows, local_dir)
    write_json(exp, "remote:" + run_dir, rows)
    return


def main():
    only_exp = None
    force = False
    if "--local" in sys.argv:
        # 数据已在本地（由 _pack_remote.py 全量打包），仅本地解析，不连远程
        exps = [os.path.basename(p) for p in glob.glob(os.path.join(OUT_ROOT, "*"))]
        if only_exp:
            exps = [e for e in exps if e == only_exp]
        print(f"[local] 解析 {len(exps)} 个本地实验 ...")
        for exp in sorted(exps):
            print(f"== {exp}")
            try:
                pull_one(exp, os.path.join(OUT_ROOT, exp), "", local_only=True)
            except Exception as e:
                print(f"  [error] {exp}: {e}")
        print("完成。下一步运行: python build_dashboards.py")
        return
    if "--list" in sys.argv:
        entries = list_remote_runs()
        print(f"远程共 {len(entries)} 个 run，按 exp 合并为 {len(pick_latest_per_exp(entries))} 个实验：")
        for exp, path, mt, log in sorted(pick_latest_per_exp(entries)):
            print(f"  {exp:40s}  {path}")
        return
    if "--set-active" in sys.argv:
        # 设置清单: python pull_all.py --set-active exp1 exp2 ...
        idx = sys.argv.index("--set-active") + 1
        active = sys.argv[idx:]
        save_manifest(active)
        print(f"已设置 active 清单: {active}")
        return
    if "--exp" in sys.argv:
        only_exp = sys.argv[sys.argv.index("--exp") + 1]
    if "--force" in sys.argv:
        force = True

    # 默认只同步 manifest 里的 active 实验；--all 才全量扫描远程
    if "--all" in sys.argv:
        entries = list_remote_runs()
        chosen = pick_latest_per_exp(entries)
        if only_exp:
            chosen = [c for c in chosen if c[0] == only_exp]
            if not chosen:
                print(f"未找到实验: {only_exp}")
                return
    else:
        active = load_manifest() if not only_exp else [only_exp]
        if not active:
            print("清单为空（tools/active_exps.json 未设置）。")
            print("  - 设置: python pull_all.py --set-active s8_structv2_b8all")
            print("  - 或全量: python pull_all.py --all")
            return
        # 只解析 active 实验的远程 run（按 exp 名匹配最新 run）
        entries = list_remote_runs()
        all_exp = pick_latest_per_exp(entries)
        chosen = [c for c in all_exp if c[0] in set(active)]
        missing = set(active) - {c[0] for c in chosen}
        if missing:
            print(f"  [warn] 清单里有但远程未找到: {sorted(missing)}")
        if only_exp:
            chosen = [c for c in chosen if c[0] == only_exp]

    print(f"将拉取 {len(chosen)} 个实验到 {OUT_ROOT}")
    for exp, run_dir, mt, log in sorted(chosen):
        print(f"== {exp}")
        try:
            pull_one(exp, run_dir, log, force=force)
        except Exception as e:
            print(f"  [error] {exp}: {e}")
    print("完成。下一步运行: python build_dashboards.py")


if __name__ == "__main__":
    main()
