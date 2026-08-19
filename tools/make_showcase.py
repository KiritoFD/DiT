#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_showcase.py — 展示图统一打包脚本（本地运行）。

职责：
  1. 找远程"最新在跑实验"的 ckpt 目录（5script/results/<exp>/<run>/checkpoints，mtime 最新）
  2. 拉 eval_latest.png（CPU eval 生成的当前 ckpt 快照）-> tools/eval_latest.png
  3. 拉 eval_samples（历史各 ckpt 取样图）-> tools/remote_eval_samples/（覆盖旧 run）
  4. 本地重生成 eval_poster.png（每 ckpt 一行 + GT 行，GT canny/skel 本地即时算）
  5. 归档：把旧 eval_poster/eval_latest 按实验 run 编号归档到 poster_archive/<exp>__<run>/
  6. dashboard 固定读 eval_latest.png + eval_poster.png；历史都入 archive

用法: python make_showcase.py [--show5 eval5_top30.csv] [--archive-dir poster_archive]
"""
import os, sys, glob, subprocess, datetime, json, re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = "root@10.176.54.17"
REMOTE_PORT = "36430"
REMOTE_BASE = "/root/Workspace/xy/DiT"
OUT_JSON = os.path.join(HERE, "train_data.json")
LOCAL_ES = os.path.join(HERE, "remote_eval_samples")
LOCAL_SEEN = os.path.join(HERE, "remote_seen_samples")
POSTER = os.path.join(HERE, "eval_poster.png")
LATEST = os.path.join(HERE, "eval_latest.png")
ARCHIVE = os.path.join(HERE, "poster_archive")


def ssh(cmd, timeout=30):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-p", REMOTE_PORT, REMOTE, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout


def find_latest_ckpt_dir():
    out = ssh("ls -td %s/5script/results/*/*/checkpoints 2>/dev/null | head -1" % REMOTE_BASE, 20)
    return out.strip() if out.strip() else ""


def exp_key_from_ckpt(ckpt_dir):
    """ckpt_dir: .../results/<exp>/<run>/checkpoints -> exp/<run>"""
    parts = ckpt_dir.rstrip("/").split("/")
    return parts[-3] + "__" + parts[-2]


def pull(remote_path, local_path, timeout=120):
    r = subprocess.run(["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT,
                        f"{REMOTE}:{remote_path}", local_path],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0


def archive_current(exp_key):
    """把当前 eval_poster/eval_latest（上一实验的）归档到 archive/<exp_key>/，编号留存。"""
    d = os.path.join(ARCHIVE, exp_key)
    os.makedirs(d, exist_ok=True)
    moved = []
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for src, nm in ((POSTER, "eval_poster"), (LATEST, "eval_latest")):
        if os.path.exists(src):
            dst = os.path.join(d, f"{nm}.{ts}.png")
            shutil.move(src, dst)
            moved.append(nm)
    # 归档当前 remote_eval_samples（旧 run 取样）到该子目录
    if os.path.isdir(LOCAL_ES) and os.listdir(LOCAL_ES):
        shutil.rmtree(os.path.join(d, "eval_samples"), ignore_errors=True)
        shutil.copytree(LOCAL_ES, os.path.join(d, "eval_samples"))
    return moved


def gen_poster(show5_csv, seen5_csv=None, eval_json_dir=None):
    args = [sys.executable, os.path.join(HERE, "make_eval_poster.py"), LOCAL_ES,
            "--gt-dir", os.path.join(HERE, "remote_gt"),
            "--show5-csv", os.path.join(HERE, show5_csv)]
    if seen5_csv and os.path.isdir(LOCAL_SEEN):
        args += ["--seen5-dir", LOCAL_SEEN, "--seen5-csv", os.path.join(HERE, seen5_csv)]
    if eval_json_dir and os.path.isdir(eval_json_dir):
        args += ["--eval-json-dir", eval_json_dir]
    args += ["-o", POSTER]
    subprocess.run(args, capture_output=True, text=True, timeout=180)


STATE_F = os.path.join(HERE, "showcase_state.json")


def _load_state():
    try:
        return json.load(open(STATE_F, encoding="utf-8")) if os.path.exists(STATE_F) else {}
    except Exception:
        return {}


def sync_once(show5_csv="eval5_top30.csv", seen5_csv="seen5_top30.csv", verbose=True):
    """同步一次展示图：仅在实验/run 变化时归档旧图；拉最新 eval/seen 样本 + 生成海报。"""
    ckpt = find_latest_ckpt_dir()
    if not ckpt:
        if verbose: print("[showcase] 未找到远程实验 ckpt 目录")
        return
    exp_key = exp_key_from_ckpt(ckpt)
    st = _load_state()
    moved = []
    if st.get("ckpt") != ckpt and st.get("ckpt"):
        # 实验或 run 变了 -> 归档旧图
        moved = archive_current(exp_key)
    # 拉 eval_latest (最新实验当前)
    ok_latest = pull(f"{ckpt}/eval_latest.png", LATEST, 120)
    # 拉 eval_samples + seen_samples（覆盖合并）
    os.makedirs(LOCAL_ES, exist_ok=True)
    subprocess.run(["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT, "-r",
                    f"{REMOTE}:{ckpt}/eval_samples/.", LOCAL_ES + "/"],
                   capture_output=True, text=True, timeout=300)
    os.makedirs(LOCAL_SEEN, exist_ok=True)
    subprocess.run(["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT, "-r",
                    f"{REMOTE}:{ckpt}/seen_samples/.", LOCAL_SEEN + "/"],
                   capture_output=True, text=True, timeout=300)
    steps = [d for d in os.listdir(LOCAL_ES) if d.startswith("step")]
    seen_steps = [d for d in os.listdir(LOCAL_SEEN) if d.startswith("step")] if os.path.isdir(LOCAL_SEEN) else []
    # 拉 eval_auto_*.json（每行显示 step 的 MSE/SSIM）
    EVJSON = os.path.join(HERE, "remote_eval_jsons")
    os.makedirs(EVJSON, exist_ok=True)
    for f in os.listdir(EVJSON):
        if f.startswith("eval_auto_"): os.remove(os.path.join(EVJSON, f))
    subprocess.run(["scp", "-o", "ConnectTimeout=15", "-P", REMOTE_PORT, "-r",
                    f"{REMOTE}:{ckpt}/eval_auto_*.json", EVJSON + "/"],
                   capture_output=True, text=True, timeout=60)
    # 生成海报（show5|seen5 并列, 每行含 step + MSE/SSIM）
    gen_poster(show5_csv, seen5_csv, EVJSON)
    json.dump({"ckpt": ckpt, "ts": datetime.datetime.now().isoformat()},
              open(STATE_F, "w", encoding="utf-8"))
    if verbose:
        print(f"[showcase] {datetime.datetime.now():%H:%M:%S} exp={exp_key} "
              f"steps={len(steps)} seen={len(seen_steps)} "
              f"latest={'OK' if ok_latest else 'MISS'} poster=OK "
              f"archived={','.join(moved) if moved else '-'}")


def main():
    show5_csv = "eval5_top30.csv"
    seen5_csv = "seen5_top30.csv"
    if "--show5" in sys.argv:
        show5_csv = sys.argv[sys.argv.index("--show5") + 1]
    if "--seen5" in sys.argv:
        seen5_csv = sys.argv[sys.argv.index("--seen5") + 1]
    interval = 120
    if "--interval" in sys.argv:
        try: interval = int(sys.argv[sys.argv.index("--interval") + 1])
        except Exception: pass
    if "--loop" in sys.argv:
        import time
        print(f"[showcase] loop 每 {interval}s 同步展示图")
        while True:
            try: sync_once(show5_csv, seen5_csv)
            except Exception as e: print(f"[showcase] err {e}")
            time.sleep(interval)
    else:
        sync_once(show5_csv, seen5_csv)


if __name__ == "__main__":
    main()

