# -*- coding: utf-8 -*-
"""
build_ctrl_dashboard.py — 为 ControlNet 训练生成自包含静态 HTML dashboard.

基于 train_dashboard.html 模板，适配 ControlNet 日志格式:
  - 训练行只有 loss/LR/Steps/Sec/Mem (无 Total/Diff/Canny/Skel 等分项)
  - eval 有 base/ctrl 两组 MSE/SSIM

流程:
  1. pull_ctrl_monitor.py 从远程拉日志 + eval json → train_data.json
  2. 本脚本把 train_data.json 内联进 HTML → tools/dashboards/ctrl_skel.html

用法:
  python tools/build_ctrl_dashboard.py          # 拉一次 + build
  python tools/build_ctrl_dashboard.py --loop  # 每 30s 循环
"""
import os
import sys
import json
import html

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATE = os.path.join(HERE, "train_dashboard.html")
DASH_DIR = os.path.join(HERE, "dashboards")
DATA_JSON = os.path.join(HERE, "train_data.json")
MONITOR = os.path.join(HERE, "pull_ctrl_monitor.py")


def run_monitor():
    """调用 pull_ctrl_monitor.py 拉一次."""
    import subprocess
    r = subprocess.run([sys.executable, MONITOR], capture_output=True, text=True,
                       timeout=300, encoding="utf-8", errors="replace")
    return r.stdout.strip() if r.returncode == 0 else f"FAIL: {r.stderr[:200]}"


def build():
    """把 train_data.json 内联进 HTML 模板, 生成静态 dashboard."""
    if not os.path.isfile(DATA_JSON):
        print("[build] 无 train_data.json, 先跑 monitor")
        return None
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        t = f.read()

    # 注入内联数据
    data_js = json.dumps(data, ensure_ascii=False)
    t = t.replace("const COLORS = {",
                  f"const __DATA__ = {data_js};\nconst COLORS = {{", 1)

    # 替换 fetch 为内联
    old_fetch = (
        "    const res = await fetch('train_data.json?t='+Date.now());\n"
        "    if(!res.ok) throw new Error('HTTP '+res.status);\n"
        "    const data = await res.json();"
    )
    new_inline = "    const data = __DATA__;"
    if old_fetch in t:
        t = t.replace(old_fetch, new_inline, 1)
    else:
        print("[warn] 未找到 fetch 片段")

    # 去掉自动刷新轮询
    t = t.replace("load();\nsyncAuto();", "load();", 1)

    # 替换 loadPoster 为空（ControlNet 暂无 eval 图）
    old_poster = (
        "async function loadPoster(){\n"
        "  await loadImg('latestImg', ['eval_latest.png?t='+Date.now()]);\n"
        "  await loadImg('posterImg', ['eval_poster.png?t='+Date.now()]);\n"
        "}"
    )
    new_poster = "function loadPoster(){}"
    if old_poster in t:
        t = t.replace(old_poster, new_poster, 1)

    # 标题
    exp = data.get("experiment", "ctrl_skel")
    t = t.replace("<title>DiT 训练监控</title>",
                  f"<title>DiT 监控 · {html.escape(exp)}</title>", 1)

    os.makedirs(DASH_DIR, exist_ok=True)
    out = os.path.join(DASH_DIR, "ctrl_skel.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(t)

    rows = data.get("rows", [])
    last = rows[-1] if rows else {}
    print(f"[build] {out} | step={last.get('step','?')} "
          f"loss={last.get('loss','?')} | rows={len(rows)}")
    return out


def main():
    import time
    interval = 30
    if "--interval" in sys.argv:
        try:
            interval = int(sys.argv[sys.argv.index("--interval") + 1])
        except Exception:
            pass
    if "--loop" in sys.argv:
        print(f"[dashboard] loop 每 {interval}s")
        while True:
            t0 = time.time()
            try:
                run_monitor()
                build()
            except Exception as e:
                print(f"[dashboard] error: {e}")
            dt = max(1, interval - (time.time() - t0))
            time.sleep(dt)
    else:
        run_monitor()
        build()


if __name__ == "__main__":
    main()
