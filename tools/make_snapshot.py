#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把当前监控数据固化成自包含 HTML 快照，方便与后续 run 对比。

从 tools/ 下的 train_data.json + eval_poster.png + remote_gt 生成一个独立的
HTML 文件：数据内嵌为 JS 常量、poster 内嵌为 base64（不依赖 http server）。
命名带时间戳（如 snapshots/monitor_20260816_115000.html）。

用法:
  python make_snapshot.py [-o snapshots/monitor_<ts>.html]
  可选: --dashboard <train_dashboard.html 模板>
"""
import os
import sys
import json
import base64
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "snapshots")
DATA_JSON = os.path.join(HERE, "train_data.json")
POSTER_PNG = os.path.join(HERE, "eval_poster.png")
DASH_TMPL = os.path.join(HERE, "train_dashboard.html")
GT_PREFIX = os.path.join(HERE, "remote_gt")


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def main():
    out = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "-o" and i + 1 < len(args):
            out = args[i + 1]; i += 2
        elif args[i] == "--dashboard" and i + 1 < len(args):
            global DASH_TMPL
            DASH_TMPL = args[i + 1]; i += 2
        else:
            out = args[i]; i += 1

    if not os.path.exists(DASH_TMPL):
        print(f"[snap] 模板缺失: {DASH_TMPL}")
        return 1
    if not os.path.exists(DATA_JSON):
        print(f"[snap] 数据缺失: {DATA_JSON}")
        return 1

    html = open(DASH_TMPL, encoding="utf-8").read()
    data = json.load(open(DATA_JSON, encoding="utf-8"))
    rows = data.get("rows", [])
    src = data.get("source", "?")
    pulled = data.get("pulledAt", "?")

    # 1) 把 fetch('train_data.json') 替换为内嵌数据
    data_js = json.dumps(data, ensure_ascii=False)
    html = html.replace("await fetch('train_data.json?t=' + Date.now());", "")
    html = html.replace("if(!res.ok) throw new Error('HTTP '+res.status);", "")
    html = html.replace("const data = await res.json();", f"const data = {data_js};")

    # 2) Chart.js CDN -> 内嵌本地副本，快照完全离线自包含
    local_chartjs = os.path.join(HERE, "chart.umd.min.js")
    if os.path.exists(local_chartjs):
        chartjs = open(local_chartjs, encoding="utf-8", errors="replace").read()
        html = html.replace(
            '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>',
            "<script>" + chartjs + "</script>")
        html = html.replace(
            '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>' +
            "", "")
        # 上面 replace 可能未命中（引号/路径差异），用通用替换
        if "cdn.jsdelivr.net/npm/chart.js" in html:
            import re
            html = re.sub(r'<script src="https://cdn\.jsdelivr\.net/npm/chart\.js[^"]*"></script>',
                          "<script>" + chartjs + "</script>", html)
    else:
        print("[snap] 警告：本地 chart.umd.min.js 缺失，快照将依赖 CDN")

    # 2) poster 内嵌 base64（用正则替换 loadPoster 函数体）
    import re
    poster_js = "null"
    if os.path.exists(POSTER_PNG):
        poster_js = '"' + b64(POSTER_PNG) + '"'
    new_poster_fn = (
        "async function loadPoster(){\n"
        "  const img = document.getElementById('posterImg');\n"
        f"  if({poster_js}){{ img.src = {poster_js}; }}\n"
        "}"
    )
    html2 = re.sub(r'async function loadPoster\(\)\{.*?\n\}', new_poster_fn, html,
                   flags=re.DOTALL, count=1)
    if html2 == html:
        print("[snap] 警告：loadPoster 正则未匹配，poster 可能未内嵌")
    html = html2

    # 3) 去掉自动刷新/手动刷新（快照是冻结的）
    html = html.replace("document.getElementById('refreshBtn').addEventListener('click', () => {",
                        "document.getElementById('refreshBtn') && document.getElementById('refreshBtn').addEventListener('click', () => {")
    html = html.replace("load(); syncAuto();", "load();")

    # 4) 标题加时间与来源标记
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = html.replace(
        "<title>DiT 训练监控 · Diff Loss</title>",
        f"<title>监控快照 · {ts}</title>", 1)
    html = html.replace('id="meta">未加载', f'id="meta">快照 {ts} · 来源 {src} · 数据拉取 {pulled} · 记录 {len(rows)} 条')
    # 面板标题加"快照"
    html = html.replace("DiT-3Cond 训练监控", "DiT 训练监控 · 快照")

    os.makedirs(OUT_DIR, exist_ok=True)
    if out is None:
        out = os.path.join(OUT_DIR, f"monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[snap] 已固化 -> {out}  ({len(rows)} 条记录)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
