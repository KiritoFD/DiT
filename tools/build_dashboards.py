#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_dashboards.py —— 为每个实验生成一个【自包含静态 html】到 tools/dashboards/。

每个 html:
  - 内联该实验的 train_data.json（无需 http server，双击即可看）
  - 直接引用本地 eval 图（../remote_pulled/<exp>/checkpoints/eval_latest.png）
  - 无自动刷新轮询（静态快照）

同时生成 dashboards/index.html 汇总所有实验的链接与最新 step/时间。

依赖: pull_all.py 先把数据拉到 tools/remote_pulled/<exp>/train_data.json

用法:
    python build_dashboards.py
    python build_dashboards.py --exp s8_structv2_b8all
"""
import os
import sys
import json
import glob
import html

HERE = os.path.dirname(os.path.abspath(__file__))
PULLED_ROOT = os.path.join(HERE, "remote_pulled")
DASH_ROOT = os.path.join(HERE, "dashboards")
TEMPLATE = os.path.join(HERE, "train_dashboard.html")


def load_template():
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        return f.read()


def build_one(exp, template):
    data_path = os.path.join(PULLED_ROOT, exp, "train_data.json")
    if not os.path.isfile(data_path):
        print(f"  [skip] 无数据: {exp} (先跑 pull_all.py)")
        return None
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 图片相对路径（dashboards/<exp>.html -> ../remote_pulled/<exp>/...）
    # 注意: scp 落地后文件直接在 <exp>/ 下（不含 checkpoints/ 层级）
    rel_eval = f"../remote_pulled/{exp}/eval_latest.png"
    rel_poster = f"../remote_pulled/{exp}/eval_poster.png"

    # 注入内联数据
    data_js = json.dumps(data, ensure_ascii=False)
    t = template.replace(
        "const COLORS = {",
        f"const __DATA__ = {data_js};\nconst COLORS = {{",
        1,
    )

    # 替换 load() 的 fetch 为内联数据
    old_fetch = (
        "    const res = await fetch('train_data.json?t='+Date.now());\n"
        "    if(!res.ok) throw new Error('HTTP '+res.status);\n"
        "    const data = await res.json();"
    )
    new_inline = "    const data = __DATA__;"
    if old_fetch in t:
        t = t.replace(old_fetch, new_inline, 1)
    else:
        print(f"  [warn] {exp}: 未找到 fetch 片段，模板可能已变")

    # 替换 loadPoster 为直接设 src（file:// 下 fetch 受限）
    old_poster = (
        "async function loadPoster(){\n"
        "  await loadImg('latestImg', ['eval_latest.png?t='+Date.now()]);\n"
        "  await loadImg('posterImg', ['eval_poster.png?t='+Date.now()]);\n"
        "}"
    )
    new_poster = (
        "function loadPoster(){\n"
        "  const li=document.getElementById('latestImg');\n"
        f"  if(li){{li.src='{rel_eval}';li.onerror=()=>{{li.style.display='none';}};}}\n"
        "  const pi=document.getElementById('posterImg');\n"
        f"  if(pi){{pi.src='{rel_poster}';pi.onerror=()=>{{pi.style.display='none';}};}}\n"
        "}"
    )
    if old_poster in t:
        t = t.replace(old_poster, new_poster, 1)
    else:
        print(f"  [warn] {exp}: 未找到 loadPoster 片段")

    # 去掉自动刷新轮询：把末尾 load(); syncAuto(); 改为仅 load()
    t = t.replace("load();\nsyncAuto();", "load();", 1)

    # 标题带上实验名
    t = t.replace("<title>DiT 训练监控</title>",
                  f"<title>DiT 监控 · {html.escape(exp)}</title>", 1)

    os.makedirs(DASH_ROOT, exist_ok=True)
    out = os.path.join(DASH_ROOT, f"{exp}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(t)
    return out


def build_index(exps_info):
    rows = []
    for exp, info in sorted(exps_info.items()):
        last = info.get("last_step")
        pulled = info.get("pulled_at", "")
        rows.append(
            f'    <tr><td><a href="{html.escape(exp)}.html">{html.escape(exp)}</a></td>'
            f'<td>{last if last is not None else "-" }</td>'
            f'<td>{html.escape(str(pulled))}</td></tr>'
        )
    body = "\n".join(rows)
    idx = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>DiT 实验总览</title>
<style>
body{{background:#0f1117;color:#e6e9ef;font-family:-apple-system,Segoe UI,Roboto,sans-serif;padding:24px}}
h1{{font-size:18px}} table{{border-collapse:collapse;width:100%;margin-top:16px}}
th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid #2a2f3c;font-size:13px}}
a{{color:#4f9cff;text-decoration:none}} a:hover{{text-decoration:underline}}
th{{color:#8b93a7;font-weight:600}}.meta{{color:#8b93a7;font-size:12px}}
</style></head><body>
<h1>DiT 训练实验总览</h1>
<div class="meta">共 {len(exps_info)} 个实验 · 数据来自远程静态快照（双击 html 即可离线查看）</div>
<table>
<thead><tr><th>实验</th><th>最新 step</th><th>拉取时间</th></tr></thead>
<tbody>
{body}
</tbody></table>
</body></html>"""
    os.makedirs(DASH_ROOT, exist_ok=True)
    with open(os.path.join(DASH_ROOT, "index.html"), "w", encoding="utf-8") as f:
        f.write(idx)


MANIFEST = os.path.join(HERE, "active_exps.json")


def load_manifest():
    if not os.path.isfile(MANIFEST):
        return []
    try:
        return json.load(open(MANIFEST, encoding="utf-8")).get("active", [])
    except Exception:
        return []


def main():
    only_exp = sys.argv[sys.argv.index("--exp") + 1] if "--exp" in sys.argv else None
    template = load_template()
    all_exps = [os.path.basename(p) for p in glob.glob(os.path.join(PULLED_ROOT, "*"))]
    if "--all" in sys.argv:
        exps = all_exps
    elif only_exp:
        exps = [only_exp]
    else:
        active = load_manifest()
        exps = [e for e in all_exps if e in set(active)] if active else all_exps
        if not active:
            print("清单为空，build 全部已拉取实验。设置 active: python pull_all.py --set-active <exp>")
    if only_exp:
        exps = [e for e in exps if e == only_exp]
    print(f"生成 {len(exps)} 个静态 dashboard 到 {DASH_ROOT}")
    exps_info = {}
    for exp in sorted(exps):
        out = build_one(exp, template)
        if out:
            dp = os.path.join(PULLED_ROOT, exp, "train_data.json")
            with open(dp, "r", encoding="utf-8") as f:
                d = json.load(f)
            rows = d.get("rows", [])
            exps_info[exp] = {
                "last_step": rows[-1]["step"] if rows else None,
                "pulled_at": d.get("pulledAt", ""),
            }
            print(f"  [ok] {exp} -> {out}")
    build_index(exps_info)
    print(f"  [ok] 总览 -> {os.path.join(DASH_ROOT, 'index.html')}")


if __name__ == "__main__":
    main()
