#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s11 三实验 (s11_top6_p4 / s11_top6_p2 / s11_top6_p8) 串行训练的本地监控 loop。

复用同目录 tools/evaluation/pull_monitor.py 的底层函数 (_ssh/_scp_dir/parse/
merge_eval_json/_pull_sample_dir/_existing_steps) 和 make_eval_poster.py,
参数化为多实验:

  每个实验独立: log 收集 -> train_data_{exp}.json -> 拉 show/seen 样本图
                -> 生成 poster_{exp}.png -> 生成 dashboards/{exp}.html
  外加:   一个三实验对比 dashboard: dashboards/s11_compare.html
          (MSE / SSIM / SkelIoU / LPIPS 四条折线, 每图 3 条曲线)

用法:
  python pull_s11.py                # 拉一轮 (所有实验)
  python pull_s11.py --loop         # 每 --interval 秒循环 (默认 120)
  python pull_s11.py --no-poster    # 跳过样本图拉取+海报 (只更新曲线数据)
"""
import os, re, sys, json, time, glob, subprocess, datetime, shutil, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DASH_DIR = os.path.join(HERE, "dashboards")

# 复用 pull_monitor 的底层函数 (不触发其 __main__)
import pull_monitor as pm  # noqa: E402

# ── monkey-patch: 健壮的 _ssh (errors='replace' 防非 UTF-8 字节解码崩溃;
#    带重试逻辑应对间歇性 SSH banner timeout / connection closed) ───────────
def _ssh_robust(remote_cmd, timeout=30):
    last_err = None
    for attempt in range(3):
        cmd = ["ssh", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=5",
               "-p", pm.REMOTE_PORT,
               f"{pm.REMOTE_USER}@{pm.REMOTE_HOST}", remote_cmd]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout.decode("utf-8", errors="replace")
            last_err = f"rc={r.returncode}"
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except Exception as e:
            last_err = str(e)
        if attempt < 2:
            time.sleep(3 * (attempt + 1))
    if last_err:
        print(f"[ssh] failed after 3 retries: {last_err}", file=sys.stderr)
    return ""

pm._ssh = _ssh_robust  # 让所有复用函数也走健壮版

# 健壮的 _scp_dir: 原版 text=True 遇到非 UTF-8 stderr 会 UnicodeDecodeError 崩溃。
def _scp_dir_robust(remote_path, local_dir, timeout=120):
    os.makedirs(local_dir, exist_ok=True)
    for attempt in range(2):
        try:
            r = subprocess.run(
                ["scp", "-o", "ConnectTimeout=15", "-P", pm.REMOTE_PORT, "-r",
                 f"{pm.REMOTE_USER}@{pm.REMOTE_HOST}:{remote_path}", local_dir + "/"],
                capture_output=True, timeout=timeout)
            return r.returncode == 0
        except Exception:
            if attempt == 0:
                time.sleep(3)
    return False

pm._scp_dir = _scp_dir_robust

# ── 三实验配置 ──────────────────────────────────────────────────────────────
# series: 远程 results/<series> 目录名; exp: 本地标识; patch/batch: 展示用
EXPERIMENTS = [
    {"series": "s11_top6_p4", "exp": "s11_top6_p4", "patch": 4, "batch": 224},
    {"series": "s11_top6_p2", "exp": "s11_top6_p2", "patch": 2, "batch": 24},
    {"series": "s11_top6_p8", "exp": "s11_top6_p8", "patch": 8, "batch": 640},
]
DATASET_SIZE = 10866          # top6 训练集
TOTAL_STEPS = 600000

# 本地 CSV (show/seen 样本标注 + GT 顺序), 缺失时从远程拉取
SHOW_CSV = os.path.join(HERE, "show2_top6.csv")
SEEN_CSV = os.path.join(HERE, "seen2_top6.csv")


# ── 远程收集 (按 series, 不依赖全局最新) ─────────────────────────────────────
def collect_series(series):
    """收集单个实验: 最新 log + eval_auto jsons + show/seen 目录列表。
    返回 (latest_log, log_content, eval_json, ckpt_dir, show_steps, seen_steps)"""
    # 用 -printf 直接排序, 避免 `find|xargs ls -t` 在 find 无输出时
    # xargs 退化为 `ls -t` 列出远程 cwd 的坑。
    latest = pm._ssh(
        f"find {pm.REMOTE_BASE}/5script/results/{series} -name log.txt "
        f"-printf '%T@ %p\\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-",
        timeout=20).strip()
    # 去掉可能的 float 时间戳前缀残留 (cut -f2- 已处理, 但防御一下)
    if latest and "/" not in latest:
        latest = ""
    if not latest:
        return None, "", "", None, [], []
    run_dir = "/".join(latest.split("/")[:-1])
    series_dir = "/".join(latest.split("/")[:-2])
    ckpt_dir = f"{run_dir}/checkpoints"
    SEP = "===_DSH_SEP_==="
    script = (
        f"echo '{SEP}LOG'; cat {run_dir}/log.txt 2>/dev/null; "
        f"echo '{SEP}EVAL'; cat {series_dir}/*/checkpoints/eval_auto_*.json 2>/dev/null; "
        f"echo '{SEP}SHOW'; ls {ckpt_dir}/show_samples/ 2>/dev/null; "
        f"echo '{SEP}SEEN'; ls {ckpt_dir}/seen_samples/ 2>/dev/null; "
        f"echo '{SEP}END'"
    )
    combined = pm._ssh(script, timeout=60)
    parts = combined.split(SEP)
    def _strip(s):
        lines = s.strip().splitlines()
        if lines and lines[0].strip() in ("LOG", "EVAL", "SHOW", "SEEN", "END"):
            return "\n".join(lines[1:]).strip()
        return s.strip()
    log_content = _strip(parts[1]) if len(parts) >= 2 else ""
    eval_json = _strip(parts[2]) if len(parts) >= 3 else ""
    show_steps = [l.strip() for l in _strip(parts[3]).splitlines() if l.strip()] if len(parts) >= 4 else []
    seen_steps = [l.strip() for l in _strip(parts[4]).splitlines() if l.strip()] if len(parts) >= 5 else []
    return latest, log_content, eval_json, ckpt_dir, show_steps, seen_steps


# ── 每实验流程 ──────────────────────────────────────────────────────────────
def _exp_local_dirs(exp):
    """每实验独立的本地 samples 目录 (防三实验互相覆盖)"""
    base = os.path.join(HERE, f"s11_{exp['patch']}")
    os.makedirs(base, exist_ok=True)
    return {
        "show": os.path.join(base, "show_samples"),
        "seen": os.path.join(base, "seen_samples"),
    }


def run_one(exp, verbose=True, do_poster=True):
    """处理单个实验: 收集 -> 解析 -> 写 train_data JSON -> 样本/poster/dashboard"""
    series, exp_name, patch, batch = exp["series"], exp["exp"], exp["patch"], exp["batch"]
    latest, log_content, eval_json, ckpt_dir, show_steps, seen_steps = collect_series(series)
    if not latest:
        if verbose:
            print(f"[{datetime.datetime.now():%H:%M:%S}] {exp_name}: 无 log (尚未开始?)")
        return 0

    rows = pm.parse(log_content or "")
    pm.merge_eval_json(rows, eval_json or "")

    # 写本实验 train_data JSON
    out_json = os.path.join(HERE, f"train_data_{exp_name}.json")
    out_rows = []
    latest_sps = None
    for r in rows:
        nr = dict(r)
        nr["epoch"] = (r["step"] * batch) / DATASET_SIZE
        if r.get("stepsPerSec"):
            latest_sps = r["stepsPerSec"]
        out_rows.append(nr)
    sps = latest_sps or 3.5
    last_step = out_rows[-1]["step"] if out_rows else 0
    out = {
        "source": f"remote:{pm.REMOTE_HOST}:{series}",
        "pulledAt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(rows), "rows": out_rows, "patch": patch, "batch": batch,
        "dataset_size": DATASET_SIZE, "stepsPerSec": sps,
        "total_steps": TOTAL_STEPS,
        "eta_hours": max(0, TOTAL_STEPS - last_step) / (sps * 3600) if sps else 0,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    eval_rows = [r for r in rows if r.get("is_eval")]
    poster_path = None
    if do_poster and ckpt_dir and (show_steps or seen_steps):
        # 拉 show/seen 样本 (极小), 生成 poster
        dirs = _exp_local_dirs(exp)
        if show_steps:
            pm._pull_sample_dir(ckpt_dir, "show_samples", dirs["show"], False)
        if seen_steps:
            pm._pull_sample_dir(ckpt_dir, "seen_samples", dirs["seen"], False)
        poster_path = _regen_poster(exp, ckpt_dir, dirs, verbose)

    # 生成本实验 dashboard
    dash = build_dashboard(out_json, exp_name, poster_path, verbose)
    if verbose:
        tail = f"step={last_step}" if last_step else "无数据"
        print(f"[{datetime.datetime.now():%H:%M:%S}] {exp_name}: rows={len(rows)} {tail}"
              f" evals={len(eval_rows)} sps={sps:.1f}")
        if dash:
            print(f"  dashboard -> {dash}")
    return len(rows)


# ── poster (复用 make_eval_poster.py, 参数化 csv) ───────────────────────────
def _regen_poster(exp, ckpt_dir, dirs, verbose=True):
    tag = exp["exp"]
    poster_out = os.path.join(HERE, f"poster_{tag}.png")
    series_dir = "/".join(ckpt_dir.split("/")[:-2])

    # 收集该 series 所有 eval_auto json 到临时目录, 供 make_eval_poster 读
    eval_json_tmp = tempfile.mkdtemp(prefix="eval_json_")
    for ejf in glob.glob(os.path.join(series_dir, "*/checkpoints/eval_auto_*.json")):
        shutil.copy2(ejf, eval_json_tmp)

    show_steps = [s for s in pm._existing_steps(dirs["show"])]
    seen_steps = [s for s in pm._existing_steps(dirs["seen"])]
    if not show_steps:
        shutil.rmtree(eval_json_tmp, ignore_errors=True)
        return None
    args = [sys.executable, os.path.join(HERE, "make_eval_poster.py"),
            "--show5-dir", dirs["show"],
            "--show5-csv", SHOW_CSV,
            "--eval-json-dir", eval_json_tmp,
            "--exp", tag, "-o", poster_out]
    if seen_steps and os.path.exists(SEEN_CSV):
        args += ["--seen5-dir", dirs["seen"], "--seen5-csv", SEEN_CSV]
    try:
        r = subprocess.run(args, capture_output=True, timeout=180)
        if verbose:
            if r.returncode != 0:
                print(f"[poster] {tag} rc={r.returncode}: {r.stderr.decode('utf-8','replace')[:300]}")
            else:
                print(f"[poster] {tag}: {len(show_steps)} steps -> {poster_out}")
        return poster_out if r.returncode == 0 else None
    except Exception as e:
        if verbose:
            print(f"[poster] {tag} failed: {e}")
        return None
    finally:
        shutil.rmtree(eval_json_tmp, ignore_errors=True)


# ── dashboard (参数化 OUT_JSON, 复用 __DATA__ 注入) ─────────────────────────
def build_dashboard(out_json, exp_name, poster_path=None, verbose=True):
    template_path = os.path.join(HERE, "train_dashboard.html")
    if not os.path.exists(template_path) or not os.path.exists(out_json):
        return None
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            t = f.read()
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_js = json.dumps(data, ensure_ascii=False)
        t = t.replace("const COLORS = {",
                      f"const __DATA__ = {data_js};\nconst COLORS = {{", 1)
        old_fetch = (
            "    const res = await fetch('train_data.json?t='+Date.now());\n"
            "    if(!res.ok) throw new Error('HTTP '+res.status);\n"
            "    const data = await res.json();"
        )
        t = t.replace(old_fetch, "    const data = __DATA__;", 1)
        p_name = os.path.basename(poster_path) if poster_path else f"poster_{exp_name}.png"
        old_poster = (
            "async function loadPoster(){\n"
            "  await loadImg('latestImg', ['eval_latest.png?t='+Date.now()]);\n"
            "  await loadImg('posterImg', ['eval_poster.png?t='+Date.now()]);\n"
            "}"
        )
        new_poster = (
            f"function loadPoster(){{\n"
            f"  const li=document.getElementById('latestImg'); if(li){{li.src='{p_name}';li.onerror=()=>{{li.style.display='none';}};}}\n"
            f"  const pi=document.getElementById('posterImg'); if(pi){{pi.src='{p_name}';pi.onerror=()=>{{pi.style.display='none';}};}}\n"
            f"}}"
        )
        t = t.replace(old_poster, new_poster, 1)
        t = t.replace("load();\nsyncAuto();", "load();", 1)
        os.makedirs(DASH_DIR, exist_ok=True)
        src_poster = os.path.join(HERE, p_name)
        if os.path.exists(src_poster):
            shutil.copy2(src_poster, os.path.join(DASH_DIR, p_name))
        out = os.path.join(DASH_DIR, f"{exp_name}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(t)
        return out
    except Exception as e:
        if verbose:
            print(f"[dash] {exp_name} failed: {e}")
        return None


# ── 三实验对比 dashboard ────────────────────────────────────────────────────
def build_compare_dashboard(verbose=True):
    """聚合三实验 eval rows -> dashboards/s11_compare.html (4 指标 x 3 曲线)"""
    exps_data = []
    for exp in EXPERIMENTS:
        out_json = os.path.join(HERE, f"train_data_{exp['exp']}.json")
        if not os.path.exists(out_json):
            continue
        with open(out_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        evals = [{"step": r["step"], "mse": r.get("mse"), "ssim": r.get("ssim"),
                  "skel_iou": r.get("skel_iou"), "lpips": r.get("lpips"),
                  "ts": r.get("ts")}
                 for r in d.get("rows", []) if r.get("is_eval")]
        if evals:
            exps_data.append({"name": exp["exp"], "patch": exp["patch"],
                              "batch": exp["batch"], "evals": evals})
    if len(exps_data) < 2:
        if verbose:
            print(f"[compare] 只有 {len(exps_data)}/3 实验有数据, 对比 dashboard 暂缓")
        return None
    html = COMPARE_TEMPLATE.replace("__DATA__", json.dumps(exps_data, ensure_ascii=False))
    os.makedirs(DASH_DIR, exist_ok=True)
    out = os.path.join(DASH_DIR, "s11_compare.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"[compare] {len(exps_data)}/3 实验 -> {out}")
    return out


COMPARE_TEMPLATE = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>s11 top6 patch 对比 (S model, kl-f4)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#ddd}
h1{font-size:18px} h2{font-size:14px;color:#8ab}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#1b1f27;border:1px solid #333;border-radius:8px;padding:10px}
canvas{max-height:280px}
.tbl{margin:8px 0;border-collapse:collapse;width:100%}
.tbl td,.tbl th{border:1px solid #333;padding:3px 8px;font-size:12px}
.meta{color:#888;font-size:12px;margin-bottom:8px}
</style></head><body>
<h1>s11: DiT-2Cond-S/{2,4,8} top6 kl-f4 对比
<span style="font-weight:normal;color:#888;font-size:12px">pulled: <span id="pulled"></span></span></h1>
<div class="meta">每点 = 一个 ckpt 的 eval (MSE/SSIM/SkelIoU/LPIPS, CPU daemon 计算);
patch=4: batch224 (进行中) | patch=2: batch24 | patch=8: batch640</div>
<div id="tables"></div>
<div class="grid" id="charts"></div>
<script>
const DATA = __DATA__;
document.getElementById('pulled').textContent = new Date().toLocaleString();
const PALETTE = ['#4fc3f7','#ff8c42','#81c784','#e85d75','#ba68c8'];
const METRICS = [
  ['mse','MSE (越低越好)'],
  ['ssim','SSIM (越高越好)'],
  ['skel_iou','SkelIoU (越高越好)'],
  ['lpips','LPIPS (越低越好)'],
];
const tables = document.getElementById('tables');
let t = '<table class="tbl"><tr><th>实验</th><th>patch</th><th>batch</th><th>ckpts</th>';
METRICS.forEach(m => t += `<th>最新 ${m[0]}</th>`);
t += '</tr>';
DATA.forEach(d => {
  const last = d.evals[d.evals.length-1] || {};
  t += `<tr><td>${d.name}</td><td>${d.patch}</td><td>${d.batch}</td><td>${d.evals.length}</td>`;
  METRICS.forEach(m => { const v = last[m[0]]; t += `<td>${v==null?'—':v.toFixed(4)}</td>`; });
  t += '</tr>';
});
tables.innerHTML = t + '</table>';

const grid = document.getElementById('charts');
METRICS.forEach((m, mi) => {
  const card = document.createElement('div'); card.className='card';
  card.innerHTML = `<h2>${m[1]}</h2><canvas id="c${mi}"></canvas>`;
  grid.appendChild(card);
  const ds = DATA.map(d => ({
    label: `p${d.patch} (${d.name})`,
    data: d.evals.map(e => ({x: e.step, y: e[m[0]]})),
    borderColor: PALETTE[d.patch===2?0:d.patch===4?1:2],
    backgroundColor: PALETTE[d.patch===2?0:d.patch===4?1:2],
    pointRadius: 3, tension: 0.15, borderWidth: 2, fill: false,
  }));
  new Chart(document.getElementById('c'+mi), {
    type: 'line',
    data: {datasets: ds},
    options: {
      responsive: true, interaction: {mode:'nearest', intersect:false},
      scales: {x:{type:'linear', title:{display:true,text:'step'}}, y:{beginAtZero:false}},
      plugins: {legend:{labels:{color:'#ddd', boxWidth:12}}},
    },
  });
});
</script></body></html>"""


# ── 单轮 & loop ─────────────────────────────────────────────────────────────
def run_once(verbose=True, do_poster=True):
    # 确保本地 CSV 存在 (show2/seen2 标注+GT 顺序)
    for local, remote_path in [(SHOW_CSV, f"{pm.REMOTE_BASE}/5script/show2_top6.csv"),
                               (SEEN_CSV, f"{pm.REMOTE_BASE}/5script/seen2_top6.csv")]:
        if not os.path.exists(local):
            subprocess.run(["scp", "-o", "ConnectTimeout=15", "-P", pm.REMOTE_PORT,
                            f"{pm.REMOTE_USER}@{pm.REMOTE_HOST}:{remote_path}", local],
                           capture_output=True, timeout=60)
            if os.path.exists(local):
                print(f"[csv] pulled {os.path.basename(local)}")
    for exp in EXPERIMENTS:
        run_one(exp, verbose, do_poster)
    build_compare_dashboard(verbose)


def main():
    interval = 120
    if "--interval" in sys.argv:
        try:
            interval = int(sys.argv[sys.argv.index("--interval") + 1])
        except Exception:
            pass
    do_poster = "--no-poster" not in sys.argv
    if "--loop" in sys.argv:
        print(f"[pull_s11] loop 每 {interval}s (poster={'on' if do_poster else 'off'})")
        while True:
            t0 = time.time()
            try:
                run_once(do_poster=do_poster)
            except Exception as e:
                print(f"[pull_s11] error: {e}")
            time.sleep(max(1, interval - (time.time() - t0)))
    else:
        run_once(do_poster=do_poster)


if __name__ == "__main__":
    main()