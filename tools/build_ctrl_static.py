# -*- coding: utf-8 -*-
"""
build_ctrl_static.py — 把 train_data.json 内联进一个全新的静态 HTML dashboard.

与旧 train_dashboard.html 模板无关，专门为 ControlNet 日志设计:
  - 6 个图表: Loss / LR / Steps-per-Sec / Memory / Eval MSE / Eval SSIM
  - 数据直接内联为 JS 对象，双击即可离线看，无需 http server
  - 自适应 y 轴，spanGaps=true 处理缺失点

用法:
  python tools/build_ctrl_static.py          # 读 train_data.json → 生成 HTML
  python tools/build_ctrl_static.py --loop   # 每 30s 自动重建
"""
import os
import sys
import json
import time
import html
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_JSON = os.path.join(HERE, "train_data.json")
OUT_HTML = os.path.join(HERE, "dashboards", "ctrl_skel.html")

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--line:#2a2f3c;--txt:#e6e9ef;--dim:#8b93a7;--acc:#4f9cff;--warn:#ff6b6b;--ok:#3ddc97}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:-apple-system,"Segoe UI",Roboto,sans-serif;font-size:13px;line-height:1.5}
header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px;flex-wrap:wrap}
header h1{font-size:15px;font-weight:600}
.meta{color:var(--dim);font-size:12px}
.stats{display:flex;gap:10px;padding:14px 20px 0;flex-wrap:wrap}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:110px}
.stat .k{color:var(--dim);font-size:11px}
.stat .v{font-size:20px;font-weight:700;margin-top:2px}
.v.warn{color:var(--warn)}.v.ok{color:var(--ok)}
#statusLine{padding:6px 20px;color:var(--dim);font-size:12px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px 20px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
.panel h2{font-size:13px;font-weight:600;margin-bottom:8px;color:var(--dim)}
.chart-wrap{height:220px;position:relative}
#errbar{display:none;background:rgba(255,107,107,.15);color:var(--warn);padding:8px 20px;font:11px monospace}
.footer{padding:10px 20px;color:var(--dim);font-size:11px}
@media(max-width:800px){.charts{grid-template-columns:1fr}}
</style>
</head>
<body>
<div id="errbar"></div>
<header>
  <h1>ControlNet Training Monitor</h1>
  <span class="meta" id="meta">__META__</span>
</header>
<div class="stats" id="statGrid">__STATS__</div>
<div id="statusLine">__STATUS__</div>
<div class="charts">
  <div class="panel"><h2>Training Loss</h2><div class="chart-wrap"><canvas id="lossChart"></canvas></div></div>
  <div class="panel"><h2>Learning Rate</h2><div class="chart-wrap"><canvas id="lrChart"></canvas></div></div>
  <div class="panel"><h2>Steps / Sec</h2><div class="chart-wrap"><canvas id="spdChart"></canvas></div></div>
  <div class="panel"><h2>Memory GB</h2><div class="chart-wrap"><canvas id="memChart"></canvas></div></div>
  <div class="panel"><h2>Eval MSE (lower = better)</h2><div class="chart-wrap"><canvas id="mseChart"></canvas></div></div>
  <div class="panel"><h2>Eval SSIM (higher = better)</h2><div class="chart-wrap"><canvas id="ssimChart"></canvas></div></div>
</div>
<div class="footer">Built at __BUILD_TIME__ | data: __DATA_SOURCE__</div>
<script>
window.onerror=function(msg,src,line,col){var e=document.getElementById('errbar');if(e){e.style.display='block';e.textContent='JS Error: '+msg+' @'+line+':'+col;}};
var ROWS = __ROWS__;
var META = __META_JS__;
var charts = {};

function mk(id, label, color){
  var ctx = document.getElementById(id);
  if(!ctx) return;
  charts[id] = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: label, data: [], borderColor: color, backgroundColor: color+'18', borderWidth: 1.5, pointRadius: 0, tension: 0.2, spanGaps: true }] },
    options: { responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#8b93a7', boxWidth: 12, font: { size: 11 } } } },
      scales: { x: { ticks: { color: '#8b93a7', maxTicksLimit: 10, font: { size: 10 } }, grid: { color: '#2a2f3c' } },
                y: { ticks: { color: '#8b93a7', font: { size: 10 } }, grid: { color: '#2a2f3c' } } } } });
}

mk('lossChart','Loss','#4f9cff');
mk('lrChart','LR','#f5a623');
mk('spdChart','Steps/Sec','#8b93a7');
mk('memChart','Mem GB','#b06bff');

// eval charts: dual-dataset (base + ctrl)
function mkDual(id, labelBase, colorBase, labelCtrl, colorCtrl){
  var ctx = document.getElementById(id);
  if(!ctx) return;
  charts[id] = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: labelBase, data: [], borderColor: colorBase, backgroundColor: colorBase+'18', borderWidth: 1.5, pointRadius: 2, tension: 0.2, spanGaps: true },
      { label: labelCtrl, data: [], borderColor: colorCtrl, backgroundColor: colorCtrl+'18', borderWidth: 1.5, pointRadius: 2, tension: 0.2, spanGaps: true }
    ] },
    options: { responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#8b93a7', boxWidth: 12, font: { size: 11 } } } },
      scales: { x: { ticks: { color: '#8b93a7', maxTicksLimit: 10, font: { size: 10 } }, grid: { color: '#2a2f3c' } },
                y: { ticks: { color: '#8b93a7', font: { size: 10 } }, grid: { color: '#2a2f3c' } } } } });
}
mkDual('mseChart','MSE base (no skel)','#8b93a7','MSE ctrl (GT skel)','#ff6b6b');
mkDual('ssimChart','SSIM base (no skel)','#8b93a7','SSIM ctrl (GT skel)','#3ddc97');

(function render(){
  if(!ROWS.length){
    document.getElementById('meta').textContent = 'No data';
    return;
  }
  var steps = ROWS.map(function(r){ return r.step; });

  function fill(id, key){
    if(!charts[id]) return;
    charts[id].data.labels = steps;
    charts[id].data.datasets[0].data = ROWS.map(function(r){ return r[key]; });
    charts[id].update('none');
  }
  fill('lossChart','loss');
  fill('lrChart','lr');
  fill('spdChart','stepsPerSec');
  fill('memChart','memCur');

  var evalRows = ROWS.filter(function(r){ return r.mse != null; });
  if(evalRows.length){
    var evalSteps = evalRows.map(function(r){ return r.step; });
    charts.mseChart.data.labels = evalSteps;
    charts.mseChart.data.datasets[0].data = evalRows.map(function(r){ return r.mse_base != null ? r.mse_base : null; });
    charts.mseChart.data.datasets[1].data = evalRows.map(function(r){ return r.mse; });
    charts.mseChart.update('none');
    charts.ssimChart.data.labels = evalSteps;
    charts.ssimChart.data.datasets[0].data = evalRows.map(function(r){ return r.ssim_base != null ? r.ssim_base : null; });
    charts.ssimChart.data.datasets[1].data = evalRows.map(function(r){ return r.ssim; });
    charts.ssimChart.update('none');
  }

  // stats already rendered server-side
})();
</script>
</body>
</html>
"""


def build():
    if not os.path.isfile(DATA_JSON):
        print("[build] no train_data.json")
        return None
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("rows", [])
    experiment = data.get("experiment", "ctrl_skel")
    source = data.get("source", "")
    updated = data.get("updated", "")

    # --- server-side stats ---
    last = rows[-1] if rows else {}
    stats_html = ""
    status_html = ""

    if rows:
        losses = [r["loss"] for r in rows if r.get("loss") is not None]
        first_loss = losses[0] if losses else None
        last_loss = last.get("loss")
        trend = None
        if first_loss and last_loss:
            trend = (last_loss - first_loss) / first_loss * 100

        eval_rows = [r for r in rows if r.get("mse") is not None]
        eval_info = "no eval"
        if eval_rows:
            ev = eval_rows[-1]
            eval_info = "eval@%d MSE=%.4f SSIM=%.4f" % (ev["step"], ev["mse"], ev.get("ssim", 0))

        def stat(k, v, cls=""):
            return '<div class="stat"><div class="k">%s</div><div class="v %s">%s</div></div>' % (k, cls, v)

        stats_html = (
            stat("Step", last.get("step", "-")) +
            stat("Loss", "%.4f" % last_loss if last_loss is not None else "-") +
            stat("Loss Trend", ("%.1f%%" % trend) if trend is not None else "-", "ok" if trend and trend < 0 else "") +
            stat("LR", "%.2e" % last["lr"] if last.get("lr") is not None else "-") +
            stat("Steps/s", "%.2f" % last["stepsPerSec"] if last.get("stepsPerSec") is not None else "-") +
            stat("Mem", "%.2fG" % last["memCur"] if last.get("memCur") is not None else "-") +
            stat("Eval", eval_info, "ok" if eval_rows else "")
        )
        status_html = "step %d | loss %.4f | %s" % (last["step"], last_loss, eval_info)

    meta_html = "%s | %d rows | updated %s" % (experiment, len(rows), updated)
    build_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    t = TEMPLATE
    t = t.replace("__TITLE__", html.escape("DiT Monitor - %s" % experiment))
    t = t.replace("__META__", html.escape(meta_html))
    t = t.replace("__STATS__", stats_html)
    t = t.replace("__STATUS__", html.escape(status_html))
    t = t.replace("__BUILD_TIME__", build_time)
    t = t.replace("__DATA_SOURCE__", html.escape(source))
    t = t.replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
    t = t.replace("__META_JS__", json.dumps(meta_html, ensure_ascii=False))

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(t)

    print("[build] %s | step=%s loss=%s rows=%d" % (
        OUT_HTML,
        last.get("step", "?"),
        ("%.4f" % last["loss"]) if last.get("loss") is not None else "?",
        len(rows)
    ))
    return OUT_HTML


def main():
    if "--loop" in sys.argv:
        interval = 30
        if "--interval" in sys.argv:
            try:
                interval = int(sys.argv[sys.argv.index("--interval") + 1])
            except Exception:
                pass
        print("[build_ctrl_static] loop every %ds" % interval)
        while True:
            t0 = time.time()
            try:
                build()
            except Exception as e:
                print("[build_ctrl_static] error: %s" % e)
            time.sleep(max(1, interval - (time.time() - t0)))
    else:
        build()


if __name__ == "__main__":
    main()
