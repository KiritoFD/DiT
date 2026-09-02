# -*- coding: utf-8 -*-
"""build_pretrain_dashboard.py — 生成基模 eval 对比 dashboard (自包含 HTML, 数据内嵌)。

横轴: steps, 纵轴: eval 指标 (SSIM / LPIPS / MSE / SkelIoU)。
分组: base (基模预训练) / skel (骨架 ControlNet) / repa (repa 增强) / all。
顶部按钮切换分组; 统计卡显示每个实验的 best/latest SSIM。

ControlNet 实验用 **ctrl** 分支 (带条件后的生成质量), 悬停可看 base 参考值。

用法:
  python tools/build_pretrain_dashboard.py \
      --data _sync_work/dashboard_data_all.json \
      --out tools/pretrain_eval_dashboard.html
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

COLORS = {
    # base
    "s21": "#4f9cff", "s25": "#3ddc97", "s28": "#ff6b6b", "s30": "#ffb86b",
    # skel
    "s26": "#bd93f9", "s29": "#ff79c6", "s31": "#8be9fd", "1pix": "#50fa7b",
    # repa
    "s32": "#f1fa8c", "s32b": "#ff5555", "s32c": "#00d0ff",
}

GROUP_LABEL = {"base": "Base 基模预训练", "skel": "Skel 骨架ControlNet",
               "repa": "REPA 增强"}

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>实验 eval 对比 (base / skel / repa)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1117;--card:#1a1d27;--line:#2a2f3c;--txt:#e6e9ef;--dim:#8b93a7;
--acc:#4f9cff;--warn:#ff6b6b;--ok:#3ddc97;--track:#3a4150}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:-apple-system,"Segoe UI",Roboto,sans-serif;
font-size:13px;line-height:1.5}
header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;
gap:12px;flex-wrap:wrap}
header h1{font-size:15px;font-weight:600}
.meta{color:var(--dim);font-size:12px}
.controls{margin-left:auto;display:flex;gap:6px}
button{background:var(--card);border:1px solid var(--line);color:var(--txt);
border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer}
button.active{background:var(--acc);border-color:var(--acc);color:#fff}
.legend{display:flex;gap:10px;flex-wrap:wrap;padding:10px 20px 0}
.legend-chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--dim)}
.legend-chip .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.stats{display:flex;gap:10px;padding:12px 20px 0;flex-wrap:wrap}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:9px 13px;
min-width:160px}
.stat .k{color:var(--dim);font-size:11px}
.stat .v{font-size:16px;font-weight:700;margin-top:2px}
.stat .sub{color:var(--dim);font-size:10px;margin-top:1px}
.charts{padding:14px 20px 24px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.charts{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
.panel h2{font-size:13px;margin:0 0 4px;font-weight:600}
.panel .hint{font-size:11px;color:var(--dim);margin:0 0 10px}
.chart-wrap{position:relative;height:300px}
.note{padding:0 20px 24px;color:var(--dim);font-size:12px;max-width:980px}
.note b{color:var(--txt)}
.note code{background:var(--card);padding:1px 5px;border-radius:3px}
</style>
</head>
<body>
<header>
  <h1>实验 eval 对比</h1>
  <span class="meta" id="metaLine">—</span>
  <div class="controls">
    <button data-g="base" class="active">Base</button>
    <button data-g="skel">Skel</button>
    <button data-g="repa">REPA</button>
    <button data-g="all">All</button>
  </div>
</header>
<div class="legend" id="legend"></div>
<div class="stats" id="statGrid"></div>
<div class="charts">
  <div class="panel">
    <h2>SSIM（越高越好）</h2>
    <p class="hint">结构相似度 —— 主指标</p>
    <div class="chart-wrap"><canvas id="ssimChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>LPIPS（越低越好）</h2>
    <p class="hint">感知相似度（视觉质量）</p>
    <div class="chart-wrap"><canvas id="lpipsChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>MSE（越低越好）</h2>
    <p class="hint">像素级重建误差（噪声类指标）</p>
    <div class="chart-wrap"><canvas id="mseChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>Skel IoU（越高越好）</h2>
    <p class="hint">骨架结构吻合度</p>
    <div class="chart-wrap"><canvas id="skelChart"></canvas></div>
  </div>
</div>
<div class="note" id="note">
  <b>阶段链路</b>：base（基模预训练）→ skel（骨架 ControlNet 条件）→ repa（表示对齐增强）。
  ControlNet 实验画的是 <code>ctrl</code> 分支（带骨架条件后的生成质量），悬停可看同 step 的 <code>base</code> 参考。
</div>

<script>
const DATA = __DATA__;
const COLORS = __COLORS__;
const GROUP_LABEL = __GROUP_LABEL__;
let CUR = 'base';
let CHARTS = {};

function keysIn(group){
  return Object.keys(DATA).filter(k => group==='all' || DATA[k].group===group);
}
function seriesOf(key, field){
  const e = DATA[key]; if(!e) return {labels:[],data:[],base:{}};
  const rows = e.rows.filter(r => r[field]!==null && r[field]!==undefined).slice()
                 .sort((a,b)=>a.step-b.step);
  const bs = {};
  rows.forEach(r=>{ if(r.base_ssim!==undefined) bs[r.step]=r.base_ssim; });
  return {labels: rows.map(r=>r.step), data: rows.map(r=>r[field]), base: bs};
}
function build(){
  keysIn(CUR).length;
  const ks = keysIn(CUR);
  const fields = [['ssimChart','ssim','SSIM'],['lpipsChart','lpips','LPIPS'],
                  ['mseChart','mse','MSE'],['skelChart','skel_iou','Skel IoU']];
  fields.forEach(([id, field, title])=>{
    const ctx = document.getElementById(id).getContext('2d');
    const datasets = ks.map(k=>{
      const s = seriesOf(k, field);
      return {label: DATA[k].name,
              data: s.data.map((v,i)=>({x:s.labels[i], y:v, base:s.base[s.labels[i]]})),
              borderColor: COLORS[k]||'#888', backgroundColor: COLORS[k]||'#888',
              borderWidth:2, pointRadius:2, tension:0.15, fill:false};
    });
    if(CHARTS[id]) CHARTS[id].destroy();
    CHARTS[id] = new Chart(ctx, {type:'line', data:{datasets},
      options:{responsive:true, maintainAspectRatio:false,
        interaction:{mode:'nearest', intersect:false},
        scales:{x:{type:'linear', title:{display:true,text:'steps',color:'#8b93a7'},
                   ticks:{color:'#8b93a7'}, grid:{color:'#2a2f3c'}},
                y:{title:{display:true,text:title,color:'#8b93a7'},
                   ticks:{color:'#8b93a7'}, grid:{color:'#2a2f3c'}}},
        plugins:{legend:{labels:{color:'#e6e9ef', boxWidth:12}},
          tooltip:{callbacks:{
            label:(c)=>{
              const p=c.raw; let t=`${c.dataset.label}: ${Number(p.y).toFixed(4)}`;
              if(p.base!==undefined) t += `  (base ${Number(p.base).toFixed(4)})`;
              return t;
            }}}}}});
  });
  // legend + stats
  document.getElementById('legend').innerHTML = ks.map(k=>
    `<span class="legend-chip"><span class="dot" style="background:${COLORS[k]||'#888'}"></span>${DATA[k].name}</span>`).join('');
  document.getElementById('statGrid').innerHTML = ks.map(k=>{
    const e=DATA[k]; if(!e.rows.length) return '';
    const best=e.rows.reduce((a,b)=>b.ssim>a.ssim?b:a, e.rows[0]);
    const last=e.rows[e.rows.length-1];
    let sub=`@step ${best.step} | latest ${last.ssim.toFixed(4)}@${last.step} | ${e.rows.length}pts`;
    if(last.base_ssim!==undefined) sub += ` | base ${last.base_ssim.toFixed(4)}`;
    return `<div class="stat"><div class="k">${e.name}</div>`+
      `<div class="v" style="color:${COLORS[k]}">SSIM ${best.ssim.toFixed(4)}</div>`+
      `<div class="sub">${sub}</div></div>`;
  }).join('');
  document.getElementById('metaLine').textContent =
    `${GROUP_LABEL[CUR]||'全部'} — ${ks.length} 个实验`;
}
document.querySelectorAll('.controls button').forEach(b=>{
  b.addEventListener('click', ()=>{
    document.querySelectorAll('.controls button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); CUR=b.dataset.g; build();
  });
});
build();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="_sync_work/dashboard_data_all.json")
    ap.add_argument("--out", default="tools/pretrain_eval_dashboard.html")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    html = (TEMPLATE
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__COLORS__", json.dumps(COLORS, ensure_ascii=False))
            .replace("__GROUP_LABEL__", json.dumps(GROUP_LABEL, ensure_ascii=False)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(html)} bytes)")
    for k, v in data.items():
        best = max(v["rows"], key=lambda r: r["ssim"])
        print(f"  {k:6s} [{v['group']:4s}] {len(v['rows']):>2} pts  "
              f"best_ssim={best['ssim']:.4f}@{best['step']}")


if __name__ == "__main__":
    main()
