#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 /tmp/evals_summary.csv 生成自包含 dashboard HTML (横轴 steps, 纵轴 eval 结果)。
输出: /tmp/v8_dashboard.html — 引用 chart.umd.min.js (file:// 可用, 数据内联)。
"""
import csv, json, os

rows = list(csv.DictReader(open('/tmp/evals_summary.csv', encoding='utf-8')))

# 整理: {exp_kind: [(step, ssim, lpips, mse), ...]}
series = {}
for r in rows:
    key = '%s_%s' % (r['exp'], r['kind'])
    try:
        ssim = float(r['ssim']); lpips = float(r['lpips']) if r['lpips'] else None
        mse = float(r['mse']) if r['mse'] else None
        step = int(r['step'])
    except (ValueError, TypeError):
        continue
    series.setdefault(key, []).append((step, ssim, lpips, mse))

# s31 用手动修正曲线（daemon 的 eval_auto_ctrl 是旧 bug 口径 0.54-0.56）
s31_fixed = [(2500, 0.7355), (10000, 0.7685), (20000, 0.7917), (30000, 0.8021), (42500, 0.8081)]
series['s31_ctrl_gt_skel_1px_ctrl_FIXED'] = [(s, v, None, None) for s, v in s31_fixed]

def js(key):
    pts = sorted(series[key])
    steps = [p[0] for p in pts]
    ssims = [p[1] for p in pts]
    lpips = [p[2] for p in pts]
    mses = [p[3] for p in pts]
    return {'steps': steps, 'ssim': ssims, 'lpips': lpips, 'mse': mses}

groups = {
    'base': ['s21_fame_flow_v2_base', 's30_dino_char_strong_pretrain_base', 'v8_3stage_base'],
    'ctrl': ['s31_ctrl_gt_skel_1px_ctrl_FIXED', 's32b_repa_strong_ctrl', 's32c_chain_ctrl'],
}
COLORS = {
    's21_fame_flow_v2_base': '#8b93a7',
    's30_dino_char_strong_pretrain_base': '#f5a623',
    'v8_3stage_base': '#3ddc97',
    's31_ctrl_gt_skel_1px_ctrl_FIXED': '#8b93a7',
    's32b_repa_strong_ctrl': '#f5a623',
    's32c_chain_ctrl': '#3ddc97',
}
LABELS = {
    's21_fame_flow_v2_base': 'S21 base (旧)',
    's30_dino_char_strong_pretrain_base': 'S30 base (旧, best 0.4841@130k)',
    'v8_3stage_base': 'v8a base (新, 进行中)',
    's31_ctrl_gt_skel_1px_ctrl_FIXED': 's31 ctrl (修正eval, best 0.8081)',
    's32b_repa_strong_ctrl': 's32b REPA w0.3 (best 0.8177)',
    's32c_chain_ctrl': 's32c REPA 90k (best 0.8204@40k)',
}

datasets = {}
for g, keys in groups.items():
    datasets[g] = []
    for k in keys:
        if k not in series:
            continue
        datasets[g].append({
            'label': LABELS.get(k, k), 'key': k,
            'color': COLORS.get(k, '#4f9cff'),
            'data': js(k),
        })

data_js = json.dumps(datasets, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>DiT eval 对比 dashboard (steps x eval)</title>
<style>
body{background:#0f1117;color:#e6e9ef;font-family:-apple-system,Segoe UI,Roboto,sans-serif;padding:20px}
h1{font-size:17px;margin:8px 0}h2{font-size:14px;color:#8b93a7;margin:18px 0 6px}
.card{background:#161a22;border:1px solid #2a2f3c;border-radius:8px;padding:12px;margin-bottom:18px}
canvas{width:100%;height:320px}
.meta{color:#8b93a7;font-size:12px;margin-bottom:10px}
.legend{color:#8b93a7;font-size:12px;margin-top:6px}
</style></head><body>
<h1>DiT 实验 eval 对比 · 横轴 steps / 纵轴 指标</h1>
<div class="meta">数据: 远程 eval_auto_*.json 汇总 (2026-09-02) · s31 用修正 eval (原 daemon 为 bug 口径)</div>
<div class="card"><h2>基模 base · SSIM</h2><canvas id="baseSsim"></canvas></div>
<div class="card"><h2>基模 base · LPIPS</h2><canvas id="baseLpips"></canvas></div>
<div class="card"><h2>ctrl / REPA · SSIM</h2><canvas id="ctrlSsim"></canvas></div>
<div class="card"><h2>ctrl / REPA · LPIPS</h2><canvas id="ctrlLpips"></canvas></div>
<script src="chart.umd.min.js"></script>
<script>
const DATA = __DATA__;
function mk(id, ds, ylabel){
  const ctx = document.getElementById(id);
  new Chart(ctx, {type:'line', data:{datasets: ds.map(d=>({
    label:d.label, data:d.data.steps.map((s,i)=>({x:s, y:d.data.ssim?d.data.ssim[i]:null})),
    borderColor:d.color, backgroundColor:d.color+'22', borderWidth:1.8, pointRadius:0, tension:0.25, spanGaps:true
  }))}, options:{
    responsive:true, maintainAspectRatio:false, animation:false,
    scales:{x:{type:'linear',title:{display:true,text:'steps',color:'#8b93a7'},ticks:{color:'#8b93a7'},grid:{color:'#2a2f3c'}},
             y:{title:{display:true,text:ylabel,color:'#8b93a7'},ticks:{color:'#8b93a7'},grid:{color:'#2a2f3c'}}},
    plugins:{legend:{labels:{color:'#8b93a7',font:{size:11}}}}
  }});
}
function seriesFor(group, metric){
  return DATA[group].map(d=>{
    const pts = d.data;
    const yv = pts[metric];
    return {label:d.label, data:pts.steps.map((s,i)=>({x:s, y:yv[i]})),
            borderColor:d.color, backgroundColor:d.color+'22', borderWidth:1.8, pointRadius:0, tension:0.25, spanGaps:true};
  });
}
mk('baseSsim', seriesFor('base','ssim'), 'SSIM');
mk('baseLpips', seriesFor('base','lpips'), 'LPIPS');
mk('ctrlSsim', seriesFor('ctrl','ssim'), 'SSIM');
mk('ctrlLpips', seriesFor('ctrl','lpips'), 'LPIPS');
</script>
</body></html>"""

html = html.replace('__DATA__', data_js)
with open('/tmp/v8_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('wrote /tmp/v8_dashboard.html', os.path.getsize('/tmp/v8_dashboard.html'), 'bytes')