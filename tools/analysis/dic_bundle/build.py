"""Build a self-contained, offline-readable experimental evidence package."""
import argparse
import csv
import hashlib
import html
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont
from content import FAMILY_NOTES, INTRO, SECTIONS


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def configuration(item):
    config = item['config']
    if 'cli' in config:
        return {**config.get('config', {}), **config['cli']}
    return config


def escape(text):
    return html.escape(str(text), quote=True)


def main(output, archives):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    models = []
    for source in archives:
        with ZipFile(source) as z:
            # Inspect generated archive paths before extracting into the dedicated package.
            for entry in z.infolist():
                target = (output / entry.filename).resolve()
                if not target.is_relative_to(output):
                    raise ValueError(entry.filename)
            z.extractall(output)
            for name in z.namelist():
                if re.fullmatch(r'evidence/registry_\w+\.json', name):
                    models.extend(json.loads(z.read(name)))
    if len(models) != 39:
        raise ValueError(f'Expected 39 selected complete snapshots, found {len(models)}')
    condition_hashes = {m['protocol'].get('g_sha256') for m in models}
    if len(condition_hashes) != 1 or None in condition_hashes:
        raise ValueError(f'Cannot use one reference STD for differing conditions: {condition_hashes}')
    expected = list(range(187))
    for m in models:
        p = output / 'images/originals' / m['id']
        actual = sorted(int(f.stem[1:]) for f in p.glob('g*.png'))
        if actual != expected:
            raise ValueError(m['id'])
    samples = list(csv.DictReader((output / 'evidence/dataset/eval200_fixed.csv').open(encoding='utf-8-sig')))
    training = list(csv.DictReader((output / 'evidence/dataset/train_top10_aug_sym.csv').open(encoding='utf-8-sig')))
    chars = {(r.get('character') or r.get('char') or '').strip() for r in training}
    for idx, r in enumerate(samples):
        r['idx'] = idx
        r['seen_in_training_csv'] = r['character'].strip() in chars
    if len(samples) != 187:
        raise ValueError(len(samples))
    write_json(output / 'evidence/dataset/sample_index.json', samples)
    source = Path(__file__).resolve().parents[3]
    audit = source / 'assets/audits/base55_20261020'
    (output / 'evidence/audit').mkdir(exist_ok=True)
    for p in audit.glob('*.json'):
        shutil.copy2(p, output / 'evidence/audit' / p.name)
    style = source / 'exp_img/metadata.json'
    if style.exists():
        shutil.copy2(style, output / 'evidence/audit/pixel_writer_v2_metadata.json')
    for filename in ['eval_all_checkpoints_4090.py', 'run_cfg_sweep_sampling.py']:
        p = source / 'tools/analysis' / filename
        if p.exists():
            shutil.copy2(p, output / 'evidence/audit' / filename)

    selected = [0, 46, 75, 80, 88, 103, 121, 133, 155, 172]
    groups = {}
    for s in samples:
        groups.setdefault(s['calligrapher'], []).append(s['idx'])
    balanced = [i for ids in groups.values() for i in (ids[len(ids)//3], ids[2*len(ids)//3])]
    font_path = source / 'tools/fonts/NotoSerifSC-VF.ttf'
    if not font_path.exists():
        font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), 18)
    small = ImageFont.truetype(str(font_path), 15)
    posters = output / 'images/posters'
    posters.mkdir(exist_ok=True)

    def poster(name, ids, cols, subtitle):
        cell, row_height, label = 180, 207, 105
        im = Image.new('RGB', (label + cell*len(cols) + 12, 110 + row_height*len(ids)), 'white')
        draw = ImageDraw.Draw(im)
        draw.text((12, 8), subtitle, font=font, fill='black')
        for j, (title, _) in enumerate(cols):
            draw.text((label + cell*j, 52), title, font=small, fill='black')
        for k, idx in enumerate(ids):
            r = samples[idx]
            y = 102 + k*row_height
            draw.text((5, y+30), f"{idx} {r['character']}", font=font, fill='black')
            draw.text((5, y+60), r['calligrapher'], font=small, fill='black')
            draw.text((5, y+84), r['script'], font=small, fill='black')
            for j, (_, relative) in enumerate(cols):
                p = output / relative / f'g{idx}.png'
                im.paste(Image.open(p).convert('RGB').resize((174, 174)), (label+cell*j, y))
        im.save(posters / name)

    ref = [('GT', 'images/reference/gt'), ('STD', 'images/reference/std')]
    def orig(model_id):
        return f'images/originals/{model_id}'
    poster('recent_models.png', selected,
           ref + [('Base55 EMA', orig('base_55k_ema')), ('DiC50 CFG1', orig('dic_50k_cfg1.0')),
                  ('Deep70 EMA', orig('deep_70k')), ('DecA70 EMA', orig('decA_70k')),
                  ('DecB10 EARLY', orig('decB_10k'))],
           '固定十字对照  B10k为早期快照  不作等预算排名')
    cols = [('GT', 'images/reference/gt'), ('Base35 EMA', orig('base_35k_ema')),
            ('Base45 EMA', orig('base_45k_ema')), ('Base55 EMA', orig('base_55k_ema')),
            ('Deep55 EMA', orig('deep_55k')), ('Deep70 EMA', orig('deep_70k'))]
    for i in range(2):
        poster(f'balanced_{i+1}.png', balanced[10*i:10*(i+1)], cols,
               '每书家按固定位置取两字  不按生成效果挑选  第'+str(i+1)+'页')
    poster('raw_ema_cfg.png', [88, 103, 46, 121, 133, 155],
           [('GT', 'images/reference/gt'), ('Base55 EMA', orig('base_55k_ema')),
            ('Base55 RAW', orig('base_55k_raw')), ('DiC50 CFG1', orig('dic_50k_cfg1.0')),
            ('DiC50 CFG1.6', orig('dic_50k_cfg1.6')), ('Dist10 CFG1.6', orig('distill_10k_cfg1.6'))],
           '权重与CFG诊断  RAW单列  不把分类器分数当作真实书风')

    fields = ['id', 'family', 'step', 'state', 'cfg', 'n', 'batch', 'lr', 'exposure_m',
              'ssim_mean', 'lpips_mean', 'ink_iou_mean', 'frag_ratio', 'soft_ink_ratio']
    summary = []
    for m in models:
        c = configuration(m)
        row = {k: m.get(k, '') for k in fields}
        row.update(batch=c.get('batch', ''), lr=c.get('lr', ''),
                   exposure_m=m['step']*c.get('batch', 0)/1e6)
        row.update({k: (m['metrics'] or {}).get(k, '') for k in fields if k.endswith('_mean') or k in ['frag_ratio', 'soft_ink_ratio']})
        summary.append(row)
    with (output / 'evidence/summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    unseen = sum(not r['seen_in_training_csv'] for r in samples)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), models=models,
                    n_snapshots=len(models), n_generated_images=187*len(models), n_reference_images=374,
                    n_eval_rows=187, n_unseen_rows_by_character_text=unseen,
                    samples=samples, balanced_indices=balanced, selected_indices=selected,
                    shared_std_sha256=next(iter(condition_hashes)),
                    source_archives=[dict(name=Path(p).name, size=Path(p).stat().st_size,
                                          sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in archives],
                    exclusions=['model checkpoints', 'VAE weights', 'full training image corpus'], files=[])
    report = render_report(models, summary, manifest)
    (output / 'report.html').write_text(report, encoding='utf-8')
    (output / 'README.txt').write_text(
        '近期 U100 与 DiC 模型完整评测包\n\n'
        '请先完整解压，再双击 report.html。无需网络、无需安装依赖。\n'
        '报告有中文说明、全部指标、逐模型配置和187字双模型对照浏览器。\n'
        'images/originals：39组×187张原始PNG；images/reference：GT与实际STD；images/posters：新生成的诊断拼图。\n'
        'evidence：配置、逐图指标、汇总曲线、协议、来源、源码身份、风格元数据与采样表。\n'
        'manifest.json：图片索引、远端来源、所有文件SHA256及限制。\n\n'
        '重要限制：Base55k为现存末点而非已证实峰值；B10k为早期结果；RAW与EMA不可混排名；'
        '蒸馏步数不是总训练曝光；旧N181风格口径不与新N187 M2混比。\n'
        '不包含checkpoint或训练图片全集。本次没有启动新训练或新采样。\n', encoding='utf-8')
    for p in sorted(output.rglob('*')):
        if p.is_file():
            manifest['files'].append(dict(path=p.relative_to(output).as_posix(), size=p.stat().st_size,
                                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_json(output / 'manifest.json', manifest)
    archive = output.with_suffix('.zip')
    with ZipFile(archive, 'x', ZIP_DEFLATED, compresslevel=3) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, str(Path(output.name) / p.relative_to(output)))
    with ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f'ZIP CRC failed: {bad}')
    print(json.dumps(dict(archive=str(archive), size=archive.stat().st_size,
                          snapshots=len(models), images=187*len(models), unseen=unseen), ensure_ascii=False))


def render_report(models, summary, manifest):
    headings = ''.join(f'<a href="#s{i}">{escape(t)}</a>' for i, (t, _) in enumerate(SECTIONS))
    body = ''.join('<section id="s'+str(i)+'"><h2>'+escape(title)+'</h2>'+
                   ''.join('<p>'+escape(p)+'</p>' for p in paragraphs)+'</section>'
                   for i, (title, paragraphs) in enumerate(SECTIONS))
    columns = ['id', 'step', 'state', 'cfg', 'exposure_m', 'ssim_mean', 'lpips_mean',
               'ink_iou_mean', 'frag_ratio', 'soft_ink_ratio']
    titles = ['模型快照', '步数', '权重', 'CFG', '阶段曝光M', 'SSIM ↑', 'LPIPS ↓', 'IoU ↑', 'Frag ↓', '墨量比']
    table = '<table><thead><tr>'+''.join(f'<th>{x}</th>' for x in titles)+'</tr></thead><tbody>'
    for r in summary:
        values = [f"{r[k]:.2f}" if k == 'exposure_m' else str(r[k]) for k in columns]
        table += '<tr>'+''.join(f'<td>{escape(v)}</td>' for v in values)+'</tr>'
    table += '</tbody></table>'
    cards = ''
    for family in FAMILY_NOTES:
        cards += '<h3>'+escape(family)+'</h3><p>'+escape(FAMILY_NOTES[family])+'</p>'
        for m in [x for x in models if x['family'] == family]:
            c = configuration(m)
            selected = {k:c.get(k, '未记录') for k in ['batch','lr','max_steps','lr_decay_steps','num_res_blocks',
                        'act_fn','residual_cond_gate','stage_specific_cond','stage_cond_mode','writer_film_rank',
                        'writer_film_layers','writer_film_spec','decouple_writer_condition','decoder_std_residual',
                        'ema_start_step','std_shape_aug']}
            p = 'evidence/'+m['id']+'/'
            cards += ('<details><summary>'+escape(m['label'])+'</summary><p>主机 '+escape(m['host'])+
                      '；权重 '+escape(m['state'])+'；N='+str(m['n'])+'；CFG='+str(m['cfg'])+'</p>')
            cards += '<p class="path">'+escape(m['training_run'])+'</p><pre>'+escape(json.dumps(selected, ensure_ascii=False, indent=2))+'</pre>'
            cards += '<p>'+ ' · '.join('<a href="'+p+filename+'">'+name+'</a>' for filename,name in
                [('config.json','完整配置'),('metrics.csv','全步数曲线'),('per_image.csv','逐图指标'),
                 ('protocol.json','采样协议'),('provenance.json','来源与日志')]
                if filename != 'config.json' or m['config'])+'</p></details>'
    images = ''.join('<figure><a href="images/posters/'+name+'"><img loading="lazy" src="images/posters/'+name+'"></a><figcaption>'+caption+'</figcaption></figure>'
                     for name,caption in [('recent_models.png','固定十字的近期模型对照。B10k为早期结果。'),
                        ('balanced_1.png','每位书家固定取两字，第一部分。'),('balanced_2.png','每位书家固定取两字，第二部分。'),
                        ('raw_ema_cfg.png','raw与EMA、CFG与蒸馏的敏感性对照。点击图片查看大图。')])
    minimal_models = [{k:m[k] for k in ['id','label','state','cfg','family']} for m in models]
    data = json.dumps(dict(models=minimal_models,samples=manifest['samples']),ensure_ascii=False).replace('</','<\\/')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>近期 U100 与 DiC 模型实验报告</title><style>
body{font:16px/1.8 "Microsoft YaHei",sans-serif;color:#222;max-width:1260px;margin:32px auto;padding:0 24px;background:#fff}
h1{font-size:30px;line-height:1.4}h2{font-size:23px;margin-top:42px}h3{font-size:19px}p{max-width:1000px}a{color:#174f83}
nav{display:flex;flex-wrap:wrap;gap:8px 22px;border-bottom:1px solid #ddd;padding:16px 0}section{scroll-margin-top:15px}
table{border-collapse:collapse;font-size:13px;white-space:nowrap;margin:18px 0;width:100%}th,td{border:1px solid #d9d9d9;padding:8px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}th{background:#e7edf4}tr:nth-child(even){background:#f7f8fa}.scroll{overflow:auto}
details{border-bottom:1px solid #ddd;padding:10px 0}summary{cursor:pointer}pre{font:13px/1.5 Consolas,monospace;white-space:pre-wrap;background:#f7f7f7;padding:14px}
.path{overflow-wrap:anywhere;font-size:13px;color:#555}figure{margin:24px 0}figure img{max-width:100%;height:auto;border:1px solid #ddd}figcaption{font-size:14px;color:#555}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;padding:16px 0}.controls label{display:flex;flex-direction:column;font-size:14px}
select,input,button{font:inherit;padding:7px;max-width:330px}button{cursor:pointer}.gridrow{display:grid;grid-template-columns:120px repeat(4,minmax(140px,1fr));border-bottom:1px solid #ddd;gap:8px;padding:12px 0}
.gridrow img{width:100%;max-width:256px;height:auto;background:white}.gridrow a{text-align:center}.tag{font-size:13px;color:#555}#gallery{overflow:auto}#galleryInner{min-width:760px}
.headerrow{position:sticky;top:0;background:white;font-weight:bold;padding-bottom:12px}.small{font-size:14px;color:#555}
@media print{nav,.controls{display:none}details{break-inside:avoid}figure{break-inside:avoid}#viewer{display:none}}
</style><h1>近期 U100 与 DiC 模型实验报告</h1><p class="small">实验产物快照 2026年10月20日 · 39组完整评测 · 7293张生成原图 · 374张GT与STD</p><p>'''+escape(INTRO)+'''</p>
<p class="small">按当前训练CSV的字符文本集合检查，未出现样本为 '''+str(manifest['n_unseen_rows_by_character_text'])+''' / 187。该标签不是作品或来源留出。</p>
<nav>'''+headings+'''<a href="#numbers">完整指标</a><a href="#settings">模型设置</a><a href="#figures">诊断拼图</a><a href="#viewer">187字浏览器</a></nav>
'''+body+'''<section id="numbers"><h2>完整快照指标</h2><p>每行指标均来自同一组实际CSV，不拼接跨步数最佳值。M为百万次样本曝光；热启为全局step对应曝光，蒸馏仅为额外阶段曝光，不可当作总训练量。缺项不补造。</p><div class="scroll">'''+table+'''</div>
<p><a href="evidence/summary.csv">下载汇总CSV</a> · <a href="manifest.json">完整索引与SHA256清单</a> · <a href="evidence/audit/paired_summary.json">Base与Deep配对统计</a></p></section>
<section id="settings"><h2>逐模型设置与来源</h2>'''+cards+'''</section><section id="figures"><h2>诊断拼图</h2>'''+images+'''</section>
<section id="viewer"><h2>完整187字并排浏览</h2><p>固定列为GT和实际STD，右侧两列可自由切换模型。选择书家或输入字符可筛选；点击原图以原始分辨率打开。默认每页16字，所有图片都已包含在本包，无需联网。</p>
<div class="controls"><label>左侧模型<select id="left"></select></label><label>右侧模型<select id="right"></select></label><label>书家<select id="writer"><option value="">全部</option></select></label><label>字符或idx<input id="search" placeholder="例如 好 或 88"></label>
<label>训练字符状态<select id="seen"><option value="">全部</option><option value="seen">训练CSV出现</option><option value="unseen">训练CSV未出现</option></select></label><button id="prev">上一页</button><button id="next">下一页</button><span id="count"></span></div>
<div id="gallery"><div id="galleryInner"></div></div></section><p class="small">交付内容：report.html、images、evidence、README.txt、manifest.json。建议解压后打开，勿在压缩软件预览窗口中直接运行HTML。</p>
<script>const DATA='''+data+''';let page=0;const $=id=>document.getElementById(id);for(const m of DATA.models){for(const id of ['left','right']){const o=document.createElement('option');o.value=m.id;o.textContent=m.label+' ['+m.state+']';$(id).appendChild(o)}}
$('left').value='base_55k_ema';$('right').value='deep_70k';for(const w of [...new Set(DATA.samples.map(s=>s.calligrapher))]){const o=document.createElement('option');o.value=w;o.textContent=w;$('writer').appendChild(o)}
function pic(src){const a=document.createElement('a');a.href=src;a.target='_blank';const im=document.createElement('img');im.src=src;im.loading='lazy';a.appendChild(im);return a}
function render(){const q=$('search').value.trim();const list=DATA.samples.filter(s=>(!$('writer').value||s.calligrapher===$('writer').value)&&(!q||s.character.includes(q)||String(s.idx)===q)&&(!$('seen').value||s.seen_in_training_csv===($('seen').value==='seen')));const pages=Math.max(1,Math.ceil(list.length/16));page=Math.min(page,pages-1);$('count').textContent=`${list.length}字 · 第${page+1}/${pages}页`;const g=$('galleryInner');g.replaceChildren();const head=document.createElement('div');head.className='gridrow headerrow';for(const x of ['样本','GT','STD',$('left').selectedOptions[0].textContent,$('right').selectedOptions[0].textContent]){const d=document.createElement('div');d.textContent=x;head.appendChild(d)}g.appendChild(head);
for(const s of list.slice(page*16,page*16+16)){const row=document.createElement('div');row.className='gridrow';const text=document.createElement('div');text.textContent=`${s.idx} ${s.character} / ${s.calligrapher} ${s.script}`;const tag=document.createElement('div');tag.className='tag';tag.textContent=(s.seen_in_training_csv?'训练出现':'训练未出现')+' · '+s.source;text.appendChild(tag);row.appendChild(text);for(const path of ['images/reference/gt','images/reference/std','images/originals/'+$('left').value,'images/originals/'+$('right').value])row.appendChild(pic(path+'/g'+s.idx+'.png'));g.appendChild(row)}}
for(const id of ['left','right','writer','seen'])$(id).onchange=()=>{page=0;render()};$('search').oninput=()=>{page=0;render()};$('prev').onclick=()=>{page=Math.max(0,page-1);render()};$('next').onclick=()=>{page++;render()};render();</script></html>'''


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('archives', nargs='+')
    a = p.parse_args()
    main(a.output, a.archives)
