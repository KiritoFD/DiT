"""Reproducible 30k cohort poster, using existing PNGs only; no GPU inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import tarfile

AUDIT=Path('assets/audits/cohort_all_ckpt_20261022')
OUT=Path('assets/posters/cohort_20261022')


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    totals=[r for r in read(AUDIT/'results.csv') if r['step']=='30000' and r['model'] in ('5','10','20')]
    data={}
    for split in ('test','train_half'):
        for model in (5,10,20):
            entry=next(r for r in totals if int(r['model'])==model and int(r['tier'])==model and r['split']==split)
            rows=read(entry['per_image'])
            data[split,model]={r['image_path']:dict(r,generated_from=entry['generated_from']) for r in rows}
    selected=[]
    def choose(pool,models,q,section,label):
        ranked=sorted(pool,key=lambda key:(mean(float(data[split,m][key]['lpips']) for m in models),key))
        key=ranked[round((len(ranked)-1)*q)]
        anchor=data[split,models[-1]][key]
        selected.append(dict(section=section,label=label,image_path=key,char=anchor['char'],
                             writer=anchor['calligrapher'],script=anchor['script'],split=split,
                             metrics={str(m):data[split,m][key] for m in models}))
    split='test'
    common=data[split,5]
    writers=sorted({r['calligrapher'] for r in common.values()})
    for writer in writers:
        pool=[k for k,r in common.items() if r['calligrapher']==writer]
        choose(pool,(5,10,20),.5,'共同Top5 / 测试集','典型：LPIPS中位')
        choose(pool,(5,10,20),.9,'共同Top5 / 测试集','困难：LPIPS约P90')
    split='train_half'
    for writer in writers:
        pool=[k for k,r in data[split,5].items() if r['calligrapher']==writer]
        choose(pool,(5,10,20),.5,'共同Top5 / 训练样本','训练：LPIPS中位')
    split='test'
    for model,smaller,models in ((10,5,(10,20)),(20,10,(20,))):
        added=sorted({r['calligrapher'] for r in data[split,model].values()}-
                     {r['calligrapher'] for r in data[split,smaller].values()})[:2]
        for writer in added:
            pool=[k for k,r in data[split,model].items() if r['calligrapher']==writer]
            choose(pool,models,.5,f'Top{model}新增书家 / 测试集','新增覆盖：中位')
    for i,row in enumerate(selected):
        row['files']={}
        for m,metric in row['metrics'].items():
            base=Path(metric['generated_from'])/'eval_samples_ctrl'
            folder=base/'step0030000'/metric['set'];idx=metric['idx']
            row['files'][m]=dict(pred=(folder/f'g{idx}.png').as_posix(),gt=(folder/f'gt{idx}.png').as_posix(),
                                std=(base/(metric['set']+'_input_g')/f'g{idx}.png').as_posix())
    manifest=dict(step=30000,state='EMA',cfg=1,sampler='Heun50',rows=selected,
        totals=totals,selection='Per-writer average LPIPS across supported models: nearest rank P50/P90; train P50. '
        'Two added writers per tier selected by Unicode name order, not visual quality. '
        'No postprocessing, ink normalization, crop, or per-image geometric alignment.',
        caveat='Historical 114M models: writer table was FM-trainable, extra Top20 rows were not independently pretrained. '
        'Not the corrected frozen-table 48.33M run. Training probes and test are different images.')
    (OUT/'selection.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('SELECTED',len(selected))


def pack():
    manifest=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    hashes={}
    with tarfile.open(OUT/'images.tar.gz','w:gz') as archive:
        for i,row in enumerate(manifest['rows']):
            refs=list(row['files'].values())
            for field in ('gt','std'):
                values=[hashlib.sha256(Path(r[field]).read_bytes()).hexdigest() for r in refs]
                if len(set(values))!=1:raise ValueError(f'Different {field} for the same image: {row["image_path"]}')
                path=Path(refs[0][field]);name=f'images/r{i:02d}_{field}.png'
                archive.add(path,arcname=name);hashes[name]=values[0]
            for model,ref in row['files'].items():
                path=Path(ref['pred']);name=f'images/r{i:02d}_m{model}.png'
                archive.add(path,arcname=name);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'image_hashes.json').write_text(json.dumps(hashes,indent=2))
    print('PACKED',len(hashes),'PNGs; identical GT and STD across models checked')


def render():
    from PIL import Image,ImageDraw,ImageFont
    manifest=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    font_path='C:/Windows/Fonts/msyh.ttc'
    font=lambda size:ImageFont.truetype(font_path,size)
    width=1584;left=185;cw=270;margin=24;rh=244
    sections=list(dict.fromkeys(r['section'] for r in manifest['rows']))
    height=465+rh*len(manifest['rows'])+48*len(sections)+145
    canvas=Image.new('RGB',(width,height),'#eef2f6');d=ImageDraw.Draw(canvas)
    def text(x,y,value,size=18,color='#283548'):d.text((x,y),value,font=font(size),fill=color)
    text(28,22,'Top5 / Top10 / Top20：30k同口径代表样本',34)
    text(28,74,'历史114M共享条件U-Net  ·  EMA  ·  CFG=1  ·  Heun-50  ·  固定逐图噪声',21)
    text(28,110,'选样：每位共同书家测试P50/P90、训练P50；按三模型平均LPIPS排名，非人工挑优。',18)
    text(28,140,'下方图像均为原始生成：不调粗细、不增强对比、不裁切、不做字形对齐。',18)
    # Fair, full common-set aggregates rather than selected-row averages.
    text(28,184,'共同Top5完整集合（非所选样本均值）',20)
    for j,model in enumerate((5,10,20)):
        x=left+(j+2)*cw
        text(x+12,184,f'Top{model} @30k',22)
        for k,split in enumerate(('test','train_half')):
            r=next(r for r in manifest['totals'] if r['model']==str(model) and r['tier']=='5' and r['split']==split)
            label='测试100张' if split=='test' else '训练50张'
            text(x+10,224+k*68,label+f'  S={float(r["ssim_mean"]):.3f}',17)
            text(x+10,250+k*68,f'L={float(r["lpips_mean"]):.3f}  I={float(r["ink_iou_mean"]):.3f}',17)
    text(28,290,'S：SSIM↑  L：LPIPS↓',18)
    text(28,321,'I：Ink-IoU↑  K：软墨量/GT',18)
    text(28,369,'范围不同的测试均值不横比；新增书家只展示支持该Writer ID的模型。训练/测试不是同一张GT。',17)
    for j,title in enumerate(('STD条件','GT真迹','Top5 @30k','Top10 @30k','Top20 @30k')):
        x=left+j*cw;d.rounded_rectangle((x,410,x+cw-7,455),8,fill='#dce5ef')
        text(x+16,420,title,21)
    y=465;section=None
    palette={'共同Top5 / 测试集':'#2563eb','共同Top5 / 训练样本':'#128565',
             'Top10新增书家 / 测试集':'#a65d14','Top20新增书家 / 测试集':'#8054b0'}
    for i,row in enumerate(manifest['rows']):
        if row['section']!=section:
            section=row['section'];d.rectangle((24,y,width-24,y+39),fill=palette[section])
            text(38,y+6,section,22,'white');y+=48
        text(28,y+18,f'【{row["char"]}】',28)
        text(28,y+63,row['writer'],21)
        text(28,y+96,row['script']+'书',18)
        for line,value in enumerate(row['label'].split('：')):text(28,y+137+26*line,value,17)
        text(28,y+211,Path(row['image_path']).stem,13,'#748093')
        for j,key in enumerate(('std','gt','m5','m10','m20')):
            x=left+j*cw;d.rectangle((x,y,x+cw-8,y+rh-8),fill='white')
            path=OUT/'images'/f'r{i:02d}_{key}.png'
            if path.exists():
                im=Image.open(path).convert('RGB');scale=207/max(im.size)
                im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
                canvas.paste(im,(x+(cw-8-im.width)//2,y+4+(207-im.height)//2))
                if key.startswith('m'):
                    r=row['metrics'][key[1:]]
                    text(x+9,y+212,f'S {float(r["ssim"]):.3f}  L {float(r["lpips"]):.3f}  K {float(r["soft_ink_ratio"]):.2f}',14)
            else:text(x+52,y+90,'未训练该书家',19,'#8792a3')
        y+=rh
    text(28,y+16,'注意：旧模型书家表随FM更新，Top20后10行初始为随机；不属于新48.33M冻结表实验。',18)
    text(28,y+48,'这些是固定单seed样本，用于读图与定位差异；不代表人工盲评或已证明的书风质量排名。',18)
    text(28,y+81,'选样、逐图指标、原图路径与SHA见同目录selection.json / image_hashes.json。',17)
    canvas=canvas.crop((0,0,width,y+125))
    canvas.save(OUT/'poster.png')
    # Small QA panels remain directly readable without downscaling a long poster.
    canvas.crop((0,0,width,1050)).save(OUT/'qa_top.png')
    canvas.crop((0,465+48+10*rh,width,465+48+10*rh+850)).save(OUT/'qa_train.png')
    print('POSTER',canvas.size,OUT/'poster.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','pack','render'))
    a=p.parse_args();globals()[a.action]()
