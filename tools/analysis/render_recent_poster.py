"""Paired real-PNG comparison, matched to the historical V68 sample columns."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eval.metrics import ssim,frag_ratio
from src.eval.metrics_ink import ink_iou

DATA=ROOT/'assets/recent_poster_20261017'
OUT=ROOT/'docs/experiments/recent_vs_v68_20261017'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixels(path):return np.array(Image.open(path).convert('RGB'),copy=True)
def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc',size)


def prepare(partial=False):
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    history=json.loads((ROOT/'docs/experiments/v68_v70_sde/evidence.json').read_text(encoding='utf-8'))
    samples=history['samples'].copy()
    metadata=list(csv.DictReader((ROOT/'exp-std/csv/eval200_fixed.csv').open(encoding='utf-8-sig')))
    # Two deterministic additional writers cover li-script absent from the old8.
    selected_writers=set()
    for i,row in enumerate(metadata):
        if row['script']=='隶' and row['calligrapher'] not in selected_writers:
            samples.append(dict(idx=i,**{k:row[k] for k in ('character','calligrapher','script')}))
            selected_writers.add(row['calligrapher'])
            if len(selected_writers)==2:break
    snapshots={}
    remote_folder='png48_selected' if (DATA/'png48_selected/snapshot.json').exists() else 'png48'
    for folder in ('png',remote_folder):
        if partial and not (DATA/folder/'snapshot.json').exists():continue
        snapshot=json.loads((DATA/folder/'snapshot.json').read_text(encoding='utf-8'))
        for key,record in snapshot.items():
            record['local_folder']=folder
            snapshots[key]=record
    gts=[ROOT/f'eval/gt/{i}.png' for i in range(187)]
    gt_pixels=[hashlib.sha256(pixels(p).tobytes()).hexdigest() for p in gts]
    for key,row in snapshots.items():
        if key=='std':continue
        assert len(row['gt_pixel_sha256'])==187,key
        for i in range(187):
            assert row['gt_pixel_sha256'][str(i)]==gt_pixels[i],(key,i,'GT alignment mismatch')
    def local(key):return str(DATA/snapshots[key]['local_folder']/key/'g{idx}.png')
    rows=[dict(key='gt',title='GT · 真迹',description='同一批原图 / 固定书家与字符',pattern=str(ROOT/'eval/gt/{idx}.png')),
          dict(key='std',title='STD · 当前软字形条件',description='64² 输入；V68 使用字表条件',pattern=local('std')),
          dict(key='v68',title='V68 · ODE',description='200k · 增广 + C2OT · 历史基准',pattern=str(ROOT/'eval/v68/{idx}.png')),
          dict(key='v68_sde',title='V68 · 历史 SDE',description='200k · 旧漂移 γ0.5 · 历史归档',pattern=str(ROOT/'eval/v68_sde/{idx}.png'))]
    for row in manifest['rows']:
        if row['key']=='std':continue
        if partial and row['key'] not in snapshots:continue
        rows.append(dict(row,pattern=local(row['key'])))
    return rows,samples,gts,snapshots


def measure(rows,gts,snapshots):
    import lpips
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    net=lpips.LPIPS(net='alex',verbose=False).eval()
    old=json.loads((ROOT/'docs/experiments/v68_v70_sde/row_metrics.json').read_text(encoding='utf-8'))
    path=DATA/'metrics.json'
    result=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    truth=[pixels(p).astype(np.float32)/255 for p in gts]
    for row in rows:
        key=row['key'];paths=[Path(row['pattern'].format(idx=i)) for i in range(187)]
        supplied=snapshots.get(key,{}).get('unified_metrics')
        if supplied:
            hashes=[snapshots[key]['image_sha256'][str(i)] for i in range(187)]
            assert all(sha(p)==hashes[i] for i,p in enumerate(paths) if p.exists())
        else:hashes=[sha(p) for p in paths]
        gt_hashes=[sha(p) for p in gts]
        if key in result and result[key]['hashes']==hashes and result[key]['gt_hashes']==gt_hashes:continue
        if supplied:
            values=supplied['per_image'];means=supplied['mean']
            assert supplied['metric_source_sha256']=={k:sha(ROOT/'src/eval'/k) for k in ('metrics.py','metrics_ink.py')}
        elif key in ('v68','v68_sde'):
            assert hashes==[old[key]['image_sha256'][f'eval/{key}/{i}.png'] for i in range(187)]
            assert all(old[key]['gt_sha256'][f'eval/gt/{i}.png']==gt_hashes[i] for i in range(187))
            values=old[key]['per_image'];means=old[key]['mean']
        elif key=='gt':
            values=[];means=dict(ssim=1.,lpips_alex=0.,ink_iou=1.,frag=1.)
        else:
            preds=[]
            for p in paths:
                im=Image.open(p).convert('RGB')
                if im.size!=(256,256):
                    # STD is a64px condition: metric/render use explicitly declared bilinear scaling.
                    assert key=='std'
                    im=im.resize((256,256),Image.Resampling.BILINEAR)
                preds.append(np.array(im,copy=True).astype(np.float32)/255)
            def item(i):
                p,g=preds[i],truth[i]
                return dict(idx=i,ssim=ssim(p,g),ink_iou=ink_iou(p,g),frag=frag_ratio(p,g))
            with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(item,range(187)))
            for start in range(0,187,8):
                p=torch.from_numpy(np.stack(preds[start:start+8])).permute(0,3,1,2)*2-1
                g=torch.from_numpy(np.stack(truth[start:start+8])).permute(0,3,1,2)*2-1
                scores=net(p,g).flatten().tolist()
                for j,value in enumerate(scores):values[start+j]['lpips_alex']=float(value)
            means={m:float(np.mean([v[m] for v in values])) for m in ('ssim','lpips_alex','ink_iou','frag')}
        result[key]=dict(n=187,mean=means,per_image=values,hashes=hashes,gt_hashes=gt_hashes)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print('MEASURED',key,means,flush=True)
    return result


def render(rows,samples,metrics,snapshots):
    OUT.mkdir(parents=True,exist_ok=True)
    left=690;cell=256;gap=16;row_height=274;top=425
    width=left+len(samples)*(cell+gap)+32
    height=top+len(rows)*row_height+228
    image=Image.new('RGB',(width,height),'#f7f6f2');draw=ImageDraw.Draw(image)
    ink='#142a3b';gray='#667583';blue='#176579';orange='#b85d28'
    def text(x,y,value,size=26,color=ink,bold=False):draw.text((x,y),value,font=font(size,bold),fill=color)
    text(48,30,'从 V68 到深窄 DiT：实际生成结果对照',62,ink,True)
    text(50,122,'固定 eval200fix · 同字同书家 · 原始生成 PNG · 2026-10-17 快照',31,blue,True)
    text(50,177,'沿用旧 V68 的 8 个固定样本，再按评测表顺序补两位书家的隶书；不按新模型涨幅挑样。',26,gray)
    text(50,221,'每行指标均为全部 187 张 PNG 的统一重评：SSIM↑ / LPIPS-alex↓ / Ink IoU↑ / Frag≈1。',26,gray)
    text(50,307,'实验 / checkpoint',30,ink,True)
    text(50,357,'不同架构与训练进度的实际快照',24,gray)
    for j,sample in enumerate(samples):
        x=left+j*(cell+gap)
        text(x+8,285,f"{sample['character']} · {sample['script']}",39,ink,True)
        text(x+8,343,sample['calligrapher'],25,gray)
        text(x+8,382,f"idx {sample['idx']}",20,gray)
    for i,row in enumerate(rows):
        y=top+i*row_height;key=row['key'];m=metrics[key]['mean']
        if i%2:draw.rectangle((28,y-3,width-28,y+cell+8),fill='#eef0ed')
        color=orange if key=='v68_sde' else blue if key.startswith('l20_') else ink
        text(50,y+10,row['title'],34,color,True)
        text(50,y+65,row['description'],23,gray)
        text(50,y+113,f"SSIM {m['ssim']:.4f}   LPIPS-A {m['lpips_alex']:.4f}",25,ink,True)
        text(50,y+157,f"Ink IoU {m['ink_iou']:.4f}   Frag {m['frag']:.3f}",25,ink)
        if key in snapshots and snapshots[key].get('step'):
            text(50,y+205,f"{snapshots[key]['host']} · step {snapshots[key]['step']:,} · N=187",21,gray)
        for j,sample in enumerate(samples):
            p=Path(row['pattern'].format(idx=sample['idx']));tile=Image.open(p).convert('RGB')
            if tile.size!=(cell,cell):tile=tile.resize((cell,cell),Image.Resampling.BILINEAR)
            x=left+j*(cell+gap);image.paste(tile,(x,y))
            draw.rectangle((x,y,x+cell,y+cell),outline='#dce1df',width=1)
    y=top+len(rows)*row_height+12
    text(50,y,'阅读边界',28,ink,True)
    text(50,y+46,'1  原图与当前各行 GT 像素逐张校验一致；图像未锐化、未二值化。STD 的64²输入仅放大显示。',24,gray)
    text(50,y+88,'2  LPIPS 统一为 alex；与训练日志中的 VGG 数值不能混用。Frag 是8邻域连通块比，过低也可能是粘连。',24,gray)
    text(50,y+130,'3  步数、条件和训练配方不完全匹配；历史噪声/seed 未统一。这是视觉对照，不是单因素深度消融。',24,gray)
    text(50,y+172,'4  V68 SDE 为历史旧漂移归档；V84D 与 V90B 为训练中的已完成评测快照。',24,orange)
    image.save(OUT/'poster.png')
    preview=image.copy();preview.thumbnail((1700,2000));preview.save(OUT/'preview.jpg',quality=94)
    evidence=dict(samples=samples,rows=rows,source_snapshots=snapshots,
                  metric_protocol='RGB float01 PNG; Gaussian SSIM11 sigma1.5; LPIPS-alex; ink0.5; frag8conn; N187',
                  selected_image_sha256={r['key']:{str(s['idx']):sha(r['pattern'].format(idx=s['idx'])) for s in samples} for r in rows})
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['row','n','ssim','lpips_alex','ink_iou','frag']);writer.writeheader()
        for r in rows:writer.writerow(dict(row=r['key'],n=187,**metrics[r['key']]['mean']))
    print('POSTER_READY',OUT/'poster.png',image.size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--measure-only',action='store_true',help='Measure already downloaded hosts while another archive is transferring')
    args=parser.parse_args()
    rows,samples,gts,snapshots=prepare(partial=args.measure_only)
    measured=json.loads((DATA/'metrics.json').read_text(encoding='utf-8')) if args.render_only else measure(rows,gts,snapshots)
    if not args.measure_only:render(rows,samples,measured,snapshots)
