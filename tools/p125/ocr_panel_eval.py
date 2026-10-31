"""Evaluate all historical checkpoints with the trained OCR content-correctness model.

2026-10-31 修复: 旧版有两处致命 bug ——
  1. `sorted(glob('g*.png'))` 是字典序 (g0,g1,g10,...), 与 CSV 行错位 ⇒ 全部目标配错;
  2. `resize((32,192))` 把 PIL (width,height) 参数当成了 (h,w), 输入是横置的。
修复后按文件名索引数值排序, 并按纵横比保持高度 32 动态宽度。
每个 panel 同时对 gt{i}.png 打分作为 sanity (GT 应接近真实书法水平)。
"""
import argparse, csv, json, os, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from PIL import Image

sys.path.insert(0,'.')
os.chdir('/root/Workspace/xy/DiT')

def _num_sort(paths,prefix='g'):
    out=[]
    for p in paths:
        m=re.search(prefix+r'(\d+)\.png$',p.name)
        if m:out.append((int(m.group(1)),p))
    return [p for _,p in sorted(out)]

def _load_img(path):
    img=Image.open(path).convert('L');w,h=img.size
    tw=int(np.clip(round(32*w/h),16,320))
    img=img.resize((tw,32),Image.Resampling.BICUBIC)
    x=np.asarray(img,dtype=np.float32)/255
    return (x-0.5)/0.5

def score(net,pngs,rows,char2idx,prefix='g'):
    images=[];targets=[];used_scripts=[]
    for i,(p,r) in enumerate(zip(pngs,rows)):
        ch=r.get('char') or r.get('character')
        if ch not in char2idx:continue
        images.append(torch.from_numpy(_load_img(p))[None])
        targets.append(char2idx[ch])
        used_scripts.append(r.get('script','?'))
    if not images:return None
    all_logits=[]
    with torch.no_grad():
        for i in range(0,len(images),8):
            x=torch.stack(images[i:i+8]).cuda()
            all_logits.append(net(x).cpu())
    all_logits=torch.cat(all_logits)
    t=torch.tensor(targets)
    conf=all_logits.max(-1).values
    peak=conf.argmax(-1)
    peak_logits=all_logits[torch.arange(len(peak)),peak]
    probs=F.softmax(peak_logits,dim=-1).cpu().numpy()
    top1=int((probs.argmax(1)==t.numpy()).sum())
    order=np.argsort(-probs,axis=1)
    top5=sum(1 for i in range(len(t)) if t[i] in order[i,:5])
    margins=np.array([probs[i,t[i]]-np.max(np.delete(probs[i],t[i])) for i in range(len(t))])
    by_script=defaultdict(list)
    for i,s in enumerate(used_scripts):
        by_script[s].append(1 if probs[i].argmax()==t[i] else 0)
    script_acc={s:round(100*np.mean(v),2) for s,v in sorted(by_script.items())}
    return dict(n=len(t),
        top1=round(100*top1/len(t),2),top5=round(100*top5/len(t),2),
        margin=round(float(margins.mean()),4),
        by_script=json.dumps(script_acc))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',default='assets/p125_20261030/ocr/duguang_ft_ext.pt')
    p.add_argument('--out',default='docs/1031/ocr_panel_scores.csv')
    a=p.parse_args()
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    vocab=[l.strip() for l in open('/root/.cache/modelscope/models/iic--cv_convnextTiny_ocr-recognition-handwritten_damo/snapshots/master/vocab.txt',encoding='utf-8')]
    char2idx={c:i+1 for i,c in enumerate(vocab)}
    from tools.p125.duguang_model import DuguangOCR
    net=DuguangOCR(vocab_size=len(vocab)+1)
    net.load_state_dict(ck['model'],strict=False)
    net=net.cuda().eval()
    panels=[
        ('S0@50k','assets/style_matrix_20261027/matrix64/4090/execution/runs/S0/evaluations/0050000/heldout','assets/u100_pretrain_20261024/validation_val100.csv'),
        ('S1@50k','assets/style_matrix_20261027/matrix64/4090/execution/runs/S1/evaluations/0050000/heldout','assets/u100_pretrain_20261024/validation_val100.csv'),
        ('S4@50k','assets/style_matrix_20261027/matrix64/4090/execution/runs/S4/evaluations/0050000/heldout','assets/u100_pretrain_20261024/validation_val100.csv'),
        ('S6@50k','assets/style_matrix_20261027/matrix64/4090/execution/runs/S6/evaluations/0050000/heldout','assets/u100_pretrain_20261024/validation_val100.csv'),
        ('S8@50k','assets/style_matrix_20261027/matrix64/4090/execution/runs/S8/evaluations/0050000/heldout','assets/u100_pretrain_20261024/validation_val100.csv'),
        ('S0@85k','assets/style_focus_85k_20261029/4090/runs/S0/evaluations/0085000/heldout187_seed0','exp-std/csv/eval200_fixed.csv'),
        ('S0@85k_s1','assets/style_focus_85k_20261029/4090/runs/S0/evaluations/0085000/heldout187_seed1','exp-std/csv/eval200_fixed.csv'),
        ('S4@85k','assets/style_focus_85k_20261029/4090/runs/S4/evaluations/0085000/heldout187_seed0','exp-std/csv/eval200_fixed.csv'),
        ('P0@50k','assets/p125_20261030/execution/runs/P0/evaluations/0050000/heldout','assets/u100_pretrain_20261024/validation_val100.csv'),
        ('P1@55k','assets/p125_20261030/execution/runs/P1/evaluations/0005000/heldout','assets/u100_pretrain_20261024/validation_val100.csv'),
        ('P2@120k','assets/p125_20261030/execution/runs/P2/evaluations/0015000/heldout','assets/u100_pretrain_20261024/validation_val100.csv'),
        ('P3A@125k','assets/p125_20261030/execution/runs/P3A/evaluations/0020000/heldout','assets/u100_pretrain_20261024/validation_val100.csv'),
    ]
    results=[]
    for name,png_dir,csv_path in panels:
        png_dir=Path(png_dir)
        steps=sorted(png_dir.glob('eval_samples_ctrl/step*'))
        if not steps:print(f'SKIP {name}: no steps');continue
        sd=steps[0]/'eval200fix'
        if not sd.exists():
            subs=[d for d in steps[0].iterdir() if d.is_dir()]
            print(f'SKIP {name}: no eval200fix, has {[d.name for d in subs]}');continue
        gens=_num_sort(sd.glob('g*.png'))
        gts=_num_sort(sd.glob('gt*.png'),prefix='gt')
        # 权威配对: 评测自身写出的 eval_stdskel_batch.csv (idx ↔ g{idx}.png + char)。
        # 85k heldout187 的协议 CSV 与 exp-std/csv/eval200_fixed.csv 有 3 行繁简异体差异。
        batch_csv=png_dir/'eval_stdskel_batch.csv'
        rows=[]
        if batch_csv.exists():
            by_idx={}
            for r in csv.DictReader(open(batch_csv,encoding='utf-8-sig')):
                if r.get('set')=='eval200fix':
                    by_idx[int(r['idx'])]=r
            rows=[by_idx[i] for i in sorted(by_idx) if i<len(gens)]
        if not rows:
            rows=list(csv.DictReader(open(csv_path,encoding='utf-8-sig')))
        n=min(len(gens),len(gts) or len(gens),len(rows))
        r_gen=score(net,gens[:n],rows[:n],char2idx)
        r_gt=score(net,gts[:n],rows[:n],char2idx) if gts else None
        if r_gen is None:print(f'SKIP {name}: no valid targets');continue
        row=dict(model=name,**r_gen,
            gt_top1=r_gt['top1'] if r_gt else '',gt_top5=r_gt['top5'] if r_gt else '')
        results.append(row)
        gt_s=f" | GT top1 {r_gt['top1']}%" if r_gt else ''
        print(f'{name}: Top1 {r_gen["top1"]}% Top5 {r_gen["top5"]}%{gt_s}',flush=True)
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    fields=['model','n','top1','top5','margin','by_script','gt_top1','gt_top5']
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in results:w.writerow(r)
    json.dump(results,open(a.out.replace('.csv','.json'),'w'),ensure_ascii=False,indent=1)
    print('DONE',a.out)


if __name__=='__main__':main()
