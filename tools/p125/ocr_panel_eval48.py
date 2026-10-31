"""48 机 OCR 内容正确性评测（模型自 4090 微调权重直传, md5 已校验）。

CSV 一律用各面板 request.json 记录的 csv_sha256 对上的协议文件（KB 级清单直传合规）。
验证_val100 用 4090 原版（41d2ad...），放 staging 不覆盖 48 本地同名文件。
"""
import argparse, csv, json, os, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from PIL import Image

sys.path.insert(0,'.')
os.chdir('/home/ds/Workspace/DiT')
OCR_DIR=Path('assets/p125_20261031_ocr')

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

def score(net,pngs,rows,char2idx):
    images=[];targets=[];used_scripts=[]
    for p,r in zip(pngs,rows):
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
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='docs/1031/ocr_panel_scores_48.csv')
    a=ap.parse_args()
    ck=torch.load(OCR_DIR/'duguang_ft_ext.pt',map_location='cpu',weights_only=False)
    vocab=[l.strip() for l in open(OCR_DIR/'vocab.txt',encoding='utf-8')]
    char2idx={c:i+1 for i,c in enumerate(vocab)}
    sys.path.insert(0,str(OCR_DIR.resolve()))
    from duguang_model import DuguangOCR
    net=DuguangOCR(vocab_size=len(vocab)+1)
    net.load_state_dict(ck['model'],strict=False)
    net=net.cuda().eval()
    G2='assets/style_matrix_20261028/generation2full/48'
    P125='assets/p125_20261031_ocr/validation_val100.csv'
    panels=[
        ('S5@85k_h187_s0','assets/style_focus_85k_20261029/48_v3/runs/S5/evaluations/0085000/heldout187_seed0','assets/style_focus_85k_20261029/48_v3/protocols/S_187_seed0.csv'),
        ('S5@85k_h187_s1','assets/style_focus_85k_20261029/48_v3/runs/S5/evaluations/0085000/heldout187_seed1','assets/style_focus_85k_20261029/48_v3/protocols/S_187_seed1.csv'),
        ('S5@85k_h187_s2','assets/style_focus_85k_20261029/48_v3/runs/S5/evaluations/0085000/heldout187_seed2','assets/style_focus_85k_20261029/48_v3/protocols/S_187_seed2.csv'),
        ('S5@85k_val100','assets/style_focus_85k_20261029/48_v3/runs/S5/evaluations/0085000/original_val100',P125),
        ('S11@50k_h187',G2+'/runs/S11/evaluations/0050000/heldout187_seed0',G2+'/protocols/S_187_seed0.csv'),
        ('S11@50k_val100',G2+'/runs/S11/evaluations/0050000/original_protocol',P125),
        ('J5@5k_h187',G2+'/runs/J5/evaluations/0005000/heldout187_seed0',G2+'/protocols/J_187_seed0.csv'),
        ('J6@5k_h187',G2+'/runs/J6/evaluations/0005000/heldout187_seed0',G2+'/protocols/J_187_seed0.csv'),
        ('J7@5k_h187',G2+'/runs/J7/evaluations/0005000/heldout187_seed0',G2+'/protocols/J_187_seed0.csv'),
        ('J5@5k_val100',G2+'/runs/J5/evaluations/0005000/original_protocol',P125),
        ('J6@5k_val100',G2+'/runs/J6/evaluations/0005000/original_protocol',P125),
        ('J7@5k_val100',G2+'/runs/J7/evaluations/0005000/original_protocol',P125),
        ('J4@5k_mech_orig',G2+'/mechanisms/J4_5000/original_decoder',G2+'/protocols/J_shared_train_panel.csv'),
        ('J5@5k_mech_orig',G2+'/mechanisms/J5_5000/original_decoder',G2+'/protocols/J_shared_train_panel.csv'),
        ('J6@5k_mech_orig',G2+'/mechanisms/J6_5000/original_decoder',G2+'/protocols/J_shared_train_panel.csv'),
        ('J7@5k_mech_orig',G2+'/mechanisms/J7_5000/original_decoder',G2+'/protocols/J_shared_train_panel.csv'),
        ('J5@5k_mech_swap',G2+'/mechanisms/J5_5000/decoder_swap',G2+'/protocols/J_shared_train_panel.csv'),
        ('J6@5k_mech_swap',G2+'/mechanisms/J6_5000/decoder_swap',G2+'/protocols/J_shared_train_panel.csv'),
        ('J7@5k_mech_swap',G2+'/mechanisms/J7_5000/decoder_swap',G2+'/protocols/J_shared_train_panel.csv'),
    ]
    results=[]
    for name,eval_dir,csv_path in panels:
        eval_dir=Path(eval_dir)
        steps=sorted(eval_dir.glob('eval_samples_ctrl/step*'))
        if not steps:print(f'SKIP {name}: no steps');continue
        sd=steps[0]/'eval200fix'
        if not sd.exists():
            subs=[d.name for d in steps[0].iterdir() if d.is_dir()]
            print(f'SKIP {name}: no eval200fix, has {subs}');continue
        gens=_num_sort(sd.glob('g*.png'))
        gts=_num_sort(sd.glob('gt*.png'),prefix='gt')
        # 权威配对: 评测自己写出的 eval_stdskel_batch.csv 逐样本记录
        # (idx ↔ g{idx}.png, char/script 直接给出), 无需依赖任何外部 CSV 的行序。
        batch_csv=eval_dir/'eval_stdskel_batch.csv'
        rows=[]
        if batch_csv.exists():
            by_idx={}
            for r in csv.DictReader(open(batch_csv,encoding='utf-8-sig')):
                if r.get('set')=='eval200fix':
                    by_idx[int(r['idx'])]=r
            rows=[by_idx[i] for i in sorted(by_idx) if i<len(gens)]
        if not rows:
            all_rows=list(csv.DictReader(open(csv_path,encoding='utf-8-sig')))
            manifest=sd/'samples.jsonl'
            rows=[]
            if manifest.exists():
                by_img={str(r.get('img_id','')).strip():r for r in all_rows}
                sids=[str(json.loads(l).get('sample_id','')).strip()
                      for l in open(manifest,encoding='utf-8') if l.strip()]
                rows=[by_img[s] for s in sids if s in by_img]
            if not rows:rows=all_rows
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
