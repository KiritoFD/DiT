"""OCR acceptance harness (plan §三): validate an OCR backend on REAL calligraphy
BEFORE using it to judge generators.

Tests:
 1. Real recognition: Top-1/Top-5/Margin by script (楷/行/隶 reported separately).
 2. Synthetic corruption detection: stroke deletion, radical deletion, stroke
    merging, visually-similar wrong character. Sensitivity = P(top1 != original).
 3. Style robustness: the same character across writers/scripts.
`unsupported` = target (after s2t) not in the recognizer vocabulary; excluded
from error rates and counted separately.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0,'.')
MODEL_DIR=("_models/ocr/models/cycloneboy--ch_PP-OCRv4_rec_server_infer/snapshots/master")
H,W=48,320


def load_backend(device='cpu'):
    import onnxruntime as ort
    chars=['<blank>']
    for ln in open(os.path.join(MODEL_DIR,'ch_dict.txt'),encoding='utf-8'):chars.append(ln.rstrip('\n'))
    so=ort.SessionOptions();so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    prov=['CPUExecutionProvider'] if device=='cpu' else ['CUDAExecutionProvider','CPUExecutionProvider']
    sess=ort.InferenceSession(os.path.join(MODEL_DIR,'fp16_model.onnx' if device=='cuda' else 'model.onnx'),so,providers=prov)
    return sess,sess.get_inputs()[0].name,chars


def preprocess_arr(arr,h=H,w=W):
    """arr: float32 gray [0,1] (256,256) -> rec input (1,3,h,w) in [-1,1]."""
    from PIL import Image
    im=Image.fromarray((arr*255).astype(np.uint8)).convert('RGB')
    r=im.width/max(im.height,1)
    nw=min(W,max(8,int(round(H*r))))
    im=im.resize((nw,H),Image.BICUBIC)
    if nw<W:
        c=Image.new('RGB',(W,H),(255,255,255));c.paste(im,(0,0));im=c
    x=np.asarray(im,dtype=np.float32)/255.
    return ((x-0.5)/0.5).transpose(2,0,1)[None]


def char_probs(logits,vocab):
    """CTC per-timestep softmax, marginal over T -> vocab distribution."""
    import torch
    t=torch.from_numpy(logits[0]).float()
    p=t.softmax(-1)
    marginal=p.sum(0)
    marginal=marginal/marginal.sum()
    return marginal.numpy()


def corruptions(gray,rng):
    """Return dict of corrupted grayscale images (same 256x256)."""
    from scipy import ndimage as ndi
    mask=gray<0.5
    if mask.mean()>0.5:mask=~mask
    out={}
    # stroke deletion: erase a dense band through the ink
    m2=mask.copy()
    ys,xs=np.nonzero(m2)
    if len(ys):
        cy,cx=int(ys.mean()),int(xs.mean())
        y0=min(max(cy-6,0),255);m2[y0:y0+10,max(0,cx-110):cx+110]=False
    out['stroke_del']=np.where(m2,0.,1.)
    # radical deletion: erase one quadrant's ink
    m3=mask.copy();h,w=m3.shape
    q=rng.integers(0,4)
    sl=[np.s_[:h//2,:w//2],np.s_[:h//2,w//2:],np.s_[h//2:,:w//2],np.s_[h//2:,w//2:]][q]
    m3[sl]=False
    out['radical_del']=np.where(m3,0.,1.)
    # merging: dilate
    out['merge']=np.where(ndi.binary_dilation(mask,np.ones((5,5),bool)),0.,1.)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='assets/u100_p1p2_20261025/p2_train.csv')
    p.add_argument('--n',type=int,default=1500)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',default='docs/1031/ocr_acceptance_ppocr.json')
    a=p.parse_args()
    from opencc import OpenCC
    cc=OpenCC('s2t')
    sess,iname,vocab=load_backend(a.device)
    vset=set(vocab)
    rows=[r for r in csv.DictReader(open(a.csv,encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    # stratified sample over (writer,script)
    rng=np.random.default_rng(1031)
    by=defaultdict(list)
    for i,r in enumerate(rows):by[(r['calligrapher'],int(r['script_id']))].append(i)
    per=max(1,a.n//len(by))
    pick=[]
    for k,idxs in sorted(by.items()):
        pick+=list(rng.choice(idxs,min(per,len(idxs)),replace=False))
    print('real eval images:',len(pick),flush=True)
    res=[]
    unsupported=0
    for i in pick:
        r=rows[i]
        from PIL import Image
        gray=np.asarray(Image.open(r['image_path']).convert('L'),dtype=np.float32)/255.
        target=r['character'];t_s2t=cc.convert(target)
        if t_s2t not in vset:unsupported+=1;continue
        x=preprocess_arr(gray)
        logits=sess.run(None,{iname:x})[0]
        pr=char_probs(logits,vocab)
        # target index: try original then s2t
        tid=vocab.index(t_s2t) if t_s2t in vset else None
        order=np.argsort(-pr)
        top1=vocab[order[0]];top5=[vocab[j] for j in order[:5]]
        margin=float(pr[tid]-max(pr[j] for j in range(len(vocab)) if j!=tid)) if tid is not None else None
        conf_wrong=float(pr[order[0]]) if top1!=t_s2t else 0.
        res.append(dict(target=t_s2t,orig=target,writer=r['calligrapher'],script=r['script'],
            top1=top1,top5_hit=float(t_s2t in top5),margin=margin,conf_wrong=conf_wrong,
            gray=gray,filepath=r['image_path']))
    real=[d for d in res]
    # per-script real metrics
    by_script=defaultdict(list)
    for d in real:by_script[d['script']].append(d)
    report=dict(n_real=len(real),unsupported=unsupported,vocab_size=len(vocab)-1,backend='PP-OCRv4_rec_server',scripts={})
    for script,ds in sorted(by_script.items()):
        t1=float(np.mean([d['top1']==d['target'] for d in ds]))
        t5=float(np.mean([d['top5_hit'] for d in ds]))
        mg=float(np.mean([d['margin'] for d in ds if d['margin'] is not None])) if any(d['margin'] is not None for d in ds) else None
        cw=float(np.mean([d['conf_wrong'] for d in ds]))
        report['scripts'][script]=dict(n=len(ds),top1=round(100*t1,2),top5=round(100*t5,2),
            margin=round(mg,4) if mg is not None else None,conf_wrong=round(cw,4))
    # corruption sensitivity
    rng=np.random.default_rng(1031)
    sens=defaultdict(list)
    for d in real[:400]:
        target=d['target']
        if target not in vset:continue
        for kind,img in corruptions(d['gray'],rng).items():
            x=preprocess_arr(img)
            logits=sess.run(None,{iname:x})[0]
            pr=char_probs(logits,vocab)
            order=np.argsort(-pr)
            top1=vocab[order[0]]
            tid=vocab.index(target)
            detected=float(top1!=target)
            conf=float(pr[order[0]])
            sens[kind].append((detected,conf))
    report['corruption_sensitivity']={}
    for k,v in sens.items():
        det=[x[0] for x in v]
        report['corruption_sensitivity'][k]=dict(n=len(v),detect_rate=round(float(np.mean(det)),3),
            mean_conf_when_detected=round(float(np.mean([x[1] for x in v if x[0]] or [0])),3))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=1))
    print(json.dumps(report,ensure_ascii=False,indent=1)[:1200])


if __name__=='__main__':main()
