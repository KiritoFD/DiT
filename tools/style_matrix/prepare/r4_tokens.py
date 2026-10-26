"""Offline train-only R4 initializer. Never called implicitly by a batch probe."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from ..runtime.common import save,sha,seed


class ImageRows(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        im=Image.open(self.rows[i]['image_path']).convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255


class LocalAggregator(nn.Module):
    def __init__(self,dim):
        super().__init__();self.project=nn.Sequential(nn.Linear(dim,64),nn.LayerNorm(64),nn.SiLU())
        self.query=nn.Parameter(torch.randn(8,64)*.02)
        self.attention=nn.MultiheadAttention(64,4,batch_first=True)
        self.output=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,64))
    def forward(self,x):
        x=self.project(x);q=self.query[None].expand(len(x),-1,-1)
        return self.output(self.attention(q,x,x,need_weights=False)[0]+q)


def metadata(csv_path,excluded):
    rows=list(csv.DictReader(open(csv_path,encoding='utf-8-sig')))
    assert {int(r['calligrapher_id']) for r in rows}==set(range(10))
    forbidden_paths=set();forbidden_pixels=set()
    for path in excluded:
        for row in csv.DictReader(open(path,encoding='utf-8-sig')):
            forbidden_paths.add(row['image_path'])
            if row.get('pixel_sha256'):forbidden_pixels.add(row['pixel_sha256'])
    assert not any(r['image_path'] in forbidden_paths or
                   (r.get('pixel_sha256') and r['pixel_sha256'] in forbidden_pixels) for r in rows), 'Evaluation leakage'
    return rows


def extract(a,rows,out):
    from tools.vae.train_dino_calli_vae import load_dino_model
    model=load_dino_model(a.dino,torch.device('cuda'))
    loader=DataLoader(ImageRows(rows),batch_size=64,num_workers=8,pin_memory=True)
    features=np.lib.format.open_memmap(out/'patch_features.npy',mode='w+',dtype=np.float16,
                                      shape=(len(rows),64,model.config.hidden_size))
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    offset=0
    with torch.no_grad():
        for images in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                tokens=model((images.cuda(non_blocking=True)-mean)/std).last_hidden_state[:,1:]
                side=int(tokens.shape[1]**.5);assert side*side==tokens.shape[1]
                tokens=F.adaptive_avg_pool2d(tokens.transpose(1,2).reshape(len(tokens),-1,side,side),8).flatten(2).transpose(1,2)
            features[offset:offset+len(tokens)]=tokens.float().cpu().numpy();offset+=len(tokens)
            if offset%1024<64:print('R4_EXTRACT',offset,len(rows),flush=True)
    features.flush()
    save(out/'feature_identity.json',dict(csv_sha256=sha(a.csv),dino_sha256=sha(a.dino),
        n=len(rows),shape=list(features.shape),excluded=[dict(path=p,sha256=sha(p)) for p in a.exclude_csv]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--exclude-csv',action='append',required=True)
    p.add_argument('--dino',default='pretrained_models/dinov2_vits14_pretrain.safetensors')
    p.add_argument('--out',required=True);p.add_argument('--extract',action='store_true')
    p.add_argument('--steps',type=int,default=4000)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);seed(90)
    rows=metadata(a.csv,a.exclude_csv)
    if a.extract:extract(a,rows,out)
    identity=json.loads((out/'feature_identity.json').read_text());assert identity['csv_sha256']==sha(a.csv)
    features=np.load(out/'patch_features.npy',mmap_mode='r')
    w=np.array([int(r['calligrapher_id']) for r in rows]);s=np.array([int(r['script_id']) for r in rows])
    chars=sorted({r['character'] for r in rows});char_index={c:i for i,c in enumerate(chars)}
    c=np.array([char_index[r['character']] for r in rows]);rng=np.random.default_rng(90)
    held=set(rng.choice(len(chars),max(1,len(chars)//10),replace=False).tolist())
    train=np.array([i for i in range(len(rows)) if c[i] not in held]);valid=np.array([i for i in range(len(rows)) if c[i] in held])
    groups=defaultdict(list)
    for i in train:groups[s[i],w[i],c[i]].append(i)
    cells=[]
    for sid in sorted(set(s)):
        for wa in range(10):
            for wb in range(wa+1,10):
                shared=sorted({ch for sc,ww,ch in groups if sc==sid and ww==wa}&
                              {ch for sc,ww,ch in groups if sc==sid and ww==wb})
                if len(shared)>=2:cells.append((sid,wa,wb,shared))
    if not cells:raise ValueError('R4 needs legal same-script 2x2 writer-character cells')
    model=LocalAggregator(features.shape[-1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    best=-1.;start=time.monotonic()
    def tensor(ids):return torch.from_numpy(np.array(features[ids],copy=True)).cuda().float()
    @torch.no_grad()
    def embed(ids):
        parts=[];model.eval()
        for begin in range(0,len(ids),128):parts.append(model(tensor(ids[begin:begin+128])).cpu())
        return torch.cat(parts)
    for step in range(1,a.steps+1):
        ids=[]
        for _ in range(32):
            sid,wa,wb,common=cells[int(rng.integers(len(cells)))];cx,cy=rng.choice(common,2,replace=False)
            ids.extend(int(rng.choice(groups[sid,ww,ch])) for ww,ch in ((wa,cx),(wa,cy),(wb,cx),(wb,cy)))
        ids=np.array(ids);writer=torch.from_numpy(w[ids]).cuda();script=torch.from_numpy(s[ids]).cuda();char=torch.from_numpy(c[ids]).cuda()
        model.train();e=F.normalize(model(tensor(ids)).mean(1),dim=1)
        logits=e@e.T/.1
        positive=(writer[:,None]==writer[None])&(char[:,None]!=char[None])&(script[:,None]==script[None])
        negative=(writer[:,None]!=writer[None])&(script[:,None]==script[None])
        hard=negative&(char[:,None]==char[None]);den=logits+hard.float()*np.log(2)
        logprob=logits-den.masked_fill(~(positive|negative),-torch.inf).logsumexp(1)[:,None]
        loss=-(logprob.masked_fill(~positive,0).sum(1)/positive.sum(1).clamp_min(1)).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%100==0:print('R4_PRETRAIN',step,float(loss),time.monotonic()-start,flush=True)
        if step%500==0 or step==a.steps:
            train_e=F.normalize(embed(train).mean(1),dim=1);val_e=F.normalize(embed(valid).mean(1),dim=1)
            correct=[]
            for sid in sorted(set(s[valid])):
                writers=[ww for ww in range(10) if np.any((s[train]==sid)&(w[train]==ww))]
                prototypes=torch.stack([train_e[(s[train]==sid)&(w[train]==ww)].mean(0) for ww in writers])
                query=s[valid]==sid;prediction=np.array(writers)[(val_e[query]@F.normalize(prototypes,dim=1).T).argmax(1)]
                correct.extend((prediction==w[valid][query]).tolist())
            score=float(np.mean(correct));print('R4_CHAR_HOLDOUT',step,score,flush=True)
            if score>best:
                best=score;torch.save(dict(model=model.state_dict(),step=step,validation=score),out/'aggregator_best.pt')
    ck=torch.load(out/'aggregator_best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
    tokens=embed(np.arange(len(rows)));table=torch.zeros(11,8,64)
    for writer in range(10):
        per_script=[]
        for sid in sorted(set(s[w==writer])):
            per_char=[tokens[(w==writer)&(s==sid)&(c==ch)].mean(0) for ch in sorted(set(c[(w==writer)&(s==sid)]))]
            per_script.append(torch.stack(per_char).mean(0))
        table[writer]=torch.stack(per_script).mean(0)
    torch.save(dict(tokens=table,train_csv_sha256=sha(a.csv),selected_step=ck['step']),out/'writer_tokens.pt')
    save(out/'completion.json',dict(table_sha256=sha(out/'writer_tokens.pt'),feature_identity=identity,
        selected_step=ck['step'],holdout_accuracy=best,seconds=time.monotonic()-start,
        inference='Writer ID lookup only; table remains trainable during FM'))


if __name__=='__main__':main()
