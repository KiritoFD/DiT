"""Train a 20x32 writer table independently on frozen DINO features, train images only."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.cohort_eval.datasets import read_rows, image_key, sha


def cls_feature(out):
    # HuggingFace ModelOutput is also dict-like, but has no x_norm_clstoken key.
    if isinstance(out,dict) and 'x_norm_clstoken' in out:
        return out['x_norm_clstoken']
    return out.last_hidden_state[:,0]


def extract(items, batch=128):
    import cv2
    from src.loss.losses import _default_dino_ckpt,_load_local_dinov2
    path=_default_dino_ckpt()
    if path is None:raise FileNotFoundError('Local DINO checkpoint required')
    encoder=_load_local_dinov2(path).cuda().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    def load(row):
        arr=cv2.imread(row['image_path'],cv2.IMREAD_GRAYSCALE)
        if arr is None:raise FileNotFoundError(row['image_path'])
        return torch.from_numpy(cv2.resize(arr,(256,256),interpolation=cv2.INTER_AREA)).unsqueeze(0).repeat(3,1,1)
    result=[];start=time.monotonic()
    with ThreadPoolExecutor(12) as pool,torch.inference_mode():
        for i in range(0,len(items),batch):
            x=torch.stack(list(pool.map(load,items[i:i+batch]))).cuda().float()/255
            out=encoder((x-mean)/std)
            feature=cls_feature(out)
            result.append(feature.float().cpu())
            if i//batch%20==0:print('FEATURES',min(i+batch,len(items)),len(items),round(time.monotonic()-start,1),flush=True)
    del encoder
    return torch.cat(result).numpy(),str(path)


def train(args):
    torch.set_num_threads(8);torch.manual_seed(90);np.random.seed(90)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    data=Path(args.data)
    tr=read_rows(data/'train.csv');val=read_rows(data/'validation.csv');test=read_rows(data/'test.csv')
    assert not {image_key(r) for r in tr}&{image_key(r) for r in val+test}
    mapping=json.loads((data/'tables/callig_remap.json').read_text())
    if len(mapping)!=20:raise ValueError('This experiment expects 20 writers')
    labels=np.array([mapping[r['calligrapher_id']] for r in tr],np.int64)
    val_labels=np.array([mapping[r['calligrapher_id']] for r in val],np.int64)
    feature,teacher=extract(tr+val,args.batch)
    np.savez_compressed(out/'features.npz',feat=feature,paths=np.array([image_key(r) for r in tr+val]))
    x=F.normalize(torch.from_numpy(feature[:len(tr)]).cuda(),dim=-1)
    vx=F.normalize(torch.from_numpy(feature[len(tr):]).cuda(),dim=-1)
    y=torch.from_numpy(labels).cuda();vy=torch.from_numpy(val_labels).cuda()
    table=nn.Embedding(20,32).cuda();nn.init.normal_(table.weight,std=.02)
    projection=nn.Sequential(nn.Linear(x.shape[1],128),nn.GELU(),nn.Linear(128,32)).cuda()
    opt=torch.optim.AdamW(list(table.parameters())+list(projection.parameters()),lr=.002,weight_decay=1e-4)
    # Uniform writer sampling, positives from distinct characters wherever available.
    groups=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(tr):groups[int(labels[i])][r['character']].append(i)
    rng=np.random.default_rng(90);best=(-1.,None,None);history=[]
    for step in range(1,args.steps+1):
        indices=[]
        for writer in range(20):
            chars=sorted(groups[writer]);chosen=rng.choice(chars,min(16,len(chars)),replace=False)
            indices.extend(int(rng.choice(groups[writer][ch])) for ch in chosen)
        idx=torch.tensor(indices,device='cuda');target=y[idx]
        a=F.normalize(table.weight,dim=-1);b=F.normalize(projection(x[idx]),dim=-1)
        logits=a@b.T/.07
        positives=(torch.arange(20,device='cuda')[:,None]==target[None,:]).float()
        forward=-(logits.log_softmax(1)*positives).sum(1)/positives.sum(1)
        backward=F.cross_entropy(logits.T,target)
        off=~torch.eye(20,dtype=torch.bool,device='cuda')
        repel=F.relu((a@a.T)[off]-.15).square().mean()
        loss=.5*(forward.mean()+backward)+3*repel
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%250==0 or step==1:
            with torch.no_grad():
                w=F.normalize(table.weight,dim=-1)
                z=F.normalize(projection(vx),dim=-1)
                acc=float(((z@w.T).argmax(1)==vy).float().mean())
            history.append(dict(step=step,loss=float(loss),validation_top1=acc))
            print(history[-1],flush=True)
            if acc>best[0]:best=(acc,w.detach().cpu().clone(),{k:v.detach().cpu().clone() for k,v in projection.state_dict().items()})
    # Orthogonal coordinate alignment preserves learned cosine geometry while reducing
    # arbitrary basis changes seen by the already trained generator. Match old row norms.
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    weights={k.removeprefix('_orig_mod.'):v for k,v in ck['ema'].items()}
    old=weights['writer.weight'][:20].float();raw=best[1].float()
    u,_,vh=torch.linalg.svd(raw.T@F.normalize(old,dim=-1),full_matrices=False)
    rotation=u@vh;aligned=(raw@rotation)*old.norm(dim=1,keepdim=True)
    np.save(out/'callig_table_raw.npy',raw.numpy());np.save(out/'callig_table.npy',aligned.numpy())
    (out/'callig_remap.json').write_text(json.dumps(mapping,indent=2))
    torch.save(dict(projection=best[2],rotation=rotation,table=raw),out/'contrastive.pt')
    record=dict(method='independent bidirectional row-feature SupCon + margin repulsion; distinct-character positives',
        training_csv=str(data/'train.csv'),training_csv_sha256=sha(data/'train.csv'),
        validation_csv_sha256=sha(data/'validation.csv'),test_used=False,teacher=teacher,
        teacher_sha256=sha(teacher),steps=args.steps,best_validation_top1=best[0],history=history,
        exported_shape=[20,32],table_sha256=sha(out/'callig_table.npy'),
        alignment='global orthogonal rotation to source EMA table, then preserve each source row norm',
        coordinate_reference=args.checkpoint,coordinate_reference_sha256=sha(args.checkpoint),
        normalized_geometry_max_error=float(((F.normalize(aligned,dim=-1)@F.normalize(aligned,dim=-1).T)-(raw@raw.T)).abs().max()))
    (out/'provenance.json').write_text(json.dumps(record,indent=2))
    print('TABLE_READY',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch',type=int,default=128)
    train(p.parse_args())


if __name__=='__main__':main()
