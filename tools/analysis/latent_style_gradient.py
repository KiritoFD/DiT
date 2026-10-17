"""Pure frozen-latent diagnostic helpers; never updates model weights."""
import numpy as np
import torch
import torch.nn.functional as F


def validate_pretrained_archive(path,expected_sha256):
    """Reject truncated/wrong torchvision caches before an expensive audit."""
    import hashlib
    import zipfile
    from pathlib import Path
    path=Path(path)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected_sha256:raise ValueError(f'Pretrained weight SHA256 mismatch: {path.name}')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('Corrupt pretrained weight ZIP entry')
    return actual


def rms(x):return x.float().square().flatten(1).mean(1).sqrt()


def bounded_step(z,gradient,relative_rms):
    scale=relative_rms*rms(z)/rms(gradient).clamp_min(1e-12)
    return z-gradient*scale[:,None,None,None]


def reference_ids(data,writer,script,character):
    tr=np.asarray(data['splits']['train'])
    positive=tr[(data['writer'][tr]==writer)&(data['script'][tr]==script)&(data['char'][tr]!=character)]
    negative=tr[(data['writer'][tr]!=writer)&(data['script'][tr]==script)&(data['char'][tr]==character)]
    return positive,negative


def prototypes(features,data,metadata):
    """Character-balanced, train-only prototypes excluding each query character."""
    tr=np.asarray(data['splits']['train']);all_proto=[];all_masks=[];cache={}
    for row in metadata:
        key=(row['script'],row['character'])
        if key in cache:
            per,valid=cache[key];all_proto.append(per);all_masks.append(valid);continue
        per=[];valid=[]
        for slot,(_,script) in enumerate(data['slots']):
            ids=tr[(data['labels'][tr]==slot)&(data['char'][tr]!=row['character'])]
            ok=script==row['script'] and len(ids)>0
            if ok:
                # Give each source and then each character equal weight.
                cm=[]
                for char in sorted(set(data['char'][ids])):
                    ix=ids[data['char'][ids]==char];sm=[]
                    for source in sorted(set(data['source'][ix])):sm.append(features[ix[data['source'][ix]==source]].mean(0))
                    cm.append(np.mean(sm,axis=0))
                p=np.mean(cm,axis=0);p=p/max(np.linalg.norm(p),1e-12)
            else:p=np.zeros(features.shape[1],np.float32)
            per.append(p);valid.append(ok)
        cache[key]=(per,valid);all_proto.append(per);all_masks.append(valid)
    return torch.tensor(np.asarray(all_proto),dtype=torch.float32),torch.tensor(np.asarray(all_masks),dtype=torch.bool)


def hard_reference_features(features,data,metadata,proto):
    positives=[];negatives=[];masks=[]
    for i,row in enumerate(metadata):
        target=data['slots'].index((row['writer'],row['script']))
        _,ids=reference_ids(data,row['writer'],row['script'],row['character'])
        entries=[];mask=[]
        for slot in range(len(data['slots'])):
            ix=ids[data['labels'][ids]==slot]
            e=features[ix].mean(0) if len(ix) else np.zeros(features.shape[1],np.float32)
            e=e/max(np.linalg.norm(e),1e-12);entries.append(e);mask.append(bool(len(ix)))
        positives.append(proto[i,target]);negatives.append(entries);masks.append(mask)
    return torch.stack(positives),torch.tensor(np.asarray(negatives),dtype=torch.float32),torch.tensor(masks,dtype=torch.bool)


def contrastive_per_image(embedding,positive,negative,mask):
    assert mask.any(1).all(),'No silently substituted negative protocol'
    cp=(embedding*positive).sum(1)
    cn=torch.einsum('bd,bkd->bk',embedding,negative)
    loss=F.softplus((cn-cp[:,None]+.1)/.1)
    return (loss*mask).sum(1)/mask.sum(1)


def prototype_scores(embedding,proto,mask,target):
    sims=torch.einsum('bd,bkd->bk',embedding,proto).masked_fill(~mask,-float('inf'))
    selected=sims.gather(1,target[:,None]).squeeze(1)
    others=sims.clone();others.scatter_(1,target[:,None],-float('inf'))
    return {'top1':(sims.argmax(1)==target).float(),'margin':selected-others.max(1).values,'target_cos':selected}
