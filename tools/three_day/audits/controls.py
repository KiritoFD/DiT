"""F3 nuisance normalization and F4 GT/reconstruction/generated content attribution."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt,maximum_filter
from tools.three_day.contracts import save
from .teachers import Teacher,payloads
from .metrics import retrieval,interval


def normalized(gray,target_mass,target_width):
    arr=gray.numpy()[:,0].astype(np.float32)/255;result=[]
    for a in arr:
        mask=a<.5;inside=distance_transform_edt(mask);outside=distance_transform_edt(~mask)
        ridge=mask&(inside>=maximum_filter(inside,size=5));width=np.median(2*inside[ridge]-1) if ridge.any() else target_width
        offset=np.clip((target_width-width)/2,-2,2)
        # Fractional grayscale SDF offset changes geometry; no claim of a pure nuisance intervention.
        ink=np.clip(.5+(inside-outside)+offset,0,1)
        for _ in range(8):
            ink=np.clip(ink*target_mass/max(float(ink.mean()),1e-6),0,1)
        result.append(1-ink)
    return torch.from_numpy(np.stack(result)[:,None].astype('float32'))


def main(root):
    root=Path(root);data,latent=payloads();images=torch.load(root/'bank/images.pt',map_location='cpu',weights_only=True)
    std=torch.load(root/'bank/std_gray.pt',map_location='cpu',weights_only=True)
    metadata=json.loads((root/'bank/metadata.json').read_text());chars=np.array([r['character'] for r in metadata[:48]])
    teacher=Teacher('DINO',root,len(data['slots'])).cuda()
    def features(x):
        with torch.inference_mode():
            return torch.cat([teacher(x[i:i+32].cuda()).cpu() for i in range(0,len(x),32)]).numpy()
    stdf=features(std);f4={}
    for name,x in [('GT',images['gt']),('VAE_reconstruction',images['reconstruction']),('generated',images['generated'])]:
        f=features(x);truth=np.tile(chars,len(f)//48)
        pred=chars[(f@stdf.T).argmax(1)];correct=pred==truth
        f4[name]=dict(content_retrieval=interval(correct,truth),n=len(f),
            note='DINO nearest of 48 distinct STD characters, diagnostic only; not an independently validated OCR error rate.')
    # This is a necessary usability screen, not proof that feature gradients preserve legitimate variants.
    passed=all(f4[k]['content_retrieval']['ci95'][0]>=.8 for k in ('GT','VAE_reconstruction'))
    save(root/'content_attribution.json',dict(results=f4,content_teacher_gate=bool(passed),
        supervision_gate=False,reason='Content gradient/variant robustness audit required in addition to retrieval.'))
    del teacher
    tr=data['splits']['train'];gray=data['gray'];mass=float((1-gray[tr].float()/255).mean())
    # Estimate a neutral target on train only; the test failure list is never read.
    widths=[]
    for i in tr[:2048]:
        m=gray[i,0].numpy()<128;edt=distance_transform_edt(m);ridge=m&(edt>=maximum_filter(edt,size=5))
        if ridge.any():widths.append(np.median(2*edt[ridge]-1))
    target=float(np.median(widths));norm=normalized(gray,mass,target);report={}
    for name in ('DINO','pixel'):
        teacher=Teacher(name,root,len(data['slots'])).cuda();f=features_with(teacher,norm)
        report[name]={s:retrieval(f,data,s) for s in ('test_char','test_source')};del teacher
    save(root/'nuisance_normalization.json',dict(ink_target=mass,width_target=target,normalized=report,
        caveat='Same transformation applied to gallery and query. Fractional SDF/ink normalization alters shape; not a causal isolation of style.'))


def features_with(teacher,x):
    result=[]
    with torch.inference_mode():
        for i in range(0,len(x),128):
            with torch.autocast('cuda',dtype=torch.bfloat16):result.append(teacher(x[i:i+128].cuda()).float().cpu().numpy())
    return np.concatenate(result)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();torch.set_num_threads(8);main(a.root)
