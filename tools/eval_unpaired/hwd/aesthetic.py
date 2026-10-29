"""Exploratory CHAED-trained ridge baseline, NOT a CHAED/TabNet reproduction."""
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tools.eval_unpaired.protocol import read, save
from .model import load


@torch.inference_mode()
def features(net, images):
    result = []
    for image in images:
        # Separate aesthetic baseline preprocessing: aspect-preserving white square pad.
        side = max(image.size)
        image = ImageOps.pad(image.convert('RGB'), (side, side), color='white')
        x = torch.from_numpy(np.asarray(image.resize((32,32), Image.Resampling.NEAREST)).copy())
        result.append(net(x.permute(2,0,1)[None].float()/255).reshape(512).numpy())
    return np.stack(result)


def scores(y, pred):
    return dict(mae=float(np.abs(y-pred).mean()), pcc=float(pearsonr(y,pred).statistic),
                srcc=float(spearmanr(y,pred).statistic), n=len(y))


def fit(x, y, groups):
    choices = []
    for alpha in [1,10,100,1000,10000]:
        errors = []
        for train, val in GroupKFold(4).split(x,y,groups):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[train],y[train]); errors.extend(np.abs(model.predict(x[val])-y[val]))
        choices.append((float(np.mean(errors)),alpha))
    alpha = min(choices)[1]
    model = make_pipeline(StandardScaler(),Ridge(alpha=alpha)); model.fit(x,y)
    return model, alpha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True)
    p.add_argument('--weights', required=True); p.add_argument('--hwd-root', required=True)
    p.add_argument('--out', required=True); a=p.parse_args()
    torch.set_num_threads(2); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    data=Path(a.data); net=load(a.weights)
    rows=list(csv.DictReader((data/'ScoreAnalysisPerImage.txt').open(),delimiter='\t'))
    if len(rows)!=1000: raise ValueError('Expected official CHAED 1000 images')
    missing=[r for r in rows if not (data/r['charName']/r['imageID']).is_file()]
    # Public archive actually omits GB3632 (all10 images). Never remap its labels
    # onto another character or advertise a 1000-image paper reproduction.
    if {(r['charName'],r['imageID']) for r in missing} != {('GB3632',f'{i:03d}.jpg') for i in range(1,11)}:
        raise ValueError('Unexpected public-archive coverage; inspect mapping before training')
    rows=[r for r in rows if r not in missing]
    save(out/'coverage.json',dict(expected=1000,available=len(rows),characters=99,
        excluded=[dict(character=r['charName'],image=r['imageID']) for r in missing],
        labels_sha256=hashlib.sha256((data/'ScoreAnalysisPerImage.txt').read_bytes()).hexdigest(),
        interpretation='Explicit available-source subset, no label remapping or paper reproduction'))
    paths=[data/r['charName']/r['imageID'] for r in rows]
    y=np.array([float(r['average score']) for r in rows]); groups=np.array([r['charName'] for r in rows])
    cache=out/'features.npy'
    if not cache.exists(): np.save(cache,features(net,[Image.open(p).convert('RGB') for p in paths]))
    x=np.load(cache); pred=np.zeros(len(y)); baseline=np.zeros(len(y)); folds=[]
    for train,test in GroupKFold(5).split(x,y,groups):
        model,alpha=fit(x[train],y[train],groups[train]); pred[test]=model.predict(x[test])
        baseline[test]=y[train].mean(); folds.append(dict(alpha=alpha,**scores(y[test],pred[test])))
    validation=dict(method='HWD VGG features + StandardScaler + Ridge, nested grouped CV',
        reproduction=False,split='5 outer folds by character, inner4 group folds choose alpha; score scale0..100',
        learned=scores(y,pred),mean_baseline=scores(y,baseline),folds=folds)
    save(out/'validation.json',validation)
    print('CHAED_GROUPED_VALIDATION',validation,flush=True)
    model,alpha=fit(x,y,groups)
    scale=model[0].scale_; mean=model[0].mean_
    coef=model[1].coef_/scale
    np.savez(out/'ridge.npz',coef=coef,intercept=model[1].intercept_-coef@mean)
    jobs=read(Path(a.hwd_root)/'summary.json'); panels=[]
    for item in jobs:
        j=item['job']; dest=out/('_'.join(map(str,[j['arm'],j['step'],j['protocol']])))
        result=dict(job=j)
        for prefix in ['g','gt']:
            f=np.load(Path(a.hwd_root)/dest.name/f'{prefix}.npy')
            result[prefix]=model.predict(f).tolist()
        # Only square panels share the HWD preprocessing above.
        result['exploratory_only']=True; save(dest.with_suffix('.json'),result); panels.append(result)
    # Transfer screen: fixed original GT, erased band, thickened, blurred and blank.
    controls={k:[] for k in ['real','broken','thick','blur','blank']}
    for j in jobs[:1]:
        job=j['job']
        for i in np.linspace(0,job['n']-1,24,dtype=int):
            image=Image.open(Path(job['samples'])/f'gt{i}.png').convert('RGB')
            broken=image.copy(); ImageDraw.Draw(broken).rectangle((0,118,255,126),fill='white')
            variants=dict(real=image,broken=broken,thick=image.filter(ImageFilter.MinFilter(5)),
                          blur=image.filter(ImageFilter.GaussianBlur(2)),blank=Image.new('RGB',image.size,'white'))
            for k,v in variants.items(): controls[k].append(v)
    values={k:model.predict(features(net,v)) for k,v in controls.items()}
    rates={k:float(np.mean(values['real']>v)) for k,v in values.items() if k!='real'}
    save(out/'transfer_screen.json',dict(real_preferred_fraction=rates,alpha=alpha,
        values={k:v.tolist() for k,v in values.items()},usable_as_primary=False,
        reason='Exploratory CHAED regression; no historical-calligraphy human calibration or TabNet reproduction'))
    save(out/'summary.json',panels); save(out/'complete.json',dict(jobs=len(panels)))
    print('CHAED_TRANSFER_SCREEN',rates,flush=True)


if __name__=='__main__': main()
