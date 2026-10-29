"""Independent Writer scores and paired GT--GEN direction, with content guards."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.style_matrix.runtime.common import save


def direction(samples,rows):
    from src.eval.style.brush import features_from_gray
    def feature(path):
        image=Image.open(path).convert('L');gray=np.asarray(image,dtype=np.float32)/255.
        brush=features_from_gray(gray)
        geometry=np.array([brush.width_stats()['p50']/32.,brush.scalars.get('bbox_aspect',0.),
            brush.scalars.get('centroid_dx',0.),brush.scalars.get('centroid_dy',0.)])
        pixels=1-np.asarray(image.resize((16,16)),dtype=np.float32).flatten()/255.
        # Explicitly remove global ink offset; retains local shape/width changes.
        return np.nan_to_num(np.r_[pixels-pixels.mean(),geometry])
    gt=[feature(samples/f'gt{i}.png') for i in range(len(rows))]
    gen=[feature(samples/f'g{i}.png') for i in range(len(rows))]
    cells={}
    for i,r in enumerate(rows):cells.setdefault((r['script'],r['character']),[]).append(i)
    groups=[];details=[]
    for key,ids in cells.items():
        values=[]
        for i,j in itertools.combinations(ids,2):
            if rows[i]['calligrapher']==rows[j]['calligrapher']:continue
            truth=gt[i]-gt[j];prediction=gen[i]-gen[j]
            tn=np.linalg.norm(truth);pn=np.linalg.norm(prediction)
            if tn<1e-8:continue
            cosine=float(prediction@truth/max(tn*pn,1e-8));projection=float(prediction@truth/tn**2)
            residual=float(np.linalg.norm(prediction-projection*truth)/tn)
            values.append(cosine);details.append(dict(cell=list(key),i=i,j=j,cosine=cosine,
                                                     projection=projection,residual=residual))
        if values:groups.append(float(np.mean(values)))
    if not groups:return dict(n_cells=0,n_pairs=0,cosine=None,lo=None,hi=None,pairs=[])
    values=np.asarray(groups);rng=np.random.default_rng(90)
    boot=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return dict(n_cells=len(groups),n_pairs=len(details),cosine=float(values.mean()),
                lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),pairs=details)


def measure(folder,protocol,cache):
    folder=Path(folder);request=json.loads((folder/'complete.json').read_text())
    if (folder/'research_metrics.json').exists():
        prior=json.loads((folder/'research_metrics.json').read_text())
        if prior['identity']!=request:raise ValueError('Research metrics identity changed')
        return prior
    rows=list(csv.DictReader(open(protocol['csv'],encoding='utf-8-sig')))
    samples=folder/'eval_samples_ctrl'/f"step{request['step']:07d}"/'eval200fix'
    from tools.eval_gen_with_disc import load_evaluator_model,evaluate_generated_images
    import torch
    model,kind,size=load_evaluator_model('assets/style_disc_ce_holdout.pt','cpu')
    proto=torch.load(cache,map_location='cpu',weights_only=False)
    kwargs=dict(model=model,model_type=kind,img_size=size,gen_dir=str(samples),eval_csv=protocol['csv'],
                slot_list=proto['slot_list'],P=proto['P'],slot_scripts=proto['slot_scripts'],device='cpu',max_samples=len(rows))
    writer=evaluate_generated_images(**kwargs)
    ceiling=evaluate_generated_images(**kwargs,image_prefix='gt')
    if writer['n_eval']!=len(rows) or ceiling['n_eval']!=len(rows):raise ValueError('Independent Writer silently dropped samples')
    from argparse import Namespace
    from tools.style_matrix.report_matrix import cmd_geometry
    geometry_path=folder/'geometry.json'
    if not geometry_path.exists():cmd_geometry(Namespace(samples_dir=str(samples),out=str(geometry_path)))
    result=dict(identity=request,writer=writer,gt_writer=ceiling,direction=direction(samples,rows),
                content=json.loads((folder/'result.json').read_text()),geometry=json.loads(geometry_path.read_text()))
    save(folder/'research_metrics.json',result)
    return result


def decoder_pair(scores):
    """Conservative, prespecified promotion; absence of evidence chooses S11."""
    def improves(candidate,baseline):
        a,b=scores[candidate],scores[baseline];da,db=a['direction'],b['direction']
        content=(a['content']['ssim']>=b['content']['ssim']-.005 and
                 a['content']['lpips']<=b['content']['lpips']+.005 and
                 a['content']['ink_iou']>=b['content']['ink_iou']-.01)
        style=(a['writer']['margin_mean']>=b['writer']['margin_mean']+.005 and
               a['writer']['macro_top1']>=b['writer']['macro_top1'] and
               da['n_cells']>=6 and db['n_cells']>=6 and da['lo']>0 and
               da['cosine']>=db['cosine']+.03)
        return content and style
    if improves('J7','J6'):return ['J6','J7']
    if improves('J6','J5') and improves('J6','J4'):return ['J4','J6']
    if improves('J5','J4'):return ['J4','J5']
    return []
