"""Same-character/script GT-relative Writer direction audit; CPU, no learned judge."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
from skimage.morphology import skeletonize
from tools.p2_audit.common import read,save,paired_ci
from tools.u100_pretrain.eval_offline import TOP10_RAW
from tools.u100_pretrain.eval_round1 import ridge_widths

METRICS=('ink','soft_ink','width50','width90','aspect','bbox_area','density','center_x','center_y')


def features(path):
    im=Image.open(path).convert('L').resize((256,256),Image.Resampling.LANCZOS)
    gray=np.asarray(im,dtype=np.float32)/255;ink=1-gray;mask=gray<.5
    ys,xs=np.where(mask)
    width=ridge_widths(im)
    if len(ys):
        h,w=int(ys.max()-ys.min()+1),int(xs.max()-xs.min()+1)
    else:h,w=1,1
    yy,xx=np.mgrid[:256,:256];mass=max(float(ink.sum()),1e-8)
    small=np.asarray(im.resize((64,64),Image.Resampling.LANCZOS))<128
    skel=skeletonize(small)
    return dict(ink=float(mask.mean()),soft_ink=float(ink.mean()),
        width50=float(np.median(width)) if width is not None else 0.,
        width90=float(np.quantile(width,.9)) if width is not None else 0.,
        aspect=w/h,bbox_area=w*h/65536,density=float(mask.sum())/(w*h),
        center_x=float((ink*xx).sum()/mass/255),center_y=float((ink*yy).sum()/mass/255),
        blank=not len(ys),mask=small,skel=skel)


def chamfer(a,b):
    if not a.any() or not b.any():return 0. if not a.any() and not b.any() else 1.
    return float((distance_transform_edt(~a)[b].mean()+distance_transform_edt(~b)[a].mean())/128)


def main():
    p=argparse.ArgumentParser();p.add_argument('--swap',default='assets/p2_audit_20261025/cfg/swap')
    p.add_argument('--train',default='assets/u100_p1p2_20261025/p2_train.csv')
    p.add_argument('--extra',nargs='*',default=['assets/u100_pretrain_20261024/validation_val100.csv','assets/generic48_kxl_top10_20261022/top10_test.csv'])
    p.add_argument('--out',default='assets/p2_std_consolidation_20261025/direction');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);swap=Path(a.swap)
    generated=json.loads((swap/'samples.json').read_text());wanted={(r['script'],r['character']) for r in generated}
    pools=defaultdict(list);seen=set()
    for source in [a.train]+a.extra:
        rr=read(source);ids={int(r['calligrapher_id']) for r in rr}
        raw=ids<=set(TOP10_RAW)
        if not raw and not ids<=set(range(10)):raise ValueError('Mixed Writer label space: '+source)
        for r in rr:
            if (r['script'],r['character']) not in wanted or r.get('aug','').strip():continue
            writer=TOP10_RAW[int(r['calligrapher_id'])] if raw else int(r['calligrapher_id'])
            uid=r.get('pixel_sha256') or r.get('img_id') or r['image_path']
            if uid in seen:continue
            seen.add(uid);path=Path(r['image_path'])
            if not path.exists():raise FileNotFoundError(path)
            pools[r['script'],r['character'],writer].append(dict(row=r,path=str(path),source=source,training=source==a.train))
    paths=sorted({v['path'] for vv in pools.values() for v in vv}|{str(swap/'samples'/f'g{i}.png') for i in range(len(generated))})
    with ThreadPoolExecutor(max_workers=4) as ex:cache=dict(zip(paths,ex.map(features,paths)))
    records=[];bygroup=defaultdict(list)
    for i,r in enumerate(generated):
        key=(r['script'],r['character'],int(r['calligrapher_id']));real=pools.get(key)
        if not real:continue
        fg=cache[str(swap/'samples'/f'g{i}.png')]
        gt={k:float(np.mean([cache[v['path']][k] for v in real])) for k in METRICS}
        record=dict(idx=i,script=key[0],char=key[1],writer=key[2],n_gt=len(real),n_train_gt=sum(v['training'] for v in real),
                    gt=gt,gen={k:fg[k] for k in METRICS},gt_paths=[v['path'] for v in real],gt_sources=[v['source'] for v in real])
        records.append(record);bygroup[key[:2]].append(record)
    summary={};per_writer=defaultdict(lambda:defaultdict(list));pair_records=[]
    for metric in METRICS:
        diffs=[];groups=[];gt_d=[];gen_d=[]
        for group,rr in bygroup.items():
            if len(rr)<2:continue
            gt_mean=np.mean([r['gt'][metric] for r in rr]);gen_mean=np.mean([r['gen'][metric] for r in rr])
            for r in rr:
                per_writer[r['writer']][metric].append(dict(gt=r['gt'][metric]-gt_mean,gen=r['gen'][metric]-gen_mean))
            for x,y in combinations(rr,2):
                dg=x['gt'][metric]-y['gt'][metric];dp=x['gen'][metric]-y['gen'][metric]
                if abs(dg)<=1e-4:continue
                agreement=float(dg*dp>0);diffs.append(agreement);groups.append(str(group));gt_d.append(dg);gen_d.append(dp)
                pair_records.append(dict(script=group[0],char=group[1],metric=metric,wa=x['writer'],wb=y['writer'],delta_gt=dg,delta_gen=dp,correct_direction=bool(agreement)))
        rho=float(spearmanr(gt_d,gen_d).statistic) if len(gt_d)>2 and np.std(gen_d)>0 else None
        summary[metric]=dict(direction=paired_ci(diffs,groups) if diffs else None,spearman=rho,
            macro_cell_accuracy=float(np.mean([np.mean(np.array(diffs)[np.array(groups)==g]) for g in set(groups)])) if diffs else None)
    shape=[]
    for group,rr in bygroup.items():
        if len(rr)<2:continue
        for r in rr:
            fg=cache[str(swap/'samples'/f'g{r["idx"]}.png')];pred={}
            for kind in ('mask','skel'):
                scores=[]
                for candidate in rr:
                    refs=sorted(candidate['gt_paths'])[:4]
                    scores.append(float(np.mean([chamfer(fg[kind],cache[path][kind]) for path in refs])))
                pred[kind]=int(rr[int(np.argmin(scores))]['writer'])
            shape.append(dict(script=group[0],char=group[1],writer=r['writer'],mask_pred=pred['mask'],skeleton_pred=pred['skel'],chance=1/len(rr)))
    writer_summary={str(w):{k:dict(n=len(v),gt_centered=float(np.mean([x['gt'] for x in v])),gen_centered=float(np.mean([x['gen'] for x in v]))) for k,v in metrics.items()} for w,metrics in per_writer.items()}
    save(out/'results.json',dict(metrics=summary,by_writer=writer_summary,matched_cells=len(records),total_generated=len(generated),
        cells_with_train_gt=sum(r['n_train_gt']>0 for r in records),n_gt_images=sum(r['n_gt'] for r in records),
        shared_char_script_cells=sum(len(v)>=2 for v in bygroup.values()),
        shape_only=dict(n=len(shape),mask_accuracy=float(np.mean([r['mask_pred']==r['writer'] for r in shape])) if shape else None,
                        skeleton_accuracy=float(np.mean([r['skeleton_pred']==r['writer'] for r in shape])) if shape else None,
                        mean_chance=float(np.mean([r['chance'] for r in shape])) if shape else None),
        caveat='Existing 12-character, one-seed swap at CFG1.6; partial GT coverage. GT train+validation+test used descriptively, not an independent final test. Skeleton Chamfer is width-reduced shape matching, not a learned brush-style metric.',
        samples=records,pairs=pair_records,shape_samples=shape))
    # Every available matched GT / generation pair, with missing truth shown gray.
    chars=list(dict.fromkeys((r['script'],r['character']) for r in generated))
    grid=Image.new('RGB',(1600,32+len(chars)*320),'#cccccc');draw=ImageDraw.Draw(grid)
    for w in range(10):draw.text((160*w+3,8),f'W{w}: GT / GEN',fill='black')
    for r in records:
        row=chars.index((r['script'],r['char']));x=160*r['writer'];y=32+row*320
        representative=sorted(r['gt_paths'])[0]  # Not nearest-neighbor selection on the generation.
        for offset,path in [(0,representative),(160,swap/'samples'/f'g{r["idx"]}.png')]:
            grid.paste(Image.open(path).convert('RGB').resize((160,160)),(x,y+offset))
    grid.save(out/'matched_gt_gen.png');print('DIRECTION_DONE',len(records),len(shape),flush=True)


if __name__=='__main__':main()
