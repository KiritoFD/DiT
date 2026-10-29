"""Final analysis: paired bootstrap across arms on matched val100 characters,
assemble FINAL_REPORT.md from all pilot artifacts."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from .common import PILOT_ROOT,TMP_ROOT


def per_char_ssim(dir_a,dir_b):
    """SSIM between same-index generated PNGs of two arms (val100 panels)."""
    pa=sorted(Path(dir_a).glob('g*.png'));pb=sorted(Path(dir_b).glob('g*.png'))
    n=min(len(pa),len(pb));out=[]
    for i in range(n):
        a=np.asarray(Image.open(pa[i]).convert('L'),dtype=np.float32)/255.
        b=np.asarray(Image.open(pb[i]).convert('L'),dtype=np.float32)/255.
        out.append(ssim_fn(a,b,data_range=1.))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arms',nargs='+',default=['A','B','C'])
    a=p.parse_args()
    report=[]
    # val FM trajectory summary
    curves=PILOT_ROOT/'train_arm_curves.csv'
    if curves.is_file():
        import csv as csvmod
        rows=list(csvmod.DictReader(open(curves,encoding='utf-8-sig')))
        for arm in a.arms:
            vf=[r for r in rows if r['arm']==arm and r.get('val_fm')]
            if vf:report.append(dict(section='val_fm',arm=arm,
                points=[(r['step'],r['val_fm']) for r in vf]))
    # paired SSIM between each arm and control A on val100 panels
    gen_base=TMP_ROOT/'valgen'
    for arm in a.arms:
        da=gen_base/f'{arm}_seed0';db=gen_base/'A_seed0'
        if da.is_dir() and db.is_dir() and arm!='A':
            ss=per_char_ssim(da,db)
            report.append(dict(section='paired_ssim_vs_A',arm=arm,
                mean=round(float(np.mean(ss)),4),n=len(ss),
                ci=[round(float(np.percentile(ss,2.5)),4),round(float(np.percentile(ss,97.5)),4)]))
    (PILOT_ROOT/'analysis_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=1)[:1200])


if __name__=='__main__':main()
