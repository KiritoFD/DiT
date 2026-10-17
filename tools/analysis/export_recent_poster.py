"""Snapshot existing eval PNGs and their metric JSON; never runs inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import sys
from PIL import Image


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--selected',type=int,nargs='+')
    parser.add_argument('--measure',action='store_true')
    parser.add_argument('--metric-code-root',default='')
    args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    work=Path(tempfile.mkdtemp(prefix='dit_poster_'))
    result={}
    if args.measure:
        sys.path.insert(0,args.metric_code_root)
        import numpy as np
        import torch,lpips
        from src.eval.metrics import ssim,frag_ratio
        from src.eval.metrics_ink import ink_iou
        torch.set_num_threads(4);torch.set_grad_enabled(False)
        perceptual=lpips.LPIPS(net='alex',verbose=False).eval()
    for row in manifest['rows']:
        if row.get('host')!=args.host:continue
        source=Path(row['directory']);target=work/row['key'];target.mkdir()
        hashes={};gt_hashes={}
        for i in range(187):
            p=source/f'g{i}.png'
            if args.selected is None or i in args.selected:shutil.copy2(p,target/p.name)
            hashes[str(i)]=digest(p)
            gt=source/f'gt{i}.png'
            if gt.exists():
                with Image.open(gt) as im:
                    gt_hashes[str(i)]=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
                if row['key']=='v84A':shutil.copy2(gt,target/gt.name)
        record=dict(row,image_sha256=hashes,gt_pixel_sha256=gt_hashes)
        if args.measure:
            measured=[];preds=[];truth=[]
            for i in range(187):
                p=np.asarray(Image.open(source/f'g{i}.png').convert('RGB'),dtype=np.float32)/255
                g=np.asarray(Image.open(source/f'gt{i}.png').convert('RGB'),dtype=np.float32)/255
                preds.append(p);truth.append(g)
                measured.append(dict(idx=i,ssim=ssim(p,g),ink_iou=ink_iou(p,g),frag=frag_ratio(p,g)))
            for start in range(0,187,8):
                p=torch.from_numpy(np.stack(preds[start:start+8])).permute(0,3,1,2)*2-1
                g=torch.from_numpy(np.stack(truth[start:start+8])).permute(0,3,1,2)*2-1
                for j,value in enumerate(perceptual(p,g).flatten().tolist()):measured[start+j]['lpips_alex']=value
            record['unified_metrics']=dict(n=187,per_image=measured,
                mean={k:float(np.mean([r[k] for r in measured])) for k in ('ssim','lpips_alex','ink_iou','frag')},
                metric_source_sha256={k:digest(Path(args.metric_code_root)/'src/eval'/k) for k in ('metrics.py','metrics_ink.py')})
            print('MEASURED',row['key'],record['unified_metrics']['mean'],flush=True)
        if row.get('metric_file'):
            record['native_metrics']=json.loads(Path(row['metric_file']).read_text())
            assert record['native_metrics']['n']==187
            assert record['native_metrics']['step']==row['step']
        result[row['key']]=record
        print('SNAPSHOT',row['key'],187,flush=True)
    (work/'snapshot.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with tarfile.open(args.out,'w') as archive:
        for path in work.iterdir():archive.add(path,arcname=path.name)
    print('ARCHIVE',args.out,flush=True)


if __name__=='__main__':main()
