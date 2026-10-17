"""Collect existing eval curves against nominal image exposure; no training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tools.analysis.capacity_audit.common import write, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='append',required=True,help='label=directory')
    p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    result=[]
    for spec in a.run:
        label,directory=spec.split('=',1);run=Path(directory)
        cfg=json.loads((run/'resolved_config.json').read_text(encoding='utf-8'))
        batch=int(cfg['global_batch_size'])
        for path in sorted((run/'checkpoints').glob('eval_auto_*.json')):
            m=json.loads(path.read_text(encoding='utf-8'))
            if m.get('n')!=187 or not m.get('step'):continue
            result.append({'model':label,'step':m['step'],'batch':batch,
                'nominal_images':batch*m['step'],'ssim':m.get('ssim'),'lpips':m.get('lpips'),
                'frag':m.get('frag'),'source':str(path),'source_sha256':sha(path)})
    write(out/'curves.json',{'rows':result,'limitations':[
        'Exposure=step*configured global batch, assumes batch unchanged across resume history.',
        'Optimizer phase, LR schedule and metric-code versions are not normalized.',
        'No interpolated values or causal capacity attribution.']})
    exact=[]
    for i,x in enumerate(result):
        for y in result[i+1:]:
            if x['model']!=y['model'] and x['nominal_images']==y['nominal_images']:
                exact.append({'images':x['nominal_images'],'a':x,'b':y})
    write(out/'exact_exposure_pairs.json',exact)
    print('curves',len(result),'exact exposure pairs',len(exact),flush=True)


if __name__=='__main__':main()
