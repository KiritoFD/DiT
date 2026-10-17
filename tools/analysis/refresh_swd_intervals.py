"""Recompute interval summaries only, from saved immutable distance matrices."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.swd_style_stats import summarize_distances


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    report=json.loads((out/'results.json').read_text(encoding='utf-8'))
    plans=json.loads((out/'trials.json').read_text(encoding='utf-8'))
    for name,method in report['methods'].items():
        matrices=np.load(out/f'{name}_distances.npz')
        for protocol in ('test_char','test_source'):
            cohort={(t['w'],t['v']) for t in plans[f'{protocol}_32']}
            for k in (1,4,16,32):
                key=f'{protocol}_{k}';ts=plans[key];d=matrices[key];ar=np.arange(len(ts))
                w=np.array([t['w'] for t in ts],int);v=np.array([t['v'] for t in ts],int)
                within=np.stack((d[ar,0,w],d[ar,1,v]),1);between=np.stack((d[ar,0,v],d[ar,1,w]),1)
                old=method['sets'][protocol][str(k)];fresh=summarize_distances(within,between,ts)
                old.update(fresh)
                keep=np.array([(t['w'],t['v']) in cohort for t in ts])
                old['common32_cohort']=summarize_distances(within[keep],between[keep],[t for t,b in zip(ts,keep) if b])
        print('SWD_INTERVALS_REFRESHED',name,flush=True)
    script=Path(__file__).with_name('swd_style_stats.py')
    report['interval_postprocess_sha256']=hashlib.sha256(script.read_bytes()).hexdigest()
    report['decision']={
        'frozen_swd_accepted_for_dit':False,
        'reason':'A/B/C fail source-holdout set discrimination and interval gates; no single-glyph or batch-level SWD loss is authorized by these results.',
        'next_action':'Stop expanding this frozen-feature SWD statistical route. Task-trained D1 remains a candidate for separate generation/gradient validation, not an automatically accepted DiT loss.',
        'global_permutation':'Insufficient common-character coverage for high-coverage global relabeling; do not claim a valid global-permutation significance result.'}
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
