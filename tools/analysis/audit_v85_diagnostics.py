"""Audit archived v85 outputs, brush-reference sensitivity and run provenance."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.style import bank as B
from src.eval.style.brush import features_from_image, aggregate, bsd

RUNS = {
    'v84A': ('v84A_std_cond/20261013-072215-v84A-std-cond', 200000),
    'v85A': ('v85A_style_ada/20261014-032924-v85A-style-ada', 30000),
    'v85C': ('v85C_critic/20261014-132111-v85C-critic-writer-away', 30000),
    'v85E15': ('v85E_critic/20261014-160601-v85E-critic-heavy', 15000),
    'v85E30': ('v85E_critic/20261014-160601-v85E-critic-heavy', 30000),
    'control10': ('v85_control/20261014-062124-v85-control', 10000),
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out = ROOT / 'assets/v85_diagnostics_20261014'
    out.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(ROOT/'exp-std/csv/eval200_fixed.csv', encoding='utf-8-sig')))
    result = {'metric_version': 'brush-v1-thresh055-width2edt-q51-pooled',
              'brush_source_sha256': sha(ROOT/'src/eval/style/brush.py'), 'runs': {}, 'references': {}}
    feats = {}
    for key, (run, step) in RUNS.items():
        rp = ROOT/'assets/runs'/run
        cfgp = rp/'resolved_config.json'
        protp = rp/'eval_protocol'/f'step{step:07d}_eval200fix.json'
        imgdir = rp/'eval_samples_ctrl'/f'step{step:07d}'/'eval200fix'
        paths = [imgdir/f'g{i}.png' for i in range(len(rows))]
        if not all(p.exists() for p in paths):
            result['runs'][key] = {'missing_images': sum(not p.exists() for p in paths), 'path': str(imgdir)}
            continue
        ff = [features_from_image(str(p)) for p in paths]
        feats[key] = aggregate(ff)
        per = [dict(idx=i, **f.width_stats(), ink=f.scalars.get('ink_frac'),
                    edge_mean=float(f.edge.mean()) if f.edge.size else None,
                    sha256=sha(paths[i])) for i,f in enumerate(ff)]
        result['runs'][key] = {'path': str(imgdir), 'n': len(paths), 'per_image': per,
                              'pooled_width': feats[key].width_stats(),
                              'mean_image_width_p50': float(np.nanmean([f.width_stats()['p50'] for f in ff])),
                              'mean_ink': float(np.mean([f.scalars.get('ink_frac',0) for f in ff])),
                              'config': json.loads(cfgp.read_text()), 'config_sha256': sha(cfgp),
                              'protocol': json.loads(protp.read_text()) if protp.exists() else None}
        print('IMAGES', key, result['runs'][key]['mean_image_width_p50'], result['runs'][key]['mean_ink'], flush=True)
    bank = B.build_bank(str(ROOT/'exp-std/csv/train.csv'), roots=[str(ROOT),str(ROOT/'data'),str(ROOT/'exp-std')])
    cache = {}
    for cap, refcap in [(3000,1000),(3000,1200),(4000,1200)]:
        rr = B.cap_balanced(bank, cap)
        ri, qi = B.split_index_by_work(rr)
        ref = [rr[i] for i in ri]
        groups = defaultdict(list)
        for r in ref: groups[r.slot].append(r)
        quota = max(1, refcap//len(groups))
        picks = [r for slot in sorted(groups) for r in groups[slot][:quota]]
        for r in picks:
            if r.path not in cache: cache[r.path] = features_from_image(r.path)
        fref = aggregate([cache[r.path] for r in picks])
        result['references'][f'{cap}_{refcap}'] = {'bank_n':len(rr),'gallery_n':len(ri),'query_n':len(qi),
            'bsd_reference_n':len(picks),'reference_paths_sha256':hashlib.sha256('\n'.join(r.path for r in picks).encode()).hexdigest(),
            'reference_width':fref.width_stats(), 'distances':{k:bsd(f,fref)['width'] for k,f in feats.items()}}
        print('REFERENCE', cap,refcap,result['references'][f'{cap}_{refcap}'],flush=True)
    log = ROOT/'logs/v85E_critic_160558.log'
    probes=[]
    for line in log.read_text(errors='replace').splitlines():
        if '[critic-probe]' not in line: continue
        nums = re.search(r'dL_FM/dθ‖=([\deE+.\-]+).*dL_writer/dθ‖=([\deE+.\-]+).*dL_away/dθ‖=([\deE+.\-]+)',line)
        if nums:
            fm,w,a=map(float,nums.groups()); probes.append([w/fm,a/fm])
    ar=np.array(probes)
    result['gradient_probe']={'n':len(probes),'writer_weighted_percent_quantiles':np.percentile(ar[:,0]*.02*100,[10,50,90]).tolist(),
        'away_weighted_percent_quantiles':np.percentile(ar[:,1]*.05*100,[10,50,90]).tolist(),
        'writer_weight_for_median_25percent':float(.25/np.median(ar[:,0])),
        'away_weight_for_median_10percent':float(.1/np.median(ar[:,1])),
        'writer_weight_for_mean_25percent':float(.25/np.mean(ar[:,0])),
        'away_weight_for_mean_10percent':float(.1/np.mean(ar[:,1])),
        'writer_mean_ratio':float(np.mean(ar[:,0])), 'away_mean_ratio':float(np.mean(ar[:,1]))}
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',out, result['gradient_probe'],flush=True)

if __name__=='__main__':main()
