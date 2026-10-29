"""Export small statistics and indexed scalar decisions, never images or checkpoints."""
import argparse
from pathlib import Path
import numpy as np
from tools.eval_unpaired.protocol import read,save,csv_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();bundle=[]
    for name in a.roots:
        root=Path(name)
        if not (root/'summary.json').exists():continue
        result=dict(root=str(root),complete=(root/'complete.json').exists(),
            summary=read(root/'summary.json'),calibration=read(root/'quality_calibration.json') if (root/'quality_calibration.json').exists() else None,
            provenance=read(root/'networks.json') if (root/'networks.json').exists() else None,decisions=[])
        for item in result['summary']:
            j=item['job'];folder=root/f"{j['arm']}_{j['step']}_{j['protocol']}"
            for candidate in [Path(j['samples']).parents[2]/'result.json',Path(j['samples'])/'metrics.json']:
                if candidate.exists():item['paired_content']=read(candidate);break
            if not folder.is_dir():continue
            entry=dict(arm=j['arm'],step=j['step'],protocol=j['protocol'],csv_sha256=j['csv_sha256'],
                rows=[{k:r.get(k) for k in ['calligrapher','script','character','eval_noise_seed']} for r in csv_rows(j['csv'])])
            protocol=Path(j['samples'])/'protocol.json'
            if protocol.exists():entry['generation_protocol']=read(protocol)
            for metric in ['E-G','E-L']:
                path=folder/f'g_{metric}_K64_rows.json'
                if path.exists():entry[metric]=[{k:d.get(k) for k in ['valid','top1','margin','afc','writer','character']} for d in read(path)['rows']]
            path=folder/'g_features.npz'
            if path.exists():
                data=np.load(path);entry['quality']={k:v.tolist() for k,v in data.items() if not k.startswith('E-')}
            result['decisions'].append(entry)
        bundle.append(result)
    save(a.out,bundle)


if __name__=='__main__':main()
