"""Independent cross-character verification in frozen HWD features (not HWD score)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from tools.eval_unpaired.protocol import read, save, csv_rows
from tools.eval_unpaired.networks import identity
from tools.eval_unpaired.domain.statistics import verification
from .aesthetic import features
from .model import load
from PIL import Image


def unit(x): return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def main():
    p=argparse.ArgumentParser();p.add_argument('--hwd-root',required=True)
    p.add_argument('--reference-rows',required=True);p.add_argument('--weights',required=True)
    a=p.parse_args();torch.set_num_threads(2);root=Path(a.hwd_root)
    refs=read(a.reference_rows);cache=root/'reference_square_features.npy'
    contract=dict(reference=identity(a.reference_rows),weights=identity(a.weights),
        images=[identity(r['image_path']) for r in refs],
        preprocessing='RGB aspect-preserving white square pad, nearest32, raw VGG; cosine verification only')
    manifest=root/'verification_inputs.json'
    if manifest.exists() and read(manifest)!=contract:
        raise ValueError('HWD verification inputs changed')
    save(manifest,contract)
    if not cache.exists():
        net=load(a.weights)
        # Original reference crops are padded to squares before height32. This is
        # an explicit adaptation used only for the extra cosine verification.
        np.save(cache,features(net,[Image.open(r['image_path']).convert('RGB') for r in refs]))
    bank=unit(np.load(cache));results=[]
    for item in read(root/'summary.json'):
        j=item['job'];name='_'.join(map(str,[j['arm'],j['step'],j['protocol']]))
        scores=dict(job=j)
        for prefix in ['g','gt']:
            v=verification(unit(np.load(root/name/f'{prefix}.npy')),bank,csv_rows(j['csv']),refs)
            save(root/name/f'{prefix}_verification.json',v)
            scores[prefix]={k:v for k,v in v.items() if k!='rows'}
        results.append(scores)
        print('HWD_VERIFY_COMPLETE',name,scores['g'].get('macro_writer'),flush=True)
    save(root/'verification_summary.json',results)


if __name__=='__main__': main()
