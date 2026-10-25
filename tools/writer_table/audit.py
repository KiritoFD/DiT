"""Audit independent table/data contract and export canonical, generator-independent rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.cohort_eval.datasets import read_rows,image_key,sha


def run(args):
    source=Path(args.source);data=Path(args.data);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    provenance=json.loads((source/'provenance.json').read_text())
    train=read_rows(data/'train.csv');val=read_rows(data/'validation.csv');test=read_rows(data/'test.csv')
    assert provenance['training_csv_sha256']==sha(data/'train.csv')
    assert provenance['validation_csv_sha256']==sha(data/'validation.csv')
    assert provenance['test_used'] is False
    assert not {image_key(r) for r in train}&{image_key(r) for r in val+test}
    mapping=json.loads((source/'callig_remap.json').read_text())
    assert mapping==json.loads((data/'tables/callig_remap.json').read_text())
    raw=torch.from_numpy(np.load(source/'callig_table_raw.npy')).float()
    assert raw.shape==(20,32) and torch.isfinite(raw).all()
    assert torch.allclose(raw.norm(dim=1),torch.ones(20),atol=1e-5)
    z=np.load(source/'features.npz')
    assert z['paths'].tolist()==[image_key(r) for r in train+val]
    checkpoint=torch.load(source/'contrastive.pt',map_location='cpu',weights_only=False)
    assert torch.equal(checkpoint['table'],raw)
    projector=torch.nn.Sequential(torch.nn.Linear(z['feat'].shape[1],128),torch.nn.GELU(),torch.nn.Linear(128,32)).cuda().eval()
    projector.load_state_dict(checkpoint['projection'],strict=True)
    features=torch.from_numpy(z['feat']).float().cuda();table=raw.cuda()
    labels=torch.tensor([mapping[r['calligrapher_id']] for r in train+val],device='cuda')
    hits=[]
    with torch.inference_mode():
        for start in range(0,len(features),2048):
            emb=F.normalize(projector(F.normalize(features[start:start+2048],dim=-1)),dim=-1)
            hits.extend(((emb@table.T).argmax(1)==labels[start:start+2048]).cpu().tolist())
    cosine=raw@raw.T;off=~torch.eye(20,dtype=torch.bool)
    best=max(provenance['history'],key=lambda row:row['validation_top1'])
    audit=dict(n_train=len(train),n_validation=len(val),n_test=len(test),train_test_path_overlap=0,
        shape=[20,32],all_20_rows_independently_trained=True,train_top1=float(np.mean(hits[:len(train)])),
        validation_top1=float(np.mean(hits[len(train):])),selected_step=best['step'],
        offdiag_mean_cos=float(cosine[off].mean()),offdiag_max_cos=float(cosine[off].max()),
        min_row_norm=float(raw.norm(dim=1).min()),max_row_norm=float(raw.norm(dim=1).max()),
        export='raw normalized independently learned table; no generator-coordinate rotation or row-norm matching',
        caveats='Image-held-out validation, not strict character/work-held-out style validation; not a generator loss teacher')
    assert abs(audit['validation_top1']-best['validation_top1'])<1e-6
    output.mkdir(parents=True)
    np.save(output/'callig_table.npy',raw.numpy())
    (output/'callig_remap.json').write_text(json.dumps(mapping,indent=2))
    torch.save(dict(projection=checkpoint['projection'],table=raw),output/'projection.pt')
    canonical=dict(provenance)
    for key in ('alignment','coordinate_reference','coordinate_reference_sha256','normalized_geometry_max_error'):
        canonical.pop(key,None)
    canonical.update(source_independent_run=str(source),table_sha256=sha(output/'callig_table.npy'),
        export='raw unit-normalized rows; generator-independent',audit=audit)
    (output/'provenance.json').write_text(json.dumps(canonical,indent=2))
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','data','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
