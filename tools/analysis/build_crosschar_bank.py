"""Cache teacher TRAIN features for losses only. No generator condition export."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.model.style_encoder import build_style_encoder
from src.utils.latent_dataset import extract_img_id
from src.eval.style.bank import work_id
from tools.train_prototype_style_encoder import sha
from tools.analysis.prepare_v89_crosschar_pilot import teacher_gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-run',required=True)
    p.add_argument('--data',default='scratch/disc_lat_full.npz')
    p.add_argument('--generator-csv',default='exp-std/csv/train_top10_aug_sym.csv')
    p.add_argument('--callig-remap',default='assets/triple_tables_best_minimal/callig_remap.json')
    p.add_argument('--font-remap',default='assets/triple_tables_best_minimal/font_remap.json')
    p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    run=Path(a.teacher_run)
    matched=json.loads((run/'matched_teacher.json').read_text())
    accepted=json.loads((run/'acceptance.json').read_text())
    failures=teacher_gate(matched,accepted)
    if failures:
        raise ValueError('Teacher has not passed the matched-discrimination gate: '+str(failures))
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    manifest=json.loads((run/'manifest.json').read_text())
    assert sha(a.data)==manifest['data_sha256']
    split=np.load(run/'split_indices.npz')
    train=split['train']
    assert all(not set(train)&set(split[k]) for k in ('val_char','test_char','test_source'))
    d=np.load(a.data,allow_pickle=True)
    sr=d['std_idx'].astype(np.int64)
    z=torch.from_numpy(d['z'].copy());g=1-torch.from_numpy(d['g_std'].copy()).float()[:,None]/255
    cm={int(k):int(v) for k,v in json.loads(Path(a.callig_remap).read_text()).items()}
    sm={int(k):int(v) for k,v in json.loads(Path(a.font_remap).read_text()).items()}
    rows=list(csv.DictReader(open(a.generator_csv,encoding='utf-8-sig')))
    wn={r['calligrapher'].strip():cm[int(r['calligrapher_id'])] for r in rows}
    sn={r['script'].strip():sm[int(r['script_id'])] for r in rows}
    assert len(set(wn.values()))==len(wn) and len(set(sn.values()))==len(sn)
    target={}
    for r in rows:
        key=extract_img_id(r)
        metadata=(wn[r['calligrapher'].strip()],sn[r['script'].strip()],r['character'].strip(),
                  work_id(r.get('src_image_path',''),r.get('image_path','')))
        if key in target and target[key]!=metadata:
            raise ValueError(f'Conflicting metadata for image ID {key}')
        target[key]=metadata
    all_scripts=d['script'].astype(str)
    si=np.array([ck['scripts'].index(str(s)) for s in all_scripts])
    model=build_style_encoder(len(ck['slots']),len(ck['scripts']),ck.get('architecture','legacy')).cuda().eval()
    model.load_state_dict(ck['model'],strict=True)
    features=[]
    with torch.no_grad():
        for start in range(0,len(train),256):
            ii=train[start:start+256]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                value=model(z[ii].cuda().float(),g[sr[ii]].cuda(),torch.from_numpy(si[ii]).cuda())
            features.append(value['embedding'].float().cpu())
    bank={'features':torch.cat(features),'partition':'encoder_train_only',
          'teacher_accepted':True,'matched_teacher_sha256':sha(run/'matched_teacher.json'),
          'encoder_sha256':sha(run/'best.pt'),'data_sha256':sha(a.data),'split_sha256':sha(run/'split_indices.npz'),
          'reference_indices':train,'target_metadata':target,
          'writers':np.array([wn[str(w)] for w in d['writer'][train]]),
          'scripts':np.array([sn[str(s)] for s in d['script'][train]]),
          'chars':d['char'][train].astype(str),'sources':d['work'][train].astype(str),
          'encoder_script_ids':torch.tensor([ck['scripts'].index(s) for s,i in sorted(sn.items(),key=lambda x:x[1])]),
          'generator_csv_sha256':sha(a.generator_csv),
          'usage':'training loss only; never attach to generator or inference checkpoint'}
    assert len(bank['encoder_script_ids'])==len(sn)
    output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(bank,output)
    print('CROSSCHAR_BANK_COMPLETE',len(train),'reference rows;',len(target),'target metadata rows;',output,flush=True)


if __name__=='__main__':
    main()
