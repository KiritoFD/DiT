"""Identity-preserving export and source-split audit for the actual ResNet CE checkpoint."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from tools.three_day.contracts import read,save,sha,TOP10

NAMES={53:'何绍基',365:'文徵明',461:'柳公权',479:'欧阳询',601:'王羲之',672:'米芾',703:'苏轼',760:'褚遂良',956:'颜真卿',9001:'赵孟頫'}


def export(root):
    from src.unet.io import checkpoint_config,eval_args
    from src.eval.in_mem_eval import _get_cache
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    base=Path('assets/p2_audit_20261025/cfg');rows=json.loads((base/'cfg1.0/samples.json').read_text())
    identity=json.loads((base/'cfg1.0/identity.json').read_text())
    ck=torch.load(identity['checkpoint'],map_location='cpu',weights_only=False,mmap=True);c=checkpoint_config(ck)
    c.eval_csv=str(base/'cfg1.0/eval.csv');cache=_get_cache(c.eval_csv,len(rows),None,'',eval_args(c))
    torch.save(dict(g=cache['skels_latent'].cpu(),noise=cache['noise'].cpu()),root/'common_inputs.pt')
    normalized=[];inverse={v:k for k,v in TOP10.items()}
    for i,r in enumerate(rows):
        raw=int(r.get('raw_writer_id') or inverse[int(r['calligrapher_id'])])
        if NAMES[raw]!=r['calligrapher'] or TOP10[raw]!=int(r['calligrapher_id']):raise ValueError('P2 semantic label mismatch')
        normalized.append(dict(r,raw_writer_id=raw,writer=r['calligrapher'],char=r['character'],idx=i))
    for name,folder in [('P2_CFG1','cfg1.0'),('P2_CFG1.6','cfg1.6')]:
        rr=json.loads((base/folder/'samples.json').read_text())
        if [(r['character'],r['script'],r['calligrapher_id']) for r in rr]!=[(r['character'],r['script'],r['calligrapher_id']) for r in rows]:
            raise ValueError('P2 paired rows differ')
        target=root/name;target.mkdir(exist_ok=True)
        for i in range(len(rows)):shutil.copy2(base/folder/'samples'/f'g{i}.png',target/f'g{i}.png')
        shutil.copy2(base/folder/'latents.npy',root/(name+'_latents.npy'))
        if name=='P2_CFG1':
            gt=root/'GT';gt.mkdir(exist_ok=True)
            for i in range(len(rows)):shutil.copy2(base/folder/'samples'/f'gt{i}.png',gt/f'g{i}.png')
    swap=base/'swap';ss=json.loads((swap/'samples.json').read_text());target=root/'P2_swap';target.mkdir(exist_ok=True)
    for i,r in enumerate(ss):
        raw=inverse[int(r['calligrapher_id'])]
        ss[i]=dict(r,raw_writer_id=raw,writer=NAMES[raw],char=r['character'],idx=i)
        shutil.copy2(swap/'samples'/f'g{i}.png',target/f'g{i}.png')
    save(root/'swap_rows.json',ss);save(root/'rows.json',normalized)
    save(root/'export.json',dict(p2=identity,n=len(rows),common_input_sha256=sha(root/'common_inputs.pt'),
        gt_png='exact P2 evaluation GT; no new preprocessing',swap_identity='writer inferred from checkpoint row mapping, not stale copied name'))


def real_bank(root):
    from src.eval.style.bank import work_id,resolve
    root=Path(root);ck=torch.load('assets/style_disc_ce_holdout.pt',map_location='cpu',weights_only=False)
    rows=[];hold=set(ck['holdout_works'])
    for r in read('exp-std/csv/train_top10_aug_sym.csv'):
        if (r.get('aug') or '').strip():continue
        path=resolve(r['image_path'],['.','data','exp-std'])
        if not path:raise FileNotFoundError(r['image_path'])
        work=work_id(r.get('src_image_path',''),r['image_path'])
        rows.append(dict(r,path=path,writer=r['calligrapher'],char=r['character'],work=work,source_heldout=work in hold))
    train_chars={r['char'] for r in rows if not r['source_heldout']}
    for r in rows:r['character_heldout']=r['char'] not in train_chars
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    def pixel_hash(r):
        with Image.open(r['path']) as im:return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r,h in zip(rows,pool.map(pixel_hash,rows)):r['pixel_hash']=h
    seen_hashes={r['pixel_hash'] for r in rows if not r['source_heldout']}
    for r in rows:r['dedup_source_heldout']=r['source_heldout'] and r['pixel_hash'] not in seen_hashes
    old=[];by_group={}
    for i,r in enumerate(rows):
        key=(r['writer'],r['script']);by_group.setdefault(key,[]).append(i)
    for key,indices in sorted(by_group.items()):old.extend(indices[:120])
    save(root/'teacher_identity.json',dict(checkpoint='assets/style_disc_ce_holdout.pt',sha256=sha('assets/style_disc_ce_holdout.pt'),
        architecture='ResNet18 -> Linear512/256 -> L2 embedding; CE classification used unnormalized pre-embedding',
        n_classes=ck['n_cls'],slots=ck['all_keys'],saved_train_accuracy=ck['acc_train'],saved_holdout_accuracy=ck['acc_eval'],
        holdout_works=sorted(hold),original_GT_sampling=dict(n=len(old),n_source_heldout=sum(rows[i]['source_heldout'] for i in old)),
        pca_original='ARI on full 256D for GT; comparison PCA mixed GT+generated; Qwen ARI/NMI AND displayed DINO ARI were hardcoded, not computed',
        overlap_evidence='Reconstructed from saved holdout_works and available original CSV; checkpoint did not save input hashes or exact training image IDs',
        work_caveat='work_id is a source-directory proxy, not verified artwork identity',
        current_csv_sha256=sha('exp-std/csv/train_top10_aug_sym.csv')))
    save(root/'real_rows.json',rows);save(root/'original_GT_indices.json',old)
    return rows


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--export',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    if a.export:export(a.root)
    else:real_bank(a.root)
