"""Conservative generic pool: quality/label/STD gates, global protected near-duplicate guard."""
import argparse
from collections import Counter,defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity
from tools.cohort_eval.datasets import read_rows,write_rows,sha
from tools.data_clean.scan import extract


class NearIndex:
    """Nine 7-bit bands cover all <=8-bit changes in the 63-bit DCT fingerprint."""
    def __init__(self):self.bands=defaultdict(set);self.values=[]
    def add(self,record,thumb):
        idx=len(self.values);self.values.append((record,thumb))
        code=int(record['phash'])
        for band in range(9):self.bands[band,(code>>(7*band))&127].add(idx)
    def matches(self,record,thumb,threshold=.90):
        code=int(record['phash']);ids=set()
        for band in range(9):ids.update(self.bands.get((band,(code>>(7*band))&127),()))
        for idx in ids:
            other,image=self.values[idx]
            if (code^int(other['phash'])).bit_count()>8:continue
            if np.abs(thumb.astype(float)-image.astype(float)).mean()>255*.08:continue
            if structural_similarity(thumb,image,data_range=255)>=threshold:return True
        return False


def normalized(row,generic):
    row=dict(row);row['character_id']=str(ord(row['character']))
    row['glyph_id']=str(int(row['script_id'])*0x110000+ord(row['character']))
    row['generic']='1' if generic else '0'
    if generic:row['original_calligrapher_id']=row.get('calligrapher_id','');row['calligrapher_id']='-1';row['calligrapher']='UNKNOWN'
    return row


def select_diverse(items,n,seed):
    def rank(r):return hashlib.sha256(f'{seed}|{r["image_path"]}'.encode()).digest()
    buckets=defaultdict(list)
    for r in items:buckets[r['script_id'],r['character']].append(r)
    queues=[deque(sorted(v,key=rank)) for _,v in sorted(buckets.items())]
    picked=[]
    while queues and len(picked)<n:
        remaining=[]
        for queue in queues:
            if len(picked)==n:break
            picked.append(queue.popleft())
            if queue:remaining.append(queue)
        queues=remaining
    if len(picked)!=n:raise ValueError(f'Only {len(picked)} eligible images; need {n}; do not relax quality silently')
    return picked


def prepare(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);root=Path(args.root).resolve()
    top=Path(args.top20);source=read_rows(args.pool)
    std=Path(args.std_root);provenance=json.loads((std/'provenance.json').read_text(encoding='utf-8'))
    if 'fit256' not in provenance.get('geometry',''):raise ValueError('Use corrected bbox STD cache')
    index=read_rows(std/'index.csv');coverage={(r['script_id'],r['character']) for r in index}
    protected_files=[top/f'{name}.csv' for name in ('train','validation','test')]+[Path(p) for p in args.protected]
    protected_rows=[r for f in protected_files for r in read_rows(f)]
    actual=lambda p:str((root/p).resolve())
    protected_paths={actual(r[k]) for r in protected_rows for k in ('image_path','src_image_path','original_image_path') if r.get(k)}
    guard=NearIndex();protected_hashes=set()
    for path in sorted({actual(r['image_path']) for r in protected_rows}):
        record,thumb=extract(path)
        if record.get('error'):raise ValueError('Unreadable protected target: '+path)
        protected_hashes.add(record['pixel_sha256']);guard.add(record,thumb)
    reasons=Counter();rejected=[];eligible=[];seen=NearIndex();pixels={};conflicts=set();source_groups=set()
    scripts={'楷':'0','行':'3','隶':'4'}
    def inspect(row):
        issues=[]
        if row.get('aug','').strip():issues.append('augmented_row')
        if len(row.get('character',''))!=1:issues.append('invalid_character')
        if scripts.get(row.get('script'))!=str(row.get('script_id')):issues.append('script_mismatch_or_unsupported')
        if not args.trust_source_labels and str(row.get('label_verified','')).lower() not in ('1','true','yes'):issues.append('unverified_label')
        if (row.get('script_id'),row.get('character')) not in coverage:issues.append('missing_std')
        if any(actual(row[k]) in protected_paths for k in ('image_path','src_image_path','original_image_path') if row.get(k)):issues.append('protected_source_path')
        if issues:return row,None,None,issues
        record,thumb=extract(actual(row['image_path']))
        issues=list(record.get('hard_issues',[]))+list(record.get('flags',[]))
        return row,record,thumb,issues
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        def bounded():
            for start in range(0,len(source),512):yield from executor.map(inspect,source[start:start+512])
        for i,(row,record,thumb,issues) in enumerate(bounded()):
            if not issues:
                pixel=record['pixel_sha256'];label=(row['script_id'],row['character'])
                if pixel in pixels and pixels[pixel]!=label:conflicts.add(pixel)
                if pixel in protected_hashes:issues.append('protected_exact_pixels')
                elif guard.matches(record,thumb):issues.append('protected_near_duplicate')
                elif pixel in pixels or seen.matches(record,thumb,.985):issues.append('pool_duplicate')
                origin=actual(row.get('src_image_path') or row['image_path'])
                if origin in source_groups:issues.append('same_original_crop')
                if not issues:
                    pixels[pixel]=label;source_groups.add(origin);seen.add(record,thumb)
                    eligible.append(dict(normalized(row,True),pixel_sha256=pixel))
            if issues:
                reasons.update(issues);rejected.append(dict(image_path=row['image_path'],reason=';'.join(issues)))
            if (i+1)%5000==0:print('SCREENED',i+1,'eligible',len(eligible),flush=True)
    eligible=[r for r in eligible if r['pixel_sha256'] not in conflicts]
    picked=select_diverse(eligible,args.count+args.validation,args.seed)
    # Different original-crop representatives already enforced, then fixed generic validation split.
    picked.sort(key=lambda r:hashlib.sha256(f'{args.seed+1}|{r["image_path"]}'.encode()).digest())
    validation=picked[:args.validation];train=picked[args.validation:]
    target={split:[normalized(r,False) for r in read_rows(top/f'{split}.csv')] for split in ('train','validation','test')}
    catalogue=train+validation+target['train']+target['validation']
    for i,r in enumerate(catalogue,2_000_000):r['img_id']=str(i)
    for name,items in (('generic_train',train),('generic_validation',validation),('top20_train',target['train']),
                       ('top20_validation',target['validation']),('top20_test',target['test']),
                       ('finetune_mixed',target['train']+train),('encode',catalogue)):
        write_rows(out/f'{name}.csv',items)
    if rejected:write_rows(out/'rejected.csv',rejected)
    tables=out/'tables';tables.mkdir()
    mapping=json.loads((top/'tables/callig_remap.json').read_text());mapping['-1']=len(mapping)
    (tables/'callig_remap.json').write_text(json.dumps(mapping,indent=2))
    chars=sorted({r['character_id'] for r in catalogue+target['test']},key=int)
    (tables/'char_remap.json').write_text(json.dumps({v:i for i,v in enumerate(chars)},indent=2))
    for name in ('font_remap.json','font_table.npy','callig_table.npy'):
        (tables/name).write_bytes((top/'tables'/name).read_bytes())
    manifest=dict(status='complete',generic_train=len(train),generic_validation=len(validation),
        counts_by_script=dict(Counter(r['script_id'] for r in train)),eligible=len(eligible),reasons=dict(reasons),
        input_sha256=sha(args.pool),protected=[dict(path=str(p),sha256=sha(p)) for p in protected_files],
        std_root=str(std),std_provenance_sha256=sha(std/'provenance.json'),trust_source_labels=args.trust_source_labels,
        near_guard='global 63bit DCT <=8 bits + crop64 SSIM>=.90/MAE<=.08; conservative quarantine, NOT proof of all near duplicates',
        label_caveat='Metadata consistency/approved source is not automated glyph-label or stroke-completeness verification',
        work_holdout='Original-crop and image-near-duplicate exclusion; not artwork-level holdout without reliable source IDs',
        files={p.name:sha(p) for p in out.glob('*.csv') if p.name!='rejected.csv'})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('GENERIC_DATA_READY',len(train),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pool','top20','std-root','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--protected',nargs='+',required=True,help='All legacy evaluation/holdout CSVs')
    p.add_argument('--root',default='.');p.add_argument('--count',type=int,default=100000)
    p.add_argument('--validation',type=int,default=2000);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=90);p.add_argument('--trust-source-labels',action='store_true')
    args=p.parse_args()
    if args.count<=0 or args.validation<=0:p.error('Positive dataset sizes required')
    prepare(args)


if __name__=='__main__':main()
