"""One globally held-out test pool, nested 2/5/10/20-writer training cohorts."""
import argparse
from collections import Counter,defaultdict
import hashlib
import math
from pathlib import Path

import numpy as np

from tools.data_clean.common import key,rows,sha,read_json,write_json,write_csv
from tools.data_clean.build import conflicts,coverage
from tools.data_clean.scan import ALIASES
from .prepare import FIRST_TEN


def rank_key(row,seed):
    return hashlib.sha256((str(seed)+'|'+key(row['image_path'])).encode()).hexdigest()


def select_test(writer,candidates,legacy,per_writer,seed):
    """Sqrt-proportional script quotas; unique character families and split groups."""
    by_script = defaultdict(list)
    for r in candidates:by_script[r['script']].append(r)
    counts = {s:len(v) for s,v in by_script.items()}
    quotas = {s:min(2,len(v)) for s,v in by_script.items()}
    if sum(quotas.values()) > per_writer:quotas = {s:0 for s in by_script}
    while sum(quotas.values()) < per_writer:
        possible = [s for s in by_script if quotas[s] < counts[s]]
        if not possible:raise ValueError(f'Insufficient test candidates for {writer}')
        selected = max(possible,key=lambda s:(math.sqrt(counts[s])/(quotas[s]+1),s))
        quotas[selected] += 1
    chosen,chars,groups = [],set(),set()
    for s in sorted(by_script):
        if quotas[s] == 0:continue
        items = sorted(by_script[s],key=lambda r:(r['image_path'] not in legacy,rank_key(r,seed)))
        for r in items:
            char = ALIASES.get(r['character'],r['character'])
            if char in chars or r['split_group'] in groups:continue
            chosen.append(r);chars.add(char);groups.add(r['split_group'])
            if sum(x['script']==s for x in chosen) >= quotas[s]:break
    if len(chosen) < per_writer:
        for r in sorted(candidates,key=lambda r:(r['image_path'] not in legacy,rank_key(r,seed))):
            char = ALIASES.get(r['character'],r['character'])
            if char in chars or r['split_group'] in groups:continue
            chosen.append(r);chars.add(char);groups.add(r['split_group'])
            if len(chosen) == per_writer:break
    if len(chosen) != per_writer:raise ValueError(f'Not enough independent character/groups: {writer}')
    return chosen


def normalized_row(row):
    r = {k:v for k,v in row.items() if k not in ('metadata','roles','flags')}
    r['original_character_id'] = row.get('character_id','')
    r['original_glyph_id'] = row.get('glyph_id','')
    r['character_id'] = str(ord(row['character']))
    r['glyph_id'] = str(int(row['script_id'])*0x110000+ord(row['character']))
    return r


def run(args):
    scan = Path(args.scan);m = read_json(scan/'manifest.json')
    if m['status'] != 'complete' or sha(scan/'images.json') != m['images_sha256']:raise ValueError('Invalid scan')
    for role,info in m['inputs'].items():
        if sha(scan/'inputs'/f'{role}.csv') != info['sha256']:raise ValueError('Input snapshot changed')
    records = read_json(scan/'images.json');index = {r['image_path']:r for r in records}
    decisions = read_json(args.decisions)
    if set(decisions)-set(index):raise ValueError('Decision outside pool')
    for p,d in decisions.items():
        if d['file_sha256'] != index[p].get('file_sha256') or d['action'] not in ('keep','exclude') or not d.get('reason'):
            raise ValueError(f'Invalid decision: {p}')
    bad = conflicts(records)
    dirty = {index[p]['duplicate_group'] for p,d in decisions.items() if d['action']=='exclude'}
    eligibility,quarantine = {},[]
    for r in records:
        p = r['image_path'];d = decisions.get(p,{})
        reason = list(r['hard_issues'])
        if r['duplicate_group'] in bad:reason.append('label_conflict')
        if r['duplicate_group'] in dirty:reason.append('confirmed_dirty')
        if r['flags'] and d.get('action') not in ('keep','exclude') and r['duplicate_group'] not in dirty:reason.append('quality_pending_review')
        eligibility[p] = not reason
        if reason:quarantine.append(dict(image_path=p,reasons=';'.join(reason),flags=';'.join(r['flags']),duplicate_group=r['duplicate_group']))
    train_input = rows(scan/'inputs/train.csv');eval_input = rows(scan/'inputs/eval.csv')
    metadata = {key(r['image_path']):r for r in train_input+eval_input}
    candidates = []
    used = set()
    # Keep a single original record, preferring legacy evaluation representatives.
    for p in sorted(metadata,key=lambda p:(not ('eval' in index[p]['roles']),p)):
        audit = index[p]
        if not eligibility[p] or audit['duplicate_group'] in used:continue
        used.add(audit['duplicate_group'])
        candidates.append(dict(**metadata[p],split_group=audit['split_group'],duplicate_group=audit['duplicate_group'],pixel_sha256=audit['pixel_sha256']))
    counts = Counter(r['calligrapher'] for r in candidates)
    extras = sorted((w for w,n in counts.items() if w not in FIRST_TEN and n>=args.min_writer),key=lambda w:(-counts[w],w))
    if len(extras) < 10:raise ValueError('Fewer than ten usable additional writers')
    writers = FIRST_TEN+extras[:10]
    if any(counts[w] <= args.per_writer+20 for w in writers):raise ValueError('Writer too small for stable training/test')
    legacy_paths = {key(r['image_path']) for r in eval_input}
    reserved = {index[p]['split_group'] for p in legacy_paths}
    test = []
    for writer in writers:
        test.extend(select_test(writer,[r for r in candidates if r['calligrapher']==writer],legacy_paths,args.per_writer,args.seed))
    reserved |= {r['split_group'] for r in test}
    test_groups = {r['split_group'] for r in test}
    validation = []
    for writer in writers:
        validation.extend(select_test(writer,[r for r in candidates if r['calligrapher']==writer and
            r['split_group'] not in test_groups],legacy_paths,args.per_writer_val,args.seed+1))
    val_groups = {r['split_group'] for r in validation}
    reserved |= val_groups
    training = [r for r in candidates if 'train' in index[key(r['image_path'])]['roles'] and
        r['calligrapher'] in writers and r['split_group'] not in reserved]
    for field in ('pixel_sha256','duplicate_group','split_group'):
        assert not ({r[field] for r in training} & {r[field] for r in test}),field
        assert not ({r[field] for r in training} & {r[field] for r in validation}),field
        assert not ({r[field] for r in test} & {r[field] for r in validation}),field
    assert len({r['split_group'] for r in test}) == len(test)
    assert len(val_groups) == len(validation)
    assert not ({r['split_group'] for r in training} & reserved)
    if len({int(r['img_id']) for r in training+test+validation}) != len(training+test+validation):raise ValueError('Target img_id collision')
    for r in training+test+validation:
        if sha(Path(m['root'])/r['image_path']) != index[r['image_path']]['file_sha256']:raise ValueError('Image changed since audit')
    training.sort(key=lambda r:(writers.index(r['calligrapher']),rank_key(r,args.seed)))
    test.sort(key=lambda r:(writers.index(r['calligrapher']),rank_key(r,args.seed)))
    validation.sort(key=lambda r:(writers.index(r['calligrapher']),rank_key(r,args.seed)))
    for r in test+validation:r['eval_noise_seed'] = str(int(hashlib.sha256((str(args.seed)+'|'+r['pixel_sha256']).encode()).hexdigest()[:15],16))
    out = Path(args.output);out.mkdir(parents=True,exist_ok=False)
    writer_ids = {w:{int(r['calligrapher_id']) for r in training+test if r['calligrapher']==w} for w in writers}
    if any(len(v)!=1 for v in writer_ids.values()):raise ValueError('Writer ID ambiguity')
    raw_ids = [next(iter(writer_ids[w])) for w in writers]
    if len(set(raw_ids)) != 20:raise ValueError('Writer raw IDs collide')
    # One shared initialization bank. Existing Top10 rows can retain old table values.
    table = np.random.default_rng(args.seed).normal(0,.02,(20,32)).astype('float32')
    script_table = np.random.default_rng(args.seed+1).normal(0,.02,(3,16)).astype('float32')
    init = {}
    if args.init_tables:
        base = Path(args.init_tables);old_map = read_json(base/'callig_remap.json');old_table = np.load(base/'callig_table.npy')
        for i,raw in enumerate(raw_ids):
            if str(raw) in old_map:table[i] = old_table[old_map[str(raw)]]
        font_map = read_json(base/'font_remap.json');old_script = np.load(base/'font_table.npy')
        script_table = old_script[[font_map[str(i)] for i in (0,3,4)]]
        if script_table.shape != (3,16):raise ValueError('Script table shape mismatch')
        init = {p.name:sha(p) for p in [base/'callig_remap.json',base/'callig_table.npy',base/'font_table.npy',base/'font_remap.json']}
    levels = {}
    all_annotations = []
    previous_train,previous_test = set(),set()
    for size in (2,5,10,20):
        folder = out/f'top{size}';folder.mkdir();tables = folder/'tables';tables.mkdir()
        known = set(writers[:size]);tr = [r for r in training if r['calligrapher'] in known];ev = [r for r in test if r['calligrapher'] in known]
        val = [r for r in validation if r['calligrapher'] in known]
        tr_paths,ev_paths = {r['image_path'] for r in tr},{r['image_path'] for r in ev}
        assert previous_train <= tr_paths and previous_test <= ev_paths
        previous_train,previous_test = tr_paths,ev_paths
        write_csv(folder/'train.csv',[normalized_row(r) for r in tr])
        write_csv(folder/'test.csv',[normalized_row(r) for r in ev])
        write_csv(folder/'validation.csv',[normalized_row(r) for r in val])
        char_ids = sorted({ord(r['character']) for r in tr+ev+val})
        write_json(tables/'callig_remap.json',{str(raw):i for i,raw in enumerate(raw_ids[:size])})
        write_json(tables/'font_remap.json',{'0':0,'3':1,'4':2})
        write_json(tables/'char_remap.json',{str(raw):i for i,raw in enumerate(char_ids)})
        np.save(tables/'callig_table.npy',table[:size]);np.save(tables/'font_table.npy',script_table)
        chars = {r['character'] for r in tr};slots = {(r['calligrapher'],r['script'],r['character']) for r in tr}
        annotations = []
        for r in ev:
            annotations.append(dict(image_path=r['image_path'],character=r['character'],calligrapher=r['calligrapher'],script=r['script'],
                eval_noise_seed=r['eval_noise_seed'],seen_character_global=r['character'] in chars,
                seen_writer_script_character=(r['calligrapher'],r['script'],r['character']) in slots,
                exact_pixel_seen=False,source_file_seen=False))
        write_csv(folder/'test_annotations.csv',annotations)
        levels[str(size)] = dict(writers=writers[:size],train=coverage(tr),test=coverage(ev),
            validation=coverage(val),
            new_test_from_former_training=sum(r['image_path'] not in legacy_paths for r in ev),
            all_test_groups_reserved_from_all_levels=True,
            files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()})
        all_annotations.extend(dict(cohort=size,**r) for r in annotations)
    write_csv(out/'quarantine.csv',quarantine)
    write_csv(out/'test_protocol.csv',all_annotations)
    write_csv(out/'all_train.csv',[normalized_row(r) for r in training])
    write_csv(out/'all_test.csv',[normalized_row(r) for r in test])
    write_csv(out/'all_validation.csv',[normalized_row(r) for r in validation])
    write_json(out/'decisions.json',decisions)
    report = dict(schema='nested-writer-cohorts-v1',status='screened_candidate_not_expert_certified',seed=args.seed,
        scan_manifest_sha256=sha(scan/'manifest.json'),scan_images_sha256=sha(scan/'images.json'),
        anchored_writers=FIRST_TEN,additional_rank=extras,eligible_counts=dict(counts),
        per_writer_test=args.per_writer,per_writer_validation=args.per_writer_val,levels=levels,legacy_reserved_groups=len({index[p]['split_group'] for p in legacy_paths}),
        total_reserved_groups=len(reserved),near_scope=m['near_scope'],
        label_conflict_groups=len(bad),n_quarantined=len(quarantine),
        source_holdout='same original crop only, NOT artwork-level',noise='per-image SHA256-derived eval_noise_seed',
        init_tables=init,character_id='Unicode scalar; original_character_id retained; no character label normalization',
        dataset_root=m['root'],cache_status='not_built_yet',std_status='coverage_not_checked_yet')
    write_json(out/'manifest.json',report)
    print('COHORTS', {n:dict(train=v['train']['n'],validation=v['validation']['n'],test=v['test']['n']) for n,v in levels.items()},flush=True)
    print('WRITERS',writers,flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scan','decisions','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--init-tables');p.add_argument('--seed',type=int,default=90)
    p.add_argument('--per-writer',type=int,default=20);p.add_argument('--min-writer',type=int,default=200)
    p.add_argument('--per-writer-val',type=int,default=10)
    a = p.parse_args(argv)
    if min(a.per_writer,a.per_writer_val) < 1:p.error('per-writer and per-writer-val must be positive')
    run(a)


if __name__ == '__main__':main()
