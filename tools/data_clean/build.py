"""Build non-overwriting candidate CSVs from a completed, frozen image audit."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .common import key, read_json, rows, sha, write_csv, write_json
from .scan import ALIASES


def conflicts(records):
    by_group = defaultdict(list)
    for r in records:by_group[r['duplicate_group']].extend(r['metadata'])
    bad = set()
    for group,metadata in by_group.items():
        identities = {(m['calligrapher'],m['script'],ALIASES.get(m['character'],m['character'])) for m in metadata}
        if len(identities) > 1:bad.add(group)
    return bad


def coverage(items):
    return dict(n=len(items),characters=len({r['character'] for r in items}),
        slots=dict(sorted(Counter(r['calligrapher']+'_'+r['script'] for r in items).items())),
        sources=dict(sorted(Counter(r.get('source','') for r in items).items())))


def run(args):
    scan = Path(args.scan)
    protocol = read_json(scan/'manifest.json')
    if protocol.get('status') != 'complete' or protocol.get('schema') != 'calligraphy-clean-scan-v1':
        raise ValueError('Require a complete compatible scan')
    if sha(scan/'images.json') != protocol['images_sha256']:raise ValueError('Audit records changed')
    for role,info in protocol['inputs'].items():
        if sha(scan/'inputs'/f'{role}.csv') != info['sha256']:raise ValueError('Input snapshot changed')
    records = read_json(scan/'images.json')
    index = {r['image_path']:r for r in records}
    decisions = read_json(args.decisions) if args.decisions else {}
    if set(decisions)-set(index):raise ValueError('Decision references an image absent from this scan')
    for path,decision in decisions.items():
        if decision['action'] not in ('keep','exclude'):raise ValueError('Unknown decision action')
        if decision['file_sha256'] != index[path].get('file_sha256'):raise ValueError(f'Decision hash mismatch: {path}')
        if not decision.get('reason'):raise ValueError('A human decision needs a reason')
    # Confirmed exclusions propagate through duplicate groups, not only filenames.
    excluded_groups = {index[k]['duplicate_group'] for k,d in decisions.items() if d['action']=='exclude'}
    bad_labels = conflicts(records)
    reasons = {}
    for r in records:
        p = r['image_path'];d = decisions.get(p,{})
        reason = list(r['hard_issues'])
        if r['duplicate_group'] in excluded_groups:reason.append('confirmed_dirty_duplicate_group')
        if r['duplicate_group'] in bad_labels:reason.append('conflicting_writer_script_or_character_labels')
        if (r['flags'] and d.get('action') not in ('keep','exclude') and
                r['duplicate_group'] not in excluded_groups):reason.append('quality_pending_review')
        reasons[p] = reason
    train = rows(scan/'inputs/train.csv');evaluation = rows(scan/'inputs/eval.csv')
    old_top5 = rows(scan/'inputs/top5.csv')
    train_paths = {key(r['image_path']) for r in train}
    top5_paths = {key(r['image_path']) for r in old_top5}
    if not top5_paths <= train_paths:raise ValueError('Top5 must be a subset of base Top10 train CSV')
    # Reserve every original eval group, even when that eval image is quarantined.
    held_groups = {index[key(r['image_path'])]['split_group'] for r in evaluation}
    clean_train,clean_eval,ledger = [],[],[]
    used_train,used_eval = set(),set()
    for role,source,destination,used in [('eval',evaluation,clean_eval,used_eval),('train',train,clean_train,used_train)]:
        for row in source:
            p = key(row['image_path']);record = index[p];why = list(reasons[p])
            if role == 'train' and record['split_group'] in held_groups:why.append('original_eval_group_reserved')
            if not why and record['duplicate_group'] in used:why.append('duplicate_representative_retained_elsewhere')
            if not why:
                used.add(record['duplicate_group']);destination.append(row)
            ledger.append(dict(role=role,image_path=p,decision='quarantine' if why else 'keep',
                reasons=';'.join(why),duplicate_group=record['duplicate_group'],split_group=record['split_group'],
                flags=';'.join(record['flags']),manual_reason=decisions.get(p,{}).get('reason','')))
    for field in ('pixel_sha256','duplicate_group','split_group'):
        assert not ({index[key(r['image_path'])][field] for r in clean_train} &
                    {index[key(r['image_path'])][field] for r in clean_eval}),field
    assert not ({index[key(r['image_path'])]['split_group'] for r in clean_train} & held_groups)
    for row in clean_train+clean_eval:
        p = key(row['image_path'])
        if sha(Path(protocol['root'])/p) != index[p]['file_sha256']:
            raise ValueError(f'Original image changed since scan: {p}; rescan, never reuse stale decisions')
    top5_train = [r for r in clean_train if key(r['image_path']) in top5_paths]
    writers = {r['calligrapher_id'] for r in old_top5}
    top5_eval = [r for r in clean_eval if r['calligrapher_id'] in writers]
    out = Path(args.output);out.mkdir(parents=True,exist_ok=False)
    for name,items,fields in [
        ('train_top10_candidate.csv',clean_train,list(train[0])),
        ('eval_top10_candidate.csv',clean_eval,list(evaluation[0])),
        ('train_top5_candidate.csv',top5_train,list(train[0])),
        ('eval_top5_candidate.csv',top5_eval,list(evaluation[0]))]:
        write_csv(out/name,items,fields)
    write_csv(out/'quarantine.csv',[r for r in ledger if r['decision']!='keep'],list(ledger[0]))
    char_set = {r['character'] for r in clean_train}
    slot_char = {(r['calligrapher'],r['script'],r['character']) for r in clean_train}
    slot_family = {(r['calligrapher'],r['script'],ALIASES.get(r['character'],r['character'])) for r in clean_train}
    annotations = []
    for r in clean_eval:
        rec = index[key(r['image_path'])]
        annotations.append(dict(image_path=r['image_path'],character=r['character'],calligrapher=r['calligrapher'],script=r['script'],
            duplicate_group=rec['duplicate_group'],split_group=rec['split_group'],
            seen_character_global=r['character'] in char_set,
            seen_writer_script_character=(r['calligrapher'],r['script'],r['character']) in slot_char,
            seen_writer_script_family=(r['calligrapher'],r['script'],ALIASES.get(r['character'],r['character'])) in slot_family,
            exact_pixel_seen=False))
    write_csv(out/'eval_annotations.csv',annotations,list(annotations[0]) if annotations else ['image_path'])
    write_json(out/'decisions.json',decisions)
    before = dict(train_top10=coverage(train),eval_top10=coverage(evaluation),train_top5=coverage(old_top5))
    after = dict(train_top10=coverage(clean_train),eval_top10=coverage(clean_eval),train_top5=coverage(top5_train),eval_top5=coverage(top5_eval))
    report = dict(schema='calligraphy-clean-build-v1',status='candidate_not_training_approved',
        scan_manifest_sha256=sha(scan/'manifest.json'),scan_images_sha256=sha(scan/'images.json'),
        image_pixels_changed=False,ids_changed=False,original_eval_groups_reserved=len(held_groups),
        source_scope='same src_image_path only; NOT artwork/collection-level holdout',
        near_scope=protocol['near_scope'],unreviewed_policy='quarantine_not_delete',
        label_conflict_groups=len(bad_labels),confirmed_dirty_groups=len(excluded_groups),
        pending_quality_images=sum('quality_pending_review' in v for v in reasons.values()),
        reasons=dict(Counter(x for r in ledger for x in r['reasons'].split(';') if x)),
        before=before,after=after,files={p.name:sha(p) for p in sorted(out.glob('*.csv'))})
    pixels = Counter(r['pixel_sha256'] for r in records if 'pixel_sha256' in r)
    group_roles = defaultdict(set)
    for r in records:group_roles[r['split_group']].update(r['roles'])
    report['audit_counts'] = dict(exact_pixel_duplicate_groups=sum(n>1 for n in pixels.values()),
        exact_pixel_duplicate_excess=sum(n-1 for n in pixels.values()),
        original_train_eval_linked_groups=sum(v=={'train','eval'} for v in group_roles.values()))
    write_json(out/'report.json',report)
    print('BUILD',dict(before={k:v['n'] for k,v in before.items()},after={k:v['n'] for k,v in after.items()},
        reasons=report['reasons']),flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',required=True);p.add_argument('--output',required=True);p.add_argument('--decisions')
    run(p.parse_args(argv))


if __name__ == '__main__':main()
