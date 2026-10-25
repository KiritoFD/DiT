"""Extend modern Top10 with unambiguous real-writer rows; preserve original pixels."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import shutil

from tools.data_clean.common import key,rows,sha,write_csv,write_json


SCRIPTS = {'楷':0,'行':3,'隶':4}
FIRST_TEN = ['王羲之','苏轼','欧阳询','颜真卿','赵孟頫','褚遂良','米芾','文徵明','柳公权','何绍基']


def prepare(args):
    root = Path(args.root).resolve();out = Path(args.output)
    image_dir = Path(args.image_dir)
    if out.exists() or image_dir.exists():raise FileExistsError('Use new output and image directories')
    old = rows(args.train_csv)
    if {r['calligrapher'] for r in old} != set(FIRST_TEN):raise ValueError('Expected current anchored Top10')
    protected = {}
    for path in args.legacy_eval:
        for r in rows(path):
            p = key(r['image_path'])
            if p in protected:
                for name in ('calligrapher','script','character'):
                    if protected[p][name] != r[name]:raise ValueError(f'Legacy eval metadata disagreement: {p}')
            else:protected[p] = r
    original = rows(args.master_csv)
    candidates,rejected = [],[]
    for row in original:
        reason = ''
        if row['calligrapher'] in FIRST_TEN:reason = 'anchored_writer_uses_modern_top10_only'
        elif row['script'] not in SCRIPTS:reason = 'unsupported_or_ambiguous_script_label'
        elif int(row['script_id']) != SCRIPTS[row['script']]:reason = 'script_name_id_mismatch'
        elif len(row['character']) != 1:reason = 'not_one_unicode_character'
        elif row['calligrapher'] in ('others','墨迹','唐','古文四声韵','书谱','碑刻','篆刻'):reason = 'not_an_individual_writer'
        if reason:rejected.append(dict(image_path=row['image_path'],calligrapher=row['calligrapher'],script=row['script'],reason=reason))
        else:candidates.append(row)
    counts = Counter(r['calligrapher'] for r in candidates)
    candidates = [r for r in candidates if counts[r['calligrapher']] >= args.min_writer]
    writers = defaultdict(set)
    for r in old+list(protected.values())+candidates:writers[r['calligrapher']].add(r['calligrapher_id'])
    if any(len(v)!=1 for v in writers.values()):raise ValueError('Multiple raw IDs for one writer; do not guess')
    ids = [next(iter(v)) for v in writers.values()]
    if len(set(ids)) != len(ids):raise ValueError('Raw writer ID collision')
    for r in old+list(protected.values()):
        if r['script'] not in SCRIPTS or int(r['script_id']) != SCRIPTS[r['script']]:raise ValueError('Modern script metadata mismatch')
        if not (root/r['image_path']).is_file():raise FileNotFoundError(r['image_path'])
    old_ids = {int(r['img_id']) for r in old+list(protected.values())}
    if old_ids and max(old_ids) >= args.first_id:raise ValueError('Expansion ID namespace overlaps old IDs')
    out.mkdir(parents=True);image_dir.mkdir(parents=True)
    write_json(out/'manifest.json',dict(status='incomplete'))
    bindings,expanded = [],[]
    for idx,row in enumerate(sorted(candidates,key=lambda r:key(r['image_path'])),args.first_id):
        src = root/row['image_path'];dest = image_dir/f'{idx}.png'
        shutil.copy2(src,dest)
        if sha(src) != sha(dest):raise ValueError('Pixel source copy failed')
        r = row.copy()
        r.update(original_image_path=row['image_path'],image_path=dest.resolve().relative_to(root).as_posix(),
            img_id=str(idx),aug='',std_path='',source='final_imgs_fame_v8_expansion',
            src_image_path=key(row['image_path']),slot_name=row['calligrapher']+'_'+row['script'])
        expanded.append(r)
        bindings.append(dict(image_path=r['image_path'],original_image_path=row['image_path'],img_id=idx,sha256=sha(dest)))
    write_csv(out/'train_pool.csv',old+expanded)
    write_csv(out/'legacy_eval.csv',protected.values())
    shutil.copy2(args.top5_csv,out/'top5_selector.csv')
    write_csv(out/'image_bindings.csv',bindings)
    write_csv(out/'rejected_metadata.csv',rejected)
    manifest = dict(schema='writer-cohort-prepare-v1',status='complete',root=str(root),
        anchored_writers=FIRST_TEN,additional_candidates=dict(sorted(Counter(r['calligrapher'] for r in expanded).items())),
        n_anchored_train=len(old),n_expansion=len(expanded),n_legacy_eval=len(protected),
        pixels_changed=False,new_img_id_start=args.first_id,writer_and_script_ids_changed=False,
        inputs={str(p):sha(p) for p in [args.master_csv,args.train_csv,args.top5_csv]+args.legacy_eval},
        files={p.name:sha(p) for p in sorted(out.glob('*.csv'))})
    write_json(out/'manifest.json',manifest)
    print('PREPARE',dict(base=len(old),expansion=len(expanded),legacy_eval=len(protected),candidates=manifest['additional_candidates']),flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('master-csv','train-csv','top5-csv','output','image-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--legacy-eval',nargs='+',required=True)
    p.add_argument('--root',default='.');p.add_argument('--min-writer',type=int,default=200)
    p.add_argument('--first-id',type=int,default=1000000)
    prepare(p.parse_args(argv))


if __name__ == '__main__':main()
