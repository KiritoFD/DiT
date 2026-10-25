"""Validation-only prompt selection followed by a frozen, disjoint paired test."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

VARIANTS = ('original', 'fill', 'collage', 'named')


def read(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt(char, writer, variant):
    source = '参考图是一张包含六个不同真迹汉字的风格样本板。' if variant == 'collage' else '六张参考图是同一位书家的不同真迹汉字。'
    identity = f'书家是{writer}。' if variant == 'named' else ''
    return (source + identity + f'请沿用参考真迹的笔法，另写一个楷书汉字「{char}」。'
            '内容只由文字指定，不要写参考图中的字，不要生成字帖或拼图。'
            '保留参考的笔画粗细对比、起收笔、提按、转折、撇捺及结体特点，'
            '不要使用通用电脑楷体。输出只有一个完整单字。'
            '黑色墨迹、纯白背景、无纸纹、无框线、无印章、无落款。'
            '重要：单字放大至几乎铺满方形画布，最长方向占画幅95%至98%，居中，'
            '只留极窄白边，不裁切任何笔画。')


def prepare(source, prior, root, baseline):
    from PIL import Image
    original = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    old = json.loads((prior / 'manifest.json').read_text(encoding='utf-8'))
    prior_keys = {(j['writer'], j['character']) for j in old['jobs']}
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / 'inputs', root / 'inputs')
    (root / 'gt').mkdir()
    cohort = Path('assets/data_cohorts/top20_20261021_v1/datasets_final/top20')
    pools = {split: read(cohort / (split + '.csv')) for split in ('validation', 'test')}
    samples = {}
    for split, count in (('validation', 2), ('test', 10)):
        selected = []
        for wi, writer in enumerate(original['writers']):
            refs = original['reference_records'][writer]
            excluded = {r['character'] for r in refs}
            seen = set()
            for row in sorted(pools[split], key=lambda r: sha_text(r['image_path'])):
                char = row['character']
                if row['calligrapher'] != writer or row['script'] != '楷':
                    continue
                if char in excluded or char in seen or (writer, char) in prior_keys:
                    continue
                seen.add(char)
                name = f'U{ord(char):04X}_w{wi}'
                gt = f'gt/{name}.png'
                im = Image.open(row['image_path']).convert('RGB')
                if im.size != (256, 256):
                    raise ValueError(f'Unexpected GT dimensions: {im.size}')
                im.save(root / gt)
                selected.append(dict(id=name, character=char, writer=writer,
                                     gt_file=gt, gt_sha256=sha(root / gt), gt_record=row))
                if len(seen) == count:
                    break
            if len(seen) != count:
                raise ValueError(f'Insufficient {split} characters for {writer}: {len(seen)}')
        samples[split] = selected
    if {r['character'] for r in samples['validation']} & {r['character'] for r in samples['test']}:
        raise ValueError('Validation/test character overlap')
    for wi, writer in enumerate(original['writers']):
        sheet = Image.new('RGB', (768, 512), 'white')
        for i, record in enumerate(original['reference_records'][writer]):
            sheet.paste(Image.open(root / record['file']).convert('RGB'), ((i % 3)*256, (i // 3)*256))
        sheet.save(root / 'inputs' / f'w{wi}_collage.png')
    original.update(samples=samples, input_sha256={str(p.relative_to(root)): sha(p)
                    for p in (root / 'inputs').glob('*.png')},
                    split_hashes={s: sha(cohort / (s+'.csv')) for s in pools},
                    design='24 validation generations (6 pairs x 4 variants); select minimum mean LPIPS; 60 test generations (30 pairs x 2 seeds). Prior pilot pairs excluded.')
    save(root / 'study.json', original)
    shutil.copy2(baseline, root / 'our_top20_30k.csv')
    save(root / 'baseline_identity.json', dict(source=str(baseline), sha256=sha(baseline),
                                              caveat='Historical EMA seed90 vs Qwen seeds20261022/23; same GT, not same generative noise.'))
    with tarfile.open(root / 'study.tar.gz', 'w:gz') as tar:
        for name in ('study.json', 'inputs', 'gt', 'our_top20_30k.csv', 'baseline_identity.json'):
            tar.add(root / name, arcname=name)
    print('PREPARED', {s: len(v) for s, v in samples.items()}, flush=True)


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def run(root, infer, scorer):
    study = json.loads((root / 'study.json').read_text(encoding='utf-8'))
    status = dict(pid=os.getpid(), phase='validation', completed_variants=[])
    def state():
        save(root / 'status.json', status)
    if (root / 'status.json').exists():
        raise FileExistsError('Refusing to mix study runs')
    state()

    def execute(split, variant):
        out = root / f'{split}_{variant}'
        out.mkdir()
        for name in ('inputs', 'gt', 'model'):
            (out / name).symlink_to(root / name, target_is_directory=True)
        jobs = []
        seeds = [20261022] if split == 'validation' else [20261022, 20261023]
        for item in study['samples'][split]:
            wi = study['writers'].index(item['writer'])
            for seed in seeds:
                images = [r['file'] for r in study['reference_records'][item['writer']]]
                if variant == 'collage':
                    images = [f'inputs/w{wi}_collage.png']
                job = dict(item, id=item['id']+f'_s{seed}', seed=seed,
                           mode='style_only', images=images, variant=variant)
                assert item['character'] not in {r['character'] for r in study['reference_records'][item['writer']]}
                if variant != 'original':
                    job['prompt'] = prompt(item['character'], item['writer'], variant)
                jobs.append(job)
        save(out / 'manifest.json', dict(study, jobs=jobs, n=len(jobs)))
        status.update(phase=split, variant=variant, current=str(out)); state()
        with (out / 'run.log').open('x') as log:
            subprocess.run([sys.executable, '-u', str(infer), '--root', str(out), '--skip-poster'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, '-u', str(scorer), 'score', '--root', str(out)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, cwd=root)
        return json.loads((out / 'metrics.json').read_text(encoding='utf-8'))

    try:
        validation = {}
        for variant in VARIANTS:
            validation[variant] = execute('validation', variant)
            status['completed_variants'].append(variant); state()
        winner = min(VARIANTS, key=lambda v: (validation[v]['means']['lpips'], -validation[v]['means']['ink_iou']))
        save(root / 'selection.json', dict(winner=winner, criterion='Lowest validation mean Alex LPIPS; Ink-IoU tie break; test never used', results=validation))
        test = execute('test', winner)
        historical = read(root / 'our_top20_30k.csv')
        keys = {(r['writer'], r['gt_record']['image_path']) for r in study['samples']['test']}
        ours = [r for r in historical if (r['calligrapher'], r['image_path']) in keys]
        if len(ours) != len(keys):
            raise ValueError(f'Baseline target mismatch: {len(ours)} vs {len(keys)}')
        metrics = ('ssim', 'lpips', 'ink_iou', 'frag_ratio', 'soft_ink_ratio')
        comparison = dict(qwen=test, ours_same_gt_n=len(ours),
                          ours={m: sum(float(r[m]) for r in ours)/len(ours) for m in metrics},
                          caveat='Qwen 2 seeds per GT; historical model 1 seed per GT; no matched noise claim')
        save(root / 'comparison.json', comparison)
        status.update(phase='completed', winner=winner); state()
    except Exception as error:
        status.update(phase='failed', error=repr(error)); state()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['prepare', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path); p.add_argument('--prior', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--infer', type=Path); p.add_argument('--scorer', type=Path)
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.source, a.prior, a.root, a.baseline)
    else:
        run(a.root.resolve(), a.infer.resolve(), a.scorer.resolve())
