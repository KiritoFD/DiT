"""Seal actual assets and completed preflights; never fabricate readiness."""
import argparse
import json
from pathlib import Path
from ..runtime.common import sha,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--runner-proof',required=True)
    p.add_argument('--large-proof',required=True);p.add_argument('--accumulation-proofs',nargs='+',required=True)
    p.add_argument('--joint-runner-proof',default='')
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());paths=set()
    runner=json.loads(Path(a.runner_proof).read_text())
    if not runner.get('resume_passed'):raise ValueError('Runner resume preflight missing')
    large=json.loads(Path(a.large_proof).read_text())
    if large.get('arm')!='S2-large' or large.get('capacity_only') is not True or large['gradient_checkpointing']:
        raise ValueError('Invalid S2-large proof')
    paths.update([a.runner_proof,a.large_proof,*a.accumulation_proofs])
    proofs=[json.loads(Path(p).read_text()) for p in a.accumulation_proofs]
    if not all(v.get('passed') for v in proofs):raise ValueError('Accumulation preflight failed')
    if not any(v['batch']==512 for v in proofs) or not any(v['precision'].startswith('FP32') for v in proofs):
        raise ValueError('Need both B512 and strict FP32 accumulation proof')
    for job in plan['stages']['1']:
        spec=json.loads(Path(job['spec']).read_text());cfg=spec['config']
        expected=plan['contract']['p2_sha256' if spec['kind']=='joint' else 'p1_sha256']
        if sha(spec['source'])!=expected:raise ValueError('Source checkpoint changed')
        paths.add(spec['source']);paths.add(job['spec'])
        if sha(cfg['data_csv'])!=plan['contract']['train_csv_sha256']:raise ValueError('Training CSV changed')
        for key in ('data_csv','eval_csv','std_shape_cache','char_remap_json','callig_remap_json','font_remap_json'):
            paths.add(cfg[key])
        for key in ('latent_shards_dir','std_soft_root','vae_path'):
            found=[p for p in Path(cfg[key]).rglob('*') if p.is_file()]
            if not found:raise FileNotFoundError(cfg[key])
            paths.update(map(str,found))
        if spec['kind']=='joint':
            if not a.joint_runner_proof:raise ValueError('Need real J save/resume proof before long training')
            proof=json.loads(Path(a.joint_runner_proof).read_text())
            if not proof.get('resume_passed') or not proof.get('paired_decoder_verified'):
                raise ValueError('J resume and matching decoder export not verified')
            paths.add(a.joint_runner_proof)
            cal=spec['joint']['calibration'];data=json.loads(Path(cal).read_text())
            if data['source_checkpoint_sha256']!=expected or not data['paired_gt_checked']:
                raise ValueError('Real paired J calibration missing')
            paths.add(cal)
            import csv
            paths.update(r['image_path'] for r in csv.DictReader(open(cfg['data_csv'],encoding='utf-8-sig')))
    # Capture the actual dirty source snapshot, not just a potentially stale remote Git revision.
    paths.update(str(p) for p in Path('src').rglob('*.py'))
    paths.update(str(p) for p in Path('tools/style_matrix').rglob('*.py'))
    target=Path(plan['ready'])
    if target.exists():raise FileExistsError('Ready record already exists')
    save(target,dict(contract=plan['contract'],files=[dict(path=p,sha256=sha(p)) for p in sorted(paths)],
        no_training_started=True))


if __name__=='__main__':main()
