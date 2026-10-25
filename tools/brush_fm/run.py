"""P0 -> short P1 -> stop for review; P2 only with an explicit acceptance marker."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from src.unet.io import Config
from src.unet.brush.attributes import normalize
from tools.generic_pretrain.contracts import read_rows,write_rows,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='assets/generic48_kxl_top10_20261022')
    p.add_argument('--out',default='assets/brush114_top10_20261023');p.add_argument('--accept-p2',action='store_true')
    a=p.parse_args();source=Path(a.source);out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    status=dict(phase='waiting_p0',started=time.strftime('%F %T'))
    def save():(out/'pipeline_status.json').write_text(json.dumps(status,indent=2))
    def run(command,name):
        status['phase']=name;save()
        with (out/(name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    def checkpoint(folder,step):
        files=list(Path(folder).glob(f'*/checkpoints/{step:07d}.pt'))
        if len(files)!=1:raise ValueError('Missing/ambiguous checkpoint '+str(files))
        return files[0]
    save()
    try:
        while not (out/'attributes/audit.json').exists():
            initial=json.loads((out/'status.json').read_text())
            if not (Path('/proc')/str(initial['pid'])/'cmdline').exists():raise RuntimeError('P0 exited before writing audit; inspect p0.log')
            time.sleep(30)
        audit=json.loads((out/'attributes/audit.json').read_text())
        if not audit['p0_pass']:
            status.update(phase='p0_failed_review_required',audit=audit);save();return
        mix=read_rows(source/'top10_train.csv');top=[r for r in mix if r['generic']=='0']
        known={r['character'] for r in top};test=read_rows(source/'top10_test.csv')
        held={r['character'] for r in test}-known
        if len(held)<8:raise ValueError('Not enough strict heldout characters')
        top_pixels={r['pixel_sha256'] for r in top if r.get('pixel_sha256')};seen_pixels=set();train=[]
        for row in mix:
            if row['character'] in held:continue
            pixel=row.get('pixel_sha256')
            if row['generic']=='1' and pixel:
                if pixel in top_pixels or pixel in seen_pixels:continue
                seen_pixels.add(pixel)
            train.append(row)
        write_rows(out/'attributes/pretrain.csv',train)
        write_rows(out/'attributes/finetune.csv',train)
        controls=[];seen=set()
        for row in test:
            if row['character'] in held and row['character'] not in seen:controls.append(row);seen.add(row['character'])
        write_rows(out/'attributes/control.csv',controls)
        original=np.load(out/'attributes/attributes.npz')
        data={k:original[k] for k in original.files};ids={int(r['img_id']) for r in train}
        data['train']=np.array([int(i) in ids for i in data['ids']])
        data['normalized'],data['center'],data['scale']=normalize(data['values'],data['scripts'],data['train'],data['valid'])
        cache=out/'attributes/training_attributes.npz';np.savez_compressed(cache,**data)
        base=json.loads((source/'pretrain_50k_config.json').read_text())
        base.update(channels=[192,384,576],num_res_blocks=4,brush_dim=5,brush_cache=str(cache),lr=6e-5,
            max_steps=60000,stop_at_step=5000,lr_decay_steps=60000,data_csv=str(out/'attributes/pretrain.csv'),
            output=str(out/'pretrain_runs'),initialize_from='')
        Config(**base).validate();config=out/'pretrain_config.json';config.write_text(json.dumps(base,indent=2))
        fine={**base,'writer_mode':'film','writer_film_rank':16,'writer_film_layers':['middle','decoder.'],
            'adapter_only_steps':0,'finetune_backbone_patterns':['*'],'lr':8e-5,'backbone_lr_scale':.125,
            'max_steps':30000,'stop_at_step':30000,'lr_decay_steps':30000,'warmup':200,
            'generic_replay_fraction':.1,'data_csv':str(out/'attributes/finetune.csv'),
            'eval_csv':str(source/'top10_validation.csv'),'output':str(out/'top10_runs')}
        Config(**fine).validate();(out/'top10_config.json').write_text(json.dumps(fine,indent=2))
        (out/'manifest.json').write_text(json.dumps(dict(source_manifest_sha256=sha(source/'manifest.json'),
            cache_sha256=sha(cache),pretrain_n=len(train),strict_heldout_chars=sorted(held),
            control_protocol='16 strict heldout characters x3seeds x3attributes x3levels; no target GT attributes',
            pretrain_mixture='75% generic /25% Top10, Writer completely bypassed',
            p2_gate='automatic statistics AND manual content review marker required'),indent=2))
        command=[sys.executable,'-u','-m','src.unet.train']
        if not a.accept_p2:
            run(command+['--config',str(config),'--benchmark','3'],'capacity')
            run(command+['--config',str(config)],'p1_train')
            ck=checkpoint(base['output'],5000)
            run([sys.executable,'-u','-m','tools.brush_fm.control','--checkpoint',str(ck),'--out',str(out/'control5k')],'p1_control')
            status.update(phase='p1_review_required',control_summary=json.loads((out/'control5k/summary.json').read_text()));save();return
        result=json.loads((out/'control5k/summary.json').read_text())
        if not result['p1_pass'] or not (out/'P1_CONTENT_REVIEW_ACCEPTED').exists():
            raise ValueError('P1 controls must pass and content review must be accepted before60k')
        base['stop_at_step']=60000;config.write_text(json.dumps(base,indent=2))
        run(command+['--config',str(config),'--resume',str(checkpoint(base['output'],5000))],'p2_pretrain')
        fine['initialize_from']=str(checkpoint(base['output'],60000));(out/'top10_config.json').write_text(json.dumps(fine,indent=2))
        run(command+['--config',str(out/'top10_config.json')],'p2_top10')
        status['phase']='completed';save()
    except BaseException as error:
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
