"""Minimal-gate full-pool 48M pretraining: protect evaluation, cache STD/VAE, train."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.fft import dctn

from src.unet.io import Config
from src.utils.std_soft_glyph import FontCoverage,FontFace,load_font_config,render_base_mask,sha256_file
from tools.generic_pretrain.contracts import read_rows,write_rows,sha,NearIndex,normalized


def fingerprint(path):
    with Image.open(path) as image:
        gray=np.asarray(image.convert('L'))
    if gray.shape!=(256,256):raise ValueError('Expected unified 256x256 image: '+str(path))
    pixel=hashlib.sha256(gray.tobytes()+str(gray.shape).encode()).hexdigest()
    y,x=np.where(gray<128)
    crop=gray[y.min():y.max()+1,x.min():x.max()+1] if len(x) else gray
    thumb=np.asarray(Image.fromarray(crop).resize((64,64),Image.Resampling.BILINEAR))
    small=np.asarray(Image.fromarray(thumb).resize((32,32),Image.Resampling.BILINEAR),dtype=float)
    coeff=dctn(small,type=2,norm='ortho')[:8,:8].ravel()[1:]
    phash=sum(int(v)<<i for i,v in enumerate(coeff>np.median(coeff)))
    return dict(pixel_sha256=pixel,phash=str(phash)),thumb


def prepare(args,out,status,save):
    rawroot=Path(args.pool_root);source=read_rows(rawroot/'meta/train_clean.csv')
    source=[r for r in source if r['script'] in ('楷','行','隶')]
    top=Path('assets/data_cohorts/top20_20261021_v1/datasets_final/top10')
    downstream={split:read_rows(top/f'{split}.csv') for split in ('train','validation','test')}
    globaltop=top.parent/'top20'
    protected_paths=[globaltop/'validation.csv',globaltop/'test.csv',
        Path('assets/data_cohorts/top20_20261021_v1/prepare/legacy_eval.csv')]
    guard=NearIndex();protected_hashes=set();protected_names=set()
    protected={r['image_path'] for p in protected_paths for r in read_rows(p)}
    for path in sorted(protected):
        rec,thumb=fingerprint(path);protected_hashes.add(rec['pixel_sha256']);guard.add(rec,thumb)
        protected_names.add(str(Path(path).resolve()))
    # Keep old Top20 raw script IDs; extend, do not reinterpret aliases used by the evaluator.
    script_ids={'楷':0,'行':3,'隶':4}
    names=['楷','行','隶']
    font_map={0:0,3:1,4:2}
    for raw in sorted(set(script_ids.values())-set(font_map)):font_map[raw]=len(font_map)
    _,specs,_=load_font_config('configs/std_soft_fonts.json')
    faces={sid:spec['faces'] for sid,spec in specs.items()}
    frozen={ (int(r['script_id']),r['character']):FontFace(r['font_path'],int(r['font_face']))
        for r in read_rows('assets/data_cohorts/top20_20261021_v1/std_bbox_v1/condition64/index.csv')}
    coverage=FontCoverage();assignments={};stdroot=out/'std_bbox64';stdroot.mkdir(exist_ok=True)
    index=[];unavailable=set();rendered={}
    pairs={(script_ids[r['script']],r['character']) for r in source+sum(downstream.values(),[]) if len(r['character'])==1}
    for i,(sid,char) in enumerate(sorted(pairs)):
        style=sid if sid in faces else 0
        face=frozen.get((sid,char)) or coverage.choose(faces[style],char)
        if face is None:unavailable.add((sid,char));continue
        token=(face.path,char)
        if token not in rendered:rendered[token]=render_base_mask(char,face,output_size=64)
        rel=Path(str(sid))/f'U+{ord(char):05X}.png';dest=stdroot/rel;dest.parent.mkdir(exist_ok=True)
        rendered[token].save(dest)
        assignments[sid,char]=face
        index.append(dict(script_id=sid,character=char,path=rel.as_posix(),font_path=face.path,font_face=face.face,
                          std_style_source='matched' if sid in faces else 'regular_content_prior'))
        if (i+1)%2000==0:status.update(std_done=i+1,std_total=len(pairs));save()
    write_rows(stdroot/'index.csv',index)
    (stdroot/'provenance.json').write_text(json.dumps(dict(geometry='true_ink_bbox_fit256_aspect_preserved -> BOX64',
        stored_blur_sigma=0,script_ids=script_ids,unknown_style_std='regular content prior, NOT style-matched font',
        fonts={f.path:sha256_file(f.path) for f in set(assignments.values())}),indent=2,ensure_ascii=False),encoding='utf-8')
    for split,rows in downstream.items():
        missing=[r for r in rows if (int(r['script_id']),r['character']) not in assignments]
        if missing:raise ValueError('STD missing for downstream '+split+': '+str(missing[:2]))
    status.update(phase='exclude_leakage',source_rows=len(source));save()
    kept=[];counts=Counter()
    def inspect(row):
        sid=script_ids[row['script']];char=row['character'];path=rawroot/row['image_path']
        if len(char)!=1:return None,'invalid_character'
        if (sid,char) not in assignments:return None,'std_missing'
        if str(path.resolve()) in protected_names:return None,'protected_path'
        try:record,thumb=fingerprint(path)
        except (OSError,ValueError):return None,'unreadable_or_wrong_shape'
        if record['pixel_sha256'] in protected_hashes:return None,'protected_exact'
        if guard.matches(record,thumb):return None,'protected_near'
        row=normalized(dict(row,image_path=str(path),script_id=str(sid)),True)
        row['pixel_sha256']=record['pixel_sha256']
        return row,None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0,len(source),512):
            for row,reason in executor.map(inspect,source[start:start+512]):
                if reason:counts[reason]+=1
                else:kept.append(row)
            status.update(screened=min(start+512,len(source)),accepted=len(kept),excluded=dict(counts));save()
    # Exact-pixel groups cannot straddle generic train/validation. No aesthetic quality filtering.
    hashes=sorted({r['pixel_sha256'] for r in kept},key=lambda h:hashlib.sha256(('90|'+h).encode()).digest())
    valhash=set(hashes[:1000]);validation=[r for r in kept if r['pixel_sha256'] in valhash]
    train=[r for r in kept if r['pixel_sha256'] not in valhash]
    if not train or len(hashes)<=1000:raise ValueError('Full pool unexpectedly empty/small')
    fine={split:[normalized(dict(r),False) for r in rows] for split,rows in downstream.items()}
    allrows=train+validation+fine['train']+fine['validation']
    for i,row in enumerate(allrows,2_000_000):row['img_id']=str(i)
    write_rows(out/'train.csv',train);write_rows(out/'validation.csv',validation);write_rows(out/'encode.csv',allrows)
    write_rows(out/'top10_train.csv',fine['train']+train)
    write_rows(out/'top10_validation.csv',fine['validation'])
    write_rows(out/'top10_test.csv',fine['test'])
    tables=out/'tables';tables.mkdir(exist_ok=True)
    callig=json.loads((top/'tables/callig_remap.json').read_text());realmap=dict(callig);callig['-1']=10
    (tables/'callig_remap.json').write_text(json.dumps(callig,indent=2))
    (tables/'font_remap.json').write_text(json.dumps(font_map,indent=2))
    chars=sorted({ord(r['character']) for r in allrows+sum(downstream.values(),[])})
    (tables/'char_remap.json').write_text(json.dumps({str(v):i for i,v in enumerate(chars)},indent=2))
    rng=np.random.default_rng(90);script_table=rng.normal(0,.02,(len(names),16)).astype(np.float32)
    script_table[:3]=np.load(top/'tables/font_table.npy');np.save(tables/'font_table.npy',script_table)
    sourcewriter=Path('assets/data_cohorts/top20_20261021_v1/writer_tables_supcon_v1')
    sourcemap=json.loads((sourcewriter/'callig_remap.json').read_text())
    sourcearray=np.load(sourcewriter/'callig_table.npy')
    writerroot=out/'writer_table';writerroot.mkdir(exist_ok=True);writerfile=writerroot/'callig_table.npy'
    writerarray=np.empty((10,32),dtype=np.float32)
    for raw,dense in realmap.items():writerarray[dense]=sourcearray[sourcemap[raw]]
    np.save(writerfile,writerarray)
    (writerroot/'callig_remap.json').write_text(json.dumps(realmap,indent=2))
    (writerroot/'provenance.json').write_text(json.dumps(dict(source=str(sourcewriter),
        source_sha256=sha(sourcewriter/'callig_table.npy'),selection=realmap),indent=2))
    (tables/'callig_table.npy').write_bytes(writerfile.read_bytes())
    cfg=Config(channels=(96,192,256),num_res_blocks=8,num_writers=10,num_scripts=len(names),writer_mode='none',
        decouple_writer_condition=True,writer_table_file=str(writerfile),freeze_writer_table=True,
        batch=512,lr=7e-5,warmup=1500,max_steps=50000,lr_decay_steps=50000,stop_at_step=50000,
        compile=True,std_shape_aug='off',std_soft_sigma_min=0,std_soft_sigma_max=0,std_soft_eval_sigma=0,
        data_csv=str(out/'train.csv'),eval_csv=str(out/'validation.csv'),latent_shards_dir=str(out/'latents'),
        img_root='',std_soft_root=str(stdroot),condition_tables_root=str(tables),
        callig_remap_json=str(tables/'callig_remap.json'),font_remap_json=str(tables/'font_remap.json'),
        char_remap_json=str(tables/'char_remap.json'),output=str(out/'runs'))
    cfg.validate()
    from dataclasses import asdict
    (out/'train_config.json').write_text(json.dumps(asdict(cfg),indent=2))
    target=asdict(cfg)
    target.update(writer_mode='film',writer_film_rank=16,writer_film_layers=['middle','decoder.'],
        adapter_only_steps=1000,backbone_lr_scale=.2,generic_replay_fraction=.1,
        lr=5e-5,warmup=200,max_steps=10000,lr_decay_steps=10000,stop_at_step=10000,
        ckpt_every=5000,eval_every=5000,data_csv=str(out/'top10_train.csv'),
        eval_csv=str(out/'top10_validation.csv'),output=str(out/'top10_runs'))
    (out/'top10_config.json').write_text(json.dumps(target,indent=2))
    manifest=dict(source=str(rawroot),source_rows=len(source),train=len(train),validation=len(validation),excluded=dict(counts),
        input_csv_sha256=sha(rawroot/'meta/train_clean.csv'),script_ids=script_ids,font_remap=font_map,
        quality='No new width/ink/fragment/background quality selection; use all unified readable supported glyphs',
        leakage='Exclude Top20 validation/test + legacy heldout exact pixels and conservative near duplicates; '
                '63bit DCT <=8 bits, cropped64 SSIM>=.90 and MAE<=.08. Not artwork-level proof.',
        generic_validation='Exact pixel groups disjoint; near duplicates within generic pool not newly purged',
        std_root=str(stdroot),writer_condition='none: no Writer lookup/projection in forward',
        files={p.name:sha(p) for p in (out/'train.csv',out/'validation.csv',out/'encode.csv',out/'train_config.json')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return cfg


def extend_running(out):
    """Replace only the orchestrator; preserve the running 30k trainer, then resume full state."""
    status=json.loads((out/'status.json').read_text());pid=status['train_pid']
    proc=Path('/proc')/str(pid)
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if status['phase']!='training' or 'src.unet.train' not in command or str(out/'train_config.json') not in command:
        raise ValueError('Takeover requires this experiment\'s active generic trainer')
    parent=int(next(line.split()[1] for line in (proc/'status').read_text().splitlines() if line.startswith('PPid:')))
    parent_command=(Path('/proc')/str(parent)/'cmdline').read_bytes()
    if b'tools.generic_pretrain.full' not in parent_command:raise ValueError('Unexpected old runner parent')
    original=json.loads((out/'train_config.json').read_text())
    updated={**original,'max_steps':50000,'stop_at_step':50000,'lr_decay_steps':50000,
             'ckpt_every':5000,'eval_every':5000}
    Config(**updated).validate()
    prepath=out/'pretrain_50k_config.json';prepath.write_text(json.dumps(updated,indent=2))
    fine=json.loads((out/'top10_config.json').read_text())
    fine.update(max_steps=10000,stop_at_step=10000,lr_decay_steps=10000,ckpt_every=5000,eval_every=5000)
    finepath=out/'top10_50k_config.json';finepath.write_text(json.dumps(fine,indent=2))
    os.kill(parent,signal.SIGTERM)  # Do not signal the trainer or its process group.
    status.update(runner_pid=os.getpid(),replaced_runner_pid=parent,phase='training_to_30k_then_resume_50k',
        pretrain_target=50000,finetune_target=10000,save_every=5000,
        next_steps='existing trainer -> resume full 30k to50k -> Top10 10k from50k EMA')
    def save():(out/'status.json').write_text(json.dumps(status,indent=2))
    def checkpoint(step):
        files=list(Path(original['output']).glob(f'*/checkpoints/{step:07d}.pt'))
        if len(files)!=1:raise ValueError('Missing/ambiguous checkpoint: '+str(step))
        return files[0]
    def run(config,logname,phase,resume=None):
        command=[sys.executable,'-u','-m','src.unet.train','--config',str(config)]
        if resume:command.extend(['--resume',str(resume)])
        with (out/logname).open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            status.update(phase=phase,train_pid=process.pid);save()
            if process.wait():raise RuntimeError('Stage failed: '+phase)
    save()
    try:
        while proc.exists() and '\nState:\tZ' not in (proc/'status').read_text():time.sleep(30)
        source=checkpoint(30000)
        if 'DONE checkpoint_step=' not in (out/'train.log').read_text():raise ValueError('Original trainer did not finish cleanly')
        run(prepath,'pretrain_50k.log','pretraining_30k_to_50k',source)
        source=checkpoint(50000)
        fine['initialize_from']=str(source);Config(**fine).validate()
        finepath.write_text(json.dumps(fine,indent=2))
        status['source_checkpoint']=str(source);save()
        run(finepath,'top10_train.log','top10_finetuning')
        status.update(phase='completed',ended=time.strftime('%Y-%m-%d %H:%M:%S'));save()
    except BaseException as error:
        status.update(phase='failed',error=repr(error));save();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool-root',default='/root/Workspace/xy/UNIFIED_RAW')
    p.add_argument('--out',default='assets/generic48_kxl_top10_20261022');p.add_argument('--workers',type=int,default=16)
    p.add_argument('--resume-preparation',action='store_true')
    p.add_argument('--extend-running',action='store_true')
    a=p.parse_args();out=Path(a.out)
    if a.extend_running:return extend_running(out)
    if a.resume_preparation:
        previous=json.loads((out/'status.json').read_text())
        if previous['phase']!='failed' or (out/'latents').exists():raise ValueError('Only failed, pre-encoding preparation may restart')
    out.mkdir(parents=True,exist_ok=a.resume_preparation)
    status=dict(phase='build_std',started=time.strftime('%Y-%m-%d %H:%M:%S'),next_steps='STD -> exclude heldout leakage -> VAE mode cache -> 48M FM')
    def save():(out/'status.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        cfg=prepare(a,out,status,save)
        status.update(phase='encode');save()
        with (out/'encode.log').open('w') as log:
            subprocess.run([sys.executable,'-u','tools/encode_aug_latents.py','--csv',str(out/'encode.csv'),
                '--out',str(out/'latents'),'--vae',cfg.vae_path,'--img-root','.',
                '--encode-mode','mode','--scaling-factor','0.18215','--batch','64'],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        provenance=json.loads((out/'latents/PROVENANCE.json').read_text())
        if (provenance['n_rows']!=len(read_rows(out/'encode.csv')) or provenance['scaling_factor']!=.18215 or
            provenance['shift_factor']!=0 or provenance['encode_mode']!='mode'):raise ValueError('VAE cache contract/coverage failed')
        status.update(phase='training');save()
        with (out/'train.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-u','-m','src.unet.train','--config',str(out/'train_config.json')],
                stdout=log,stderr=subprocess.STDOUT)
            status['train_pid']=process.pid;save();code=process.wait()
            if code:raise RuntimeError('Training exit '+str(code))
        checkpoints=list((out/'runs').rglob('*.pt'))
        checkpoints=[p for p in checkpoints if p.name=='0050000.pt']
        if len(checkpoints)!=1:raise ValueError('Expected exactly one generic 50k checkpoint: '+str(checkpoints))
        target=json.loads((out/'top10_config.json').read_text())
        target['initialize_from']=str(checkpoints[0]);Config(**target).validate()
        (out/'top10_config.json').write_text(json.dumps(target,indent=2))
        status.update(phase='top10_finetuning',source_checkpoint=str(checkpoints[0]));save()
        with (out/'top10_train.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-u','-m','src.unet.train','--config',str(out/'top10_config.json')],
                stdout=log,stderr=subprocess.STDOUT)
            status['train_pid']=process.pid;save();code=process.wait()
            if code:raise RuntimeError('Top10 training exit '+str(code))
        status.update(phase='completed',ended=time.strftime('%Y-%m-%d %H:%M:%S'));save()
    except BaseException as e:
        status.update(phase='failed',error=repr(e));save();raise


if __name__=='__main__':main()
