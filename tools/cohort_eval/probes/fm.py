"""Deterministic posterior-mode FM MSE by time bucket and content/source strata."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import torch

from tools.cohort_eval.datasets import read_rows,write_rows,seeded_rows,resolve_collection,image_key,sha


def run(args):
    from src.unet.io import load,diffusion,eval_args
    from src.eval.in_mem_eval import _get_cache
    from src.eval.decoder import load_eval_vae
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    model,config,ckpt=load(args.checkpoint,'cuda',use_ema=True)
    model.eval();process=diffusion(config)
    vae=load_eval_vae('cuda',config.vae_path)
    train=read_rows(config.data_csv)
    chars={r['character'] for r in train}
    slots={(r['calligrapher_id'],r['script_id'],r['character']) for r in train}
    sources={r.get(args.source_column,'') for r in train} if args.source_column else set()
    maps={k:{str(a):int(b) for a,b in json.loads(Path(getattr(config,f'{k}_remap_json')).read_text()).items()}
          for k in ('callig','font')}
    records=[];identities=[]
    with torch.inference_mode():
        for collection in args.collections.split(','):
            source=resolve_collection(collection)
            items=seeded_rows(read_rows(source))
            if args.limit:items=items[:args.limit]
            frozen=out/(collection.replace('/','_')+'.csv');write_rows(frozen,items)
            ev=eval_args(config)
            cache=_get_cache(str(frozen),len(items),None,'',ev)
            targets=[]
            for i in range(0,len(items),args.batch):
                x=cache['gts'][i:i+args.batch].cuda().float()
                z=vae.encode(x).latent_dist.mode()
                targets.append(((z-config.vae_shift_factor)*config.vae_scaling_factor).cpu())
            targets=torch.cat(targets)
            for midpoint in (.05,.15,.25,.35,.45,.55,.65,.75,.85,.95):
                for i in range(0,len(items),args.batch):
                    selected=items[i:i+args.batch];b=len(selected)
                    writer=torch.tensor([maps['callig'][r['calligrapher_id']] for r in selected],device='cuda')
                    script=torch.tensor([maps['font'][r['script_id']] for r in selected],device='cuda')
                    kwargs=dict(y_callig=writer,y_script=script,g=cache['skels_latent'][i:i+b].cuda())
                    t=torch.full((b,),midpoint,device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        losses=process.training_losses(model,targets[i:i+b].cuda(),t,model_kwargs=kwargs,
                            noise=cache['noise'][i:i+b].cuda())['loss'].float().cpu().tolist()
                    for row,loss in zip(selected,losses):
                        records.append(dict(collection=collection,image_path=image_key(row),character=row['character'],
                            writer=row['calligrapher'],script=row['script'],t_mid=midpoint,fm_mse=loss,
                            seen_character=row['character'] in chars,
                            seen_writer_script_character=(row['calligrapher_id'],row['script_id'],row['character']) in slots,
                            source_label=row.get(args.source_column,'') if args.source_column else 'not_available',
                            seen_source_label=(row.get(args.source_column,'') in sources) if args.source_column else 'not_available'))
            identities.append(dict(collection=collection,source=str(source),sha256=sha(source),n=len(items)))
    groups=defaultdict(list)
    for r in records:
        names=['all','seen_character' if r['seen_character'] else 'unseen_character',
               'seen_writer_script_character' if r['seen_writer_script_character'] else 'unseen_writer_script_character',
               'writer:'+r['writer']]
        if args.source_column:names.append('seen_source_label' if r['seen_source_label'] else 'unseen_source_label')
        for name in names:groups[(r['collection'],r['t_mid'],name)].append(r['fm_mse'])
    summary=[dict(collection=c,t_mid=t,group=g,n=len(v),fm_mse=mean(v)) for (c,t,g),v in groups.items()]
    write_rows(out/'per_image.csv',records);write_rows(out/'summary.csv',summary)
    (out/'identity.json').write_text(json.dumps(dict(checkpoint=args.checkpoint,checkpoint_sha256=sha(args.checkpoint),
        step=ckpt['train_steps'],state='ema',collections=identities,source_column=args.source_column,
        target='VAE posterior mode, (z-shift)*scale, same preprocessing as native eval GT',
        noise='fixed per-image noise shared across checkpoints and all t buckets',
        t='0=near data, 1=near noise; ten bucket midpoints, not integrated expected training loss',
        source_caveat='Source labels are not proof of work-level holdout; blank source_column records unavailable'),indent=2))
    print('FM_PROBE_DONE',len(records),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--collections',default='top20/validation,top20/train_half')
    p.add_argument('--batch',type=int,default=16);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--source-column',default='')
    run(p.parse_args())


if __name__=='__main__':main()
