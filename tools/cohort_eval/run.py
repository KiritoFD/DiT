"""Evaluate one U-Net checkpoint on one explicit collection, saving all artifacts."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .datasets import DEFAULT_ROOT, read_rows, write_rows, seeded_rows, image_key, sha, resolve_collection


def evaluate(checkpoint, collection, output, *, root=DEFAULT_ROOT, probes=None,
             state='ema', batch=4, steps=50, cfg=1., seed=90, poster=False, vae_batch=1,std_sigma=.65,std_root=None):
    import torch
    from src.unet.io import load, eval_args
    from src.eval.in_mem_eval import run_in_mem_eval
    import math
    if state not in ('ema','raw') or batch < 1 or vae_batch < 1 or steps < 1 or cfg < 0 or not math.isfinite(std_sigma) or std_sigma<0:
        raise ValueError('Invalid state/batch/steps/cfg')
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    source = resolve_collection(collection, root, probes)
    items = seeded_rows(read_rows(source), seed)
    if len({image_key(r) for r in items}) != len(items):
        raise ValueError('Duplicate images in evaluation collection')
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model, config, ckpt = load(str(checkpoint),'cuda',use_ema=state=='ema')
    for field, mapping in (('calligrapher_id',config.callig_remap_json),
                           ('character_id',config.char_remap_json),('script_id',config.font_remap_json)):
        known = json.loads(Path(mapping).read_text())
        missing = {r[field] for r in items}-set(known)
        if missing:raise ValueError(f'Checkpoint does not support {field}: {sorted(missing)}')
    train = {image_key(r) for r in read_rows(config.data_csv)}
    overlap = sum(image_key(r) in train for r in items)
    kind = collection.split('/')[-1]
    if kind in ('train_equal','train_half') and overlap != len(items):
        raise ValueError('Training probe includes samples outside this checkpoint training CSV')
    if kind in ('test','validation') and overlap:
        raise ValueError('Held-out alias overlaps this checkpoint training CSV')
    output.mkdir(parents=True)
    frozen = output/'dataset.csv'
    write_rows(frozen,items)  # UTF-8 without BOM: native raw CSV retains image metadata.
    identity = dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        step=int(ckpt['train_steps']),state=state,collection=collection,source_csv=str(source),
        source_csv_sha256=sha(source),effective_csv_sha256=sha(frozen),n=len(items),
        training_membership=overlap,training_csv=config.data_csv,training_csv_sha256=sha(config.data_csv),
        cfg=cfg,steps=steps,sampler='heun',fallback_noise_seed=seed,std_soft_eval_sigma=std_sigma,
        effective_std_soft_root=str(std_root or config.std_soft_root),
        noise='existing per-image seeds preserved; missing seeds derived from image path',config=asdict(config))
    (output/'identity.json').write_text(json.dumps(identity,indent=2,ensure_ascii=False),encoding='utf-8')
    # Native set syntax uses ':' delimiters; avoid Windows drive-letter ambiguity.
    eval_csv=os.path.relpath(frozen)
    config.eval_csv=eval_csv;config.eval_steps=steps
    args=eval_args(config);args.eval_cfg=cfg;args.eval_use_ema=state=='ema'
    args.std_soft_eval_sigma=std_sigma
    if std_root is not None:args.std_soft_root=str(std_root)
    args.in_mem_eval_sets=f'evaluation:{eval_csv}:{len(items)}'
    args.in_mem_eval_batch=batch;args.in_mem_eval_vae_batch=vae_batch
    args.in_mem_eval_poster=poster
    model.to(memory_format=torch.channels_last)
    with torch.inference_mode():
        run_in_mem_eval(model,args,int(ckpt['train_steps']),'cuda',results_dir=str(output))
    result=read_rows(output/'eval_stdskel_summary.csv')[-1]
    if int(result['n'])!=len(items) or not result['lpips_mean'] or not result['ink_iou_mean']:
        raise RuntimeError('Incomplete evaluation metrics')
    result.update(collection=collection,state=state,training_membership=overlap)
    (output/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--collection',required=True,help='topN/test, topN/train_equal, topN/train_half, or CSV path')
    p.add_argument('--out',required=True);p.add_argument('--root',default=str(DEFAULT_ROOT))
    p.add_argument('--probes');p.add_argument('--state',choices=('ema','raw'),default='ema')
    p.add_argument('--batch',type=int,default=4);p.add_argument('--steps',type=int,default=50)
    p.add_argument('--vae-batch',type=int,default=1)
    p.add_argument('--cfg',type=float,default=1.);p.add_argument('--seed',type=int,default=90)
    p.add_argument('--std-sigma',type=float,default=.65,help='Fixed STD blur; 0 uses the unblurred raster')
    p.add_argument('--std-root',help='Explicit alternative STD cache; checkpoint identity stays unchanged')
    p.add_argument('--poster',action='store_true')
    a=p.parse_args()
    evaluate(a.checkpoint,a.collection,a.out,root=a.root,probes=a.probes,state=a.state,
             batch=a.batch,steps=a.steps,cfg=a.cfg,seed=a.seed,poster=a.poster,vae_batch=a.vae_batch,
             std_sigma=a.std_sigma,std_root=a.std_root)


if __name__=='__main__':main()
