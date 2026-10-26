"""Measure common initial FM/image/reconstruction gradients on paired real GT."""
import argparse
import json
from pathlib import Path
import logging
import torch
from src.unet.io import training_args,diffusion
from src.unet.data import loader,StdBank
from src.unet.joint.objective import JointDecoderObjective,GTImages
from ..runtime.common import construct,step,save,sha,seed


def norm(parameters):
    values=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True)
    p.add_argument('--image-ratio',type=float,default=.1)
    p.add_argument('--reconstruction-weight',type=float,default=1.)
    a=p.parse_args();target=Path(a.out)
    if target.exists():raise FileExistsError('Calibration is immutable; choose a new output')
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    spec=json.loads(Path(a.spec).read_text());model,cfg,transfer=construct(spec)
    if spec['joint']['arm']!='J3':raise ValueError('Calibrate once on the common J3 source, reuse in J0--J3')
    logging.basicConfig(level=logging.INFO);log=logging.getLogger('calibration')
    args=training_args(cfg,target.parent);args.ckpt_every=1000;args.epoch_steps=1000
    dataset,_,batches,_,_=loader(cfg,args,log,0);batch=next(iter(batches))
    z=batch['latent'].cuda().float().to(memory_format=torch.channels_last)
    clean=batch['std_soft'].cuda().float()
    if batch['std_soft'].dtype==torch.uint8:clean=clean/255.
    if clean.ndim==3:clean=clean[:,None]
    g=StdBank(cfg)(batch,'cuda',0,clean=clean)
    w=batch['y_callig_raw'].cuda();s=batch['y_script'].cuda();jc=spec['joint']
    objective=JointDecoderObjective('J3',cfg,image_micro=jc['image_micro'],image_batch=jc['image_batch'],img_weight=1.,rec_weight=0.)
    # Wrapper datasets need the authoritative CSV, never guessed positional matches.
    import csv
    reader=GTImages(list(csv.DictReader(open(cfg.data_csv,encoding='utf-8-sig'))))
    model.train();model.zero_grad(set_to_none=True);seed(cfg.seed+10000)
    fm,conditions=step(model,diffusion(cfg),z,g,w,s,cfg.microbatch,return_conditions=True)
    fm_norm=norm(model.parameters());model.zero_grad(set_to_none=True)
    objective.reconstruction=False
    image_stats=objective.backward(model,z,g,conditions,0,img_ids=batch['img_id'].tolist(),reader=reader)
    image_norm=norm(model.parameters());decoder_image=norm(objective.vae.decoder.parameters())
    if not fm_norm>0 or not image_norm>0:raise ValueError('Missing FM/image gradient')
    image_weight=min(1.,a.image_ratio*fm_norm/image_norm)
    model.zero_grad(set_to_none=True);objective.vae.zero_grad(set_to_none=True)
    objective.image=False;objective.reconstruction=True;objective.rec_weight=1.
    rec_stats=objective.backward(model,z,g,conditions,0,img_ids=batch['img_id'].tolist(),reader=reader)
    decoder_rec=norm(objective.vae.decoder.parameters())
    if norm(model.parameters())!=0 or decoder_rec<=0:raise ValueError('Reconstruction leaked into U-Net')
    save(target,dict(source_checkpoint_sha256=sha(spec['source']),vae_frozen_sha256=objective.frozen_sha,
        train_csv_sha256=sha(cfg.data_csv),image_weight=image_weight,reconstruction_weight=a.reconstruction_weight,
        requested_image_gradient_ratio=a.image_ratio,actual_image_gradient_ratio=image_weight*image_norm/fm_norm,
        fm_gradient_norm=fm_norm,image_gradient_norm=image_norm,decoder_image_gradient_norm=decoder_image,
        decoder_reconstruction_gradient_norm=decoder_rec,image_samples=jc['image_batch'],image_micro=jc['image_micro'],
        image_stats=image_stats,reconstruction_stats=rec_stats,paired_gt_checked=objective.pair_checked,
        img_ids=batch['img_id'].tolist(),transfer=transfer,optimizer_steps=0,calibration_only=True))


if __name__=='__main__':main()
