"""Complete cache -> fixed-reference validation -> ordinary shared U-Net training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from .prepare import read,write
from src.unet.reference.data import ReferenceBank
from src.unet.io import Config


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='assets/reference_kxl_20261023')
    p.add_argument('--steps',type=int,default=30000);p.add_argument('--batch',type=int,default=256)
    a=p.parse_args();out=Path(a.out);cache=out/'features.npz'
    status=dict(phase='waiting_cache',pid=__import__('os').getpid())
    def save():(out/'pipeline_status.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        while not cache.with_suffix('.json').exists():time.sleep(5)
        bank=ReferenceBank(cache)
        original=read(out/'validation.csv');validation=[]
        for row in original:
            pool=bank.pools.get((int(row['style_writer_id']),{'楷':0,'行':1,'隶':2}[row['script']]),[])
            if any(bank.chars[i]!=ord(row['character']) for i in pool):validation.append(row)
        if not validation:raise ValueError('No valid reference-conditioned validation rows')
        write(out/'eval.csv',validation)
        base=json.loads(Path('assets/generic48_kxl_top10_20261022/pretrain_50k_config.json').read_text())
        base.update(channels=[192,384,576],num_res_blocks=4,reference_dim=128,
            reference_cache=str(cache),reference_max_refs=2,reference_con_weight=.05,
            reference_temperature=.1,reference_hard_weight=2.,writer_table_file='',
            data_csv=str(out/'train.csv'),eval_csv=str(out/'eval.csv'),output=str(out/'runs'),
            max_steps=a.steps,stop_at_step=a.steps,lr_decay_steps=a.steps,batch=a.batch,
            lr=6e-5,warmup=500,ckpt_every=5000,eval_every=5000,log_every=25,compile=False)
        config=Config(**base);config.validate()
        (out/'config.json').write_text(json.dumps(asdict(config),indent=2))
        manifest=dict(train_csv_sha256=hashlib.sha256((out/'train.csv').read_bytes()).hexdigest(),
            eval_n=len(validation),eval_excluded_no_reference=len(original)-len(validation),
            input='STD + Script + frozen DINOv2 reference CLS, trainable 128D head',
            generator_writer_id=False,reference_pool='train.csv ONLY; different character, same writer/script',
            sampling='50% same-script hard 2x2 quartets + 50% same-script writer-balanced crosschar pairs',
            loss='FM + .05 full-batch SupCon, hard-negative denominator weight2',
            reference_cfg=False)
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
        status.update(phase='training',config=str(out/'config.json'),eval_n=len(validation));save()
        with (out/'training.log').open('a') as log:
            subprocess.run([sys.executable,'-u','-m','src.unet.train','--config',str(out/'config.json')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        status['phase']='completed';save()
    except BaseException as error:
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
