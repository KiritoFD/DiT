"""One capacity check, then launch X1; no implicit queue or budget changes."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--compiled-capacity',action='store_true')
    args=parser.parse_args()
    suffix='' if args.batch==512 else f'_b{args.batch}'
    if args.compiled_capacity:suffix+='_compiled'
    root=Path('assets/results/extreme48m_top20_x1_launch_20261022'+suffix)
    root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('src/unet/configs/extreme48m/x1_top20.json').read_text())
    config['batch']=args.batch
    state=dict(task='Top20 frozen independent table, 48.326M X1',pid=__import__('os').getpid(),
               started=datetime.now().isoformat(),phase='capacity_check',batch=config['batch'],
               stop_at_step=config['stop_at_step'],lr_decay_steps=config['lr_decay_steps'],
               next_steps='Check first training steps, memory and throughput; retain validation/checkpoints')
    def save():
        (root/'status.json').write_text(json.dumps(state,indent=2))
    save()
    probe=dict(config,compile=args.compiled_capacity)
    (root/'capacity_config.json').write_text(json.dumps(probe,indent=2))
    with (root/'capacity.log').open('w') as log:
        code=subprocess.call([sys.executable,'-u','-m','src.unet.train','--config',str(root/'capacity_config.json'),
                              '--benchmark','3'],stdout=log,stderr=subprocess.STDOUT)
    if code:
        state.update(phase='failed_capacity',returncode=code);save()
        raise RuntimeError('Capacity check failed; inspect capacity.log, no silent batch change')
    (root/'train_config.json').write_text(json.dumps(config,indent=2))
    with (root/'train.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','src.unet.train','--config',str(root/'train_config.json')],
                                 stdout=log,stderr=subprocess.STDOUT)
        state.update(phase='training',train_pid=process.pid);save()
        code=process.wait()
    state.update(phase='completed' if code==0 else 'failed_training',returncode=code,
                 ended=datetime.now().isoformat());save()
    if code:raise RuntimeError('Training failed; inspect train.log')


if __name__=='__main__':main()
