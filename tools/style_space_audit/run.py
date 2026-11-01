"""Finite audit chain; no generator training is launched."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from tools.three_day.contracts import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    state=dict(state='running',started=time.strftime('%F %T'),phases={});save(root/'job.json',state)
    stages=[('metadata','data',[]),('deep_shared','features',['--deep']),('deep_native','features',['--deep','--native-std']),
            ('features','features',[]),('analysis','analysis',[]),('heldout_VAE','features',['--heldout-roundtrip']),('gradient','gradient',[])]
    for name,module,extra in stages:
        expected={'deep_shared':root/'Deep114/g99.png','deep_native':root/'Deep114_native/g99.png','gradient':root/'gradient/results.json'}.get(name)
        if expected is not None and expected.exists():
            state['phases'][name]={'state':'existing_artifact','path':str(expected)};save(root/'job.json',state);continue
        command=[sys.executable,'-u','-m','tools.style_space_audit.'+module,'--root',str(root),*extra]
        state['current']=name;state['phases'][name]={'state':'running','command':command};save(root/'job.json',state)
        with (root/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(state='failed',returncode=result.returncode);save(root/'job.json',state);raise SystemExit(result.returncode)
        state['phases'][name]['state']='done';save(root/'job.json',state)
    for artifact in ('results.json','writer_direction.json','matched_examples.png','gradient/results.json','dino_recomputed.json','heldout_vae_roundtrip.json'):
        if not (root/artifact).exists():raise FileNotFoundError(artifact)
    state.update(state='complete',finished=time.strftime('%F %T'));save(root/'job.json',state)


if __name__=='__main__':main()
