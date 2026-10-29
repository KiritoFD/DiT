"""Resume only future phases, retaining completed first-stage results and clock."""
import argparse
import fcntl
from pathlib import Path
import time
from .plan import prepare
from tools.style_matrix.generation2.runner import Controller
from tools.style_matrix.generation2.plan import read
from tools.style_matrix.runtime.common import save
from tools.style_matrix.runtime.serial import gpu_idle


class LaterController(Controller):
    def mature(self,arm):
        self.job(arm+'_60_matched')
        return self.job(arm+'_75_matched',final=True)

    def run_s(self):
        cold=read(self.root/'tasks/S9_50000_screen/score.json')
        gate=read(self.root/'pair_gate.json');reject=gate['reject_pair_extension']
        self.phase(2,'S10 cold 50k; S9 control/Pair maturity; matched LR baselines')
        s10=self.job('S10');baselines={arm:self.job(arm,stop=50000) for arm in ('S0','S4')}
        base0=baselines['S0']
        def degraded(value):
            return (value['content']['ssim']<base0['content']['ssim']-.02 and
                    value['content']['lpips']>base0['content']['lpips']+.015 and
                    value['writer']['margin_mean']<=base0['writer']['margin_mean'])
        if degraded(cold) and degraded(s10):
            best='S9' if cold['content']['lpips']<=s10['content']['lpips'] else 'S10'
            save(self.root/'capacity_gate.json',dict(both_degraded=True,independent_seed=best+'_seed91'))
            self.phase(3,'Both cold designs degraded; independent seed without old-baseline extension')
            self.job(best,final=True);self.job(best+'_seed91',final=True);return
        fm=self.job('S9FM_75')
        if not reject:self.job('S9Pair_75')
        self.phase(3,'Mature architecture controls with the same 50k/60k LR restart points')
        if degraded(s10):self.job('S10',final=True)
        else:self.mature('S10')
        chosen='S4' if fm['writer']['margin_mean']>=base0['writer']['margin_mean']+.01 else 'S0'
        save(self.root/'baseline_selection.json',dict(chosen=chosen,matched_lr=True))
        self.mature(chosen);self.job('S9FM_75',final=True)
        if not reject:self.job('S9Pair_75',final=True)

    def run_j(self):
        # Handoff is after S11@50k: J screening and optional 12k extensions are complete.
        gate=read(self.root/'decoder_gate.json')
        for arm in gate['selected_pair']:
            path=self.root/'runs'/arm/'checkpoints/0012000.pt.done'
            if not path.exists():raise ValueError('Selected Decoder extension is incomplete: '+arm)
        self.phase(3,'S11 60k/75k matched LR control, then independent 50k seed')
        self.mature('S11');self.job('S11_seed91',final=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',choices=['4090','48'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--resident-pid',type=int,default=0);a=p.parse_args()
    controller=LaterController(a.machine,a.root,a.resident_pid)
    with open('/tmp/dit_gpu0.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle(a.resident_pid)
        prepare(a.root,a.machine)
    controller.specs={p.stem:read(p) for p in (Path(a.root)/'specs').glob('*.json')}
    receipt=Path(a.root)/'later_handoff.json'
    data=read(receipt) if receipt.exists() else {}
    save(receipt,dict(data,state='activated',activated_unix=time.time(),deadline_unix=controller.deadline))
    try:controller.run()
    except SystemExit as exc:
        save(controller.root/'status.json',dict(state='budget_complete' if time.time()>=controller.deadline else 'stopped',returncode=exc.code));raise
    except BaseException as exc:
        save(controller.root/'failure.json',dict(error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':main()
