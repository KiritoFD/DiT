"""Reselect existing measurements; never launches a GPU workload."""
import argparse
import json
from pathlib import Path
from ..runtime.common import save,sha


def select(path,cap):
    result=[]
    for family in json.loads(Path(path).read_text()):
        candidates=[r for r in family['candidates'] if r['peak_reserved_GiB']<=cap]
        if not candidates:raise ValueError(f'No measured candidate below {cap}GiB')
        best=max(candidates,key=lambda r:r['step_s'])
        summary={k:best[k] for k in ('arm','machine','global_batch','microbatch','microbatch_sizes',
            'step_s','samples_s','peak_allocated_GiB','peak_reserved_GiB','measured_steps','gradient_checkpointing','compile')}
        result.append(dict(injection=family['injection'],chosen=summary,
            criterion='fastest measured below safety cap; existing results only',safe_cap_GiB=cap,
            source_file=str(path),source_sha256=sha(path),safe_candidates=len(candidates)))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--cap',required=True,type=float);a=p.parse_args()
    result=select(a.input,a.cap);save(a.out,result)
    for r in result:
        b=r['chosen'];print(r['injection'],b['microbatch'],b['step_s'],b['peak_reserved_GiB'])


if __name__=='__main__':main()
