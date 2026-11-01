"""Attach later stages to a specific live Day1 process; never restart an observation timeout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import subprocess
import sys
from tools.three_day.run import Queue
from tools.three_day.contracts import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--day1-pid',required=True,type=int)
    p.add_argument('--detach',action='store_true');a=p.parse_args();root=Path(a.root)
    if a.detach:
        with (root/'later_detached.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','tools.three_day.later.run',
                *[v for v in sys.argv[1:] if v!='--detach']],stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        print('LATER_SUPERVISOR_PID',child.pid,flush=True);return
    lock=(root/'later_supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(root/'later_supervisor.json',dict(pid=os.getpid(),day1_pid=a.day1_pid,state='waiting_for_live_day1'))
    while True:
        state=json.loads((root/'status.json').read_text())
        if state['state']=='day1_complete':break
        try:os.kill(a.day1_pid,0)
        except ProcessLookupError:raise RuntimeError('Day1 process ended without completed artifact state; do not continue')
        if state['state']=='failed':raise RuntimeError('Day1 failed; later stages held')
        time.sleep(30)
    # The lock is released on Day1 exit; wait for that precise process, not a status file alone.
    while True:
        try:os.kill(a.day1_pid,0)
        except ProcessLookupError:break
        time.sleep(1)
    queue=Queue(root)
    from .day2 import run
    run(queue)


if __name__=='__main__':main()
