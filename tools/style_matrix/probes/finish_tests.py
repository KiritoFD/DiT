"""Historical 24-cell capacity controller; not used by the current S/J plan."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',required=True)
    p.add_argument('--wait-pid',type=int,required=True);p.add_argument('--resident-pid',type=int,default=0)
    a=p.parse_args()
    while Path(f'/proc/{a.wait_pid}').exists():
        cmd=Path(f'/proc/{a.wait_pid}/cmdline').read_bytes()
        if b'tools.style_matrix.probes.micro_search' not in cmd:break
        time.sleep(10)
    root=Path('assets/style_matrix_20261026')
    assert json.loads((root/'micro_search_b512'/a.machine/'status.json').read_text())['state']=='complete'
    command=[sys.executable,'-u','-m','tools.style_matrix.probes.batch_sweep','--machine',a.machine,'--selected-defaults']
    if a.resident_pid:command+=['--resident-pid',str(a.resident_pid)]
    subprocess.run(command,check=True)
    if a.machine=='48':
        subprocess.run([sys.executable,'-u','-m','tools.style_matrix.probes.accumulation_check',
            '--spec',str(root/'specs'/'48'/'R0I0_repeat.json'),'--out',str(root/'accumulation_check_b512.json'),
            '--batch','512'],check=True)
        subprocess.run([sys.executable,'-u','-m','tools.style_matrix.probes.accumulation_check',
            '--spec',str(root/'specs'/'48'/'R0I0_repeat.json'),'--out',str(root/'accumulation_check_fp32.json'),'--fp32'],check=True)


if __name__=='__main__':main()
