"""Copy the compact poster's S4/S5 cells into a staging dir for transfer.

The 21-row table is assembled on the workstation (where the 187-panel columns live),
but S4/S5 only exist on the training box, so this exports just those cells. It is a
few dozen small PNGs, not a bulk transfer.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M64 = 'assets/style_matrix_20261027/matrix64'
DIRS = {
    'S4_50k': f'{M64}/4090/execution/runs/S4/evaluations/0050000/heldout/eval_samples_ctrl',
    'S5_50k': f'{M64}/48/execution/runs/S5/evaluations/0050000/heldout_s/eval_samples_ctrl',
}
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/s45')


def png_dir(cdir):
    base = ROOT / cdir
    best, best_key = None, None
    for cand in [base, *[p for p in base.rglob('*') if p.is_dir()]]:
        n_g = sum(1 for f in cand.glob('g[0-9]*.png') if f.stem[1:].isdigit())
        n_gt = sum(1 for f in cand.glob('gt*.png'))
        if n_g == 0:
            continue
        key = (1 if n_gt > 0 else 0, n_g)
        if best_key is None or key > best_key:
            best, best_key = cand, key
    return best


def main():
    cells = json.loads((ROOT / 'tools/poster_20261101/cells.json').read_text(encoding='utf-8'))['cells']
    idx = [c['idx100'] for c in cells]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cdir in DIRS.items():
        src = png_dir(cdir)
        if src is None:
            raise SystemExit(f'{name}: no panel dir under {cdir}')
        dst = OUT / name
        dst.mkdir(exist_ok=True)
        got = 0
        for i in idx:
            p = src / f'g{i}.png'
            if p.exists():
                shutil.copy2(p, dst / f'g{i}.png')
                got += 1
        print(f'{name}: {got}/{len(idx)} from {src}')
        if got != len(idx):
            raise SystemExit(f'{name}: missing cells')


if __name__ == '__main__':
    main()
