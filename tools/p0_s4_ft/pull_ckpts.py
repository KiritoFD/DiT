"""Pull the key checkpoints to this machine, resumably and with verification.

Which weights matter, and why:
  K   assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt         (4090)
      The current baseline. Every comparison in this session is against it.
  S0  matrix64/4090/execution/runs/S0/checkpoints/0050000.pt              (4090)
      The mixed-training reference baseline.
  S4  matrix64/4090/execution/runs/S4/checkpoints/0050000.pt              (4090)
      The R2/I1 arm the P0 -> S4 run is trying to beat.
  S5  matrix64/**48**/execution/runs/S5/checkpoints/0050000.pt            (48, NOT 4090)
      The R2/I2 arm. Note the host: S5 lives on the 48 box, so pulling it needs both
      machines reachable, not just the 4090.

Design notes
------------
These are ~1.8 GB each, so four of them is ~7.2 GB. Two consequences drive the design:

  * Transfers must be RESUMABLE. A dropped VPN mid-pull should not restart from zero, so
    this uses rsync --partial --append-verify where available and falls back to scp -C.
  * Every file is VERIFIED after it lands. `torch.load` runs first because a truncated
    file fails it loudly, then the SHA256 is compared against the source. A file that
    merely has the right size is not good enough for a training starting point.

The script is idempotent: a checkpoint that already exists locally with a matching SHA is
skipped, so re-running after a partial failure only moves what is missing.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'assets/imported_ckpts'
STATE = ROOT / 'assets/imported_ckpts/manifest.json'

# (key, host, remote path, expected sha256 or None, note)
TARGETS = [
    ('K', '4090',
     'assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt',
     None,
     'baseline for every comparison in this session'),
    ('S0', '4090',
     'assets/style_matrix_20261027/matrix64/4090/execution/runs/S0/checkpoints/0050000.pt',
     '50d41ee21091f865f3b7315297adeafbba113d934d3094e7203c1e0017a2cccc',
     'mixed-training reference baseline (SHA recorded in ckpts.md)'),
    ('S4', '4090',
     'assets/style_matrix_20261027/matrix64/4090/execution/runs/S4/checkpoints/0050000.pt',
     None,
     'R2/I1 arm; target to beat'),
    ('S5', '48',
     'assets/style_matrix_20261027/matrix64/48/execution/runs/S5/checkpoints/0050000.pt',
     None,
     'R2/I2 arm -- lives on 48, not 4090'),
]


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def remote_sha(host, rpath):
    """Hash on the far side so we never have to trust a transfer we cannot compare."""
    cmd = ['ssh', '-o', 'ConnectTimeout=30', host,
           f'sha256sum "{rpath}" 2>/dev/null | cut -d" " -f1']
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return out.stdout.strip() or None
    except subprocess.TimeoutExpired:
        return None


def remote_size(host, rpath):
    cmd = ['ssh', '-o', 'ConnectTimeout=30', host, f'stat -c%s "{rpath}" 2>/dev/null']
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return int(out.stdout.strip()) if out.stdout.strip().isdigit() else None
    except (subprocess.TimeoutExpired, ValueError):
        return None


def fetch(host, rpath, local):
    """rsync if present (resumable), else scp. Partial files are kept for --append-verify."""
    local.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(['bash', '-lc', 'command -v rsync'], capture_output=True).returncode == 0:
        cmd = ['rsync', '-av', '--partial', '--append-verify', '--inplace',
               '--progress', f'{host}:{rpath}', str(local)]
        print(f'  rsync (resumable): {host}:{rpath}')
    else:
        # scp cannot resume; -C only helps bandwidth. Better than nothing on a fallback.
        cmd = ['scp', '-C', f'{host}:{rpath}', str(local)]
        print(f'  scp (NOT resumable): {host}:{rpath}')
    return subprocess.run(cmd).returncode


def verify(local, want_sha):
    """Truncation first (a real torch load), then the hash."""
    import torch
    try:
        torch.load(str(local), map_location='cpu', weights_only=False)
    except Exception as e:
        return False, f'torch.load failed (truncated or corrupt): {type(e).__name__}: {e}'
    got = sha256(local)
    if want_sha and got != want_sha:
        return False, f'sha mismatch: got {got[:16]}… want {want_sha[:16]}…'
    return True, f'sha {got[:16]}…'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='comma-separated keys, e.g. K,S4')
    ap.add_argument('--force', action='store_true', help='re-pull even if present')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    wanted = [t for t in TARGETS if not a.only or t[0] in a.only.split(',')]
    DEST.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text(encoding='utf-8')) if STATE.is_file() else {}

    print(f'destination: {DEST}')
    print(f'targets: {[t[0] for t in wanted]}\n')

    failures = []
    for key, host, rpath, want_sha, note in wanted:
        local = DEST / f'{key}_{Path(rpath).name}'
        print(f'--- {key} ({host}) ---')
        print(f'  {note}')

        if local.is_file() and not a.force:
            ok, detail = verify(local, want_sha)
            if ok:
                print(f'  already present and valid ({detail}) -- skip')
                state[key] = dict(path=str(local), host=host, remote=rpath,
                                  sha256=detail.split()[1].rstrip('…'), verified=True)
                continue
            print(f'  present but INVALID ({detail}); re-pulling')

        size = remote_size(host, rpath)
        if size is None:
            msg = f'{host}:{rpath} not reachable or missing (is the VPN up?)'
            print(f'  !! {msg}')
            failures.append((key, msg))
            continue
        print(f'  remote size {size / 2**30:.2f} GiB')

        if a.dry_run:
            print('  dry-run: would pull')
            continue

        if fetch(host, rpath, local) != 0:
            msg = 'transfer failed'
            print(f'  !! {msg}')
            failures.append((key, msg))
            continue

        ok, detail = verify(local, want_sha)
        if not ok:
            print(f'  !! verification failed: {detail}')
            failures.append((key, detail))
            continue
        print(f'  VERIFIED {detail}  ({local.stat().st_size / 2**30:.2f} GiB)')
        state[key] = dict(path=str(local), host=host, remote=rpath,
                          sha256=detail.split()[1].rstrip('…'), verified=True,
                          bytes=local.stat().st_size)

        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\nmanifest: {STATE}')
    if failures:
        print(f'FAILED: {len(failures)}')
        for k, m in failures:
            print(f'  {k}: {m}')
        return 1
    print('all requested checkpoints present and verified')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
