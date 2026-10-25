"""Binary SSH download for generated archives; avoids Windows SFTP stalls."""
import argparse
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parallel_download(args):
    target = Path(args.local)
    parts = target.parent / (target.name + '.parts')
    parts.mkdir(exist_ok=True)
    size = args.size
    chunk_size = 2 * 1024 * 1024
    count = (size + chunk_size - 1) // chunk_size
    handshakes = threading.Semaphore(5)

    def fetch(index):
        expected = min(chunk_size, size-index*chunk_size)
        dest = parts / f'{index:04d}'
        if dest.exists() and dest.stat().st_size == expected:
            return
        for attempt in range(3):
            retained = dest.stat().st_size if dest.exists() else 0
            if retained > expected:
                raise ValueError(dest)
            offset = index*chunk_size + retained
            remaining = expected-retained
            command = ['ssh', '-T', '-oBatchMode=yes', '-oConnectTimeout=20',
                       '-oIPQoS=none', args.host,
                       f'dd if={args.remote} bs=65536 iflag=skip_bytes,count_bytes skip={offset} count={remaining} status=none']
            with dest.open('ab') as output:
                with handshakes:
                    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
                    first = proc.stdout.read(1)
                    output.write(first)
                while chunk := proc.stdout.read(65536):
                    output.write(chunk)
                status = proc.wait()
            if status == 0 and dest.stat().st_size == expected:
                print(args.host, f'part {index+1}/{count}', flush=True)
                return
            time.sleep(1)
        raise RuntimeError(f'Failed part {index}')

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(fetch, range(count)))
    with target.open('xb') as output:
        for i in range(count):
            with (parts / f'{i:04d}').open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    print('DONE', args.host, target.stat().st_size, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('host')
    p.add_argument('remote')
    p.add_argument('local')
    p.add_argument('--parallel', type=int, default=1)
    p.add_argument('--size', type=int, default=0)
    a = p.parse_args()
    if not a.remote.startswith('/tmp/dic_review_'):
        raise ValueError('Only task-created temporary archives may be transferred')
    if a.parallel > 1:
        if not a.size:
            raise ValueError('--size is required for parallel downloads')
        parallel_download(a)
        raise SystemExit(0)
    command = ['ssh', '-T', '-oBatchMode=yes', '-oConnectTimeout=15', a.host, 'cat', a.remote]
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    count = 0
    with Path(a.local).open('xb') as dest:
        while chunk := proc.stdout.read(1024 * 1024):
            dest.write(chunk)
            count += len(chunk)
            if count % (20 * 1024 * 1024) == 0:
                print(a.host, count, 'bytes received', flush=True)
    if proc.wait():
        raise RuntimeError(f'SSH transfer failed: {proc.returncode}')
    print('DONE', a.host, count, flush=True)
