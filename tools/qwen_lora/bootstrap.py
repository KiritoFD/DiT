"""Download non-distilled Qwen 2.1; encode ready components during download."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(root, **state):
    (root/'bootstrap_status.json').write_text(json.dumps(dict(pid=os.getpid(), **state), indent=2))


def main(root,download_only=False):
    import requests
    os.environ['MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS'] = '4'
    os.environ['TQDM_DISABLE'] = '1'
    from modelscope import snapshot_download
    model = 'Qwen/Qwen-Image-2.1'
    url = f'https://modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=true'
    response = requests.get(url, timeout=45); response.raise_for_status()
    files = [r for r in response.json()['Data']['Files'] if r['Type']=='blob'
             and (r['Path'].endswith(('.json','.safetensors','.txt','.model','.tiktoken','.jinja'))
                  or r['Path'] in ('LICENSE','README.md'))]
    (root/'base_manifest.json').write_text(json.dumps(dict(model=model, files=files), indent=2))
    save(root, phase='download', model=model, bytes=sum(f['Size'] for f in files))

    def group(prefix):
        selected = [f for f in files if f['Path'].startswith(prefix)]
        if not selected:
            raise ValueError(f'Empty component: {prefix}')
        snapshot_download(model, local_dir=str(root/'base'),
                          allow_patterns=[f['Path'] for f in selected], max_workers=4)
        for f in selected:
            path = root/'base'/f['Path']; digest = hashlib.sha256()
            if path.stat().st_size != f['Size']:
                raise ValueError(f'Size: {path}')
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(chunk)
            if digest.hexdigest()!=f['Sha256']:
                raise ValueError(f'SHA256: {path}')
        (root/(prefix.replace('/','_')+'ready')).touch()
        print('COMPONENT_READY', prefix, flush=True)

    # Large components download concurrently; cache workers wait for verified components.
    small = [f for f in files if not f['Path'].endswith('.safetensors')]
    snapshot_download(model, local_dir=str(root/'base'), allow_patterns=[f['Path'] for f in small], max_workers=4)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(group, prefix) for prefix in ('vae/', 'text_encoder/', 'transformer/')]
        while not download_only and not (root/'config.json').exists():
            time.sleep(5)
        workers = []
        for phase in (() if download_only else ('latents', 'text')):
            log = (root/f'cache_{phase}.log').open('a')
            workers.append(subprocess.Popen([sys.executable, str(Path(__file__).with_name('cache.py')),
                                            phase, '--root', str(root)], stdout=log, stderr=subprocess.STDOUT))
            log.close()
        for future in futures:
            future.result()
        save(root, phase='downloaded' if download_only else 'encoding')
        for worker in workers:
            if worker.wait()!=0:
                raise RuntimeError('Cache worker failed; inspect cache logs')
    if download_only:
        return
    save(root, phase='training_preflight')
    subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')), '--root', str(root)], check=True)
    save(root, phase='completed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--download-only',action='store_true');a=p.parse_args()
    try:
        main(a.root.resolve(),a.download_only)
    except Exception as error:
        save(a.root, phase='failed', error=repr(error))
        raise
