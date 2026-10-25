"""Reuse the existing pytorch environment; download the official checkpoint only."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path('/home/ds/Workspace/xy/qwen-calli-20261022')
BASE='/home/ds/miniconda3/envs/pytorch/bin/python'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    state=dict(phase='download',python=BASE,pid=os.getpid(),next_steps='Load task-local Diffusers source with existing pytorch environment, run fixed 32-image manifest')
    def save():
        (ROOT/'bootstrap_status.json').write_text(json.dumps(state,indent=2))
    save()
    os.environ['HF_ENDPOINT']='https://hf-mirror.com'
    os.environ['TQDM_DISABLE']='1'
    from huggingface_hub import HfApi
    from modelscope import snapshot_download
    info=HfApi().model_info('Qwen/Qwen-Image-2.1-Turbo',files_metadata=True)
    files=[f for f in info.siblings if f.rfilename.endswith(('.safetensors','.json','.txt','.model','.tiktoken','.jinja'))
           or f.rfilename in ('LICENSE','README.md')]
    url='https://modelscope.cn/api/v1/models/Qwen/Qwen-Image-2.1-Turbo/repo/files?Revision=master&Recursive=true'
    with urllib.request.urlopen(url,timeout=30) as response:ms=json.load(response)['Data']['Files']
    ms={f['Path']:f for f in ms if f['Type']=='blob'}
    for file in files:
        if file.rfilename.endswith('.safetensors'):
            expected=file.lfs.sha256
            if ms[file.rfilename]['Sha256']!=expected:raise ValueError('Official weight sources disagree: '+file.rfilename)
    manifest=dict(model=info.id,hf_revision=info.sha,source='official ModelScope, weight hashes matched to HF',
        files=[dict(path=f.rfilename,size=f.size,sha256=ms[f.rfilename]['Sha256'],revision=ms[f.rfilename]['Revision']) for f in files],
        bytes=sum(f.size or 0 for f in files))
    (ROOT/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DOWNLOAD_MS',manifest['bytes'],flush=True)
    snapshot_download(info.id,local_dir=str(ROOT/'model'),allow_patterns=[f.rfilename for f in files],max_workers=8)
    state['phase']='verify';save()
    for file in manifest['files']:
        path=ROOT/'model'/file['path'];digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if file['sha256'] and digest.hexdigest()!=file['sha256']:raise ValueError('Checksum failed: '+file['path'])
    sigmas=json.loads((ROOT/'model/model_index.json').read_text())['sample_sigmas']
    if len(sigmas)!=8:raise ValueError('Not the released 8-step Turbo schedule')
    state['phase']='ready';save()


if __name__=='__main__':
    try:main()
    except Exception as error:
        path=ROOT/'bootstrap_status.json';state=json.loads(path.read_text())
        state.update(phase='failed',error=repr(error));path.write_text(json.dumps(state,indent=2))
        raise
