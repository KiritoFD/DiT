"""Verified existing-model encoding -> conditional LoRA training, no downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main(root,source,model):
    root.mkdir(parents=True,exist_ok=True)
    status=dict(pid=os.getpid(),phase='verify',model=str(model))
    def save():
        (root/'run_status.json').write_text(json.dumps(status,indent=2))
    if (root/'run_status.json').exists():raise FileExistsError('Run already exists')
    save()
    try:
        manifest_path=model.parent/'model_manifest.json'
        if not manifest_path.exists():manifest_path=model.parent/'base_manifest.json'
        manifest=json.loads(manifest_path.read_text())
        if manifest['files'] and 'Path' in manifest['files'][0]:
            manifest=dict(model=manifest['model'],files=[dict(path=f['Path'],size=f['Size'],sha256=f['Sha256']) for f in manifest['files']])
        for row in manifest['files']:
            path=model/row['path'];digest=hashlib.sha256()
            if path.stat().st_size!=row['size']:raise ValueError(f'Size: {path}')
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
            if row['sha256'] and digest.hexdigest()!=row['sha256']:raise ValueError(f'SHA256: {path}')
        for name in ('images','data.json','writer_table.npy','vendor'):
            (root/name).symlink_to(source/name,target_is_directory=(source/name).is_dir())
        (root/'base').symlink_to(model,target_is_directory=True)
        (root/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
        for phase in ('vae','text_encoder'):(root/f'{phase}_ready').touch()
        for phase in ('latents','text'):
            status.update(phase='encoding_'+phase);save()
            with (root/f'cache_{phase}.log').open('x') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('cache.py')),phase,'--root',str(root)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        status.update(phase='training');save()
        with (root/'train.log').open('x') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('train.py')),'--root',str(root)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        status.update(phase='completed');save()
    except Exception as error:
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.source.resolve(),a.model.resolve())
