"""Download the frozen pilot model manifest using an existing Python environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'model_manifest.json').read_text())
    os.environ.setdefault('MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS', '4')
    os.environ.setdefault('TQDM_DISABLE', '1')
    from modelscope import snapshot_download
    from modelscope_hub.constants import DOWNLOAD_PARALLELS

    status = dict(pid=os.getpid(), phase='download', model=manifest['model'],
                  total_bytes=manifest['bytes'], file_workers=4,
                  range_workers=DOWNLOAD_PARALLELS)

    def save():
        (root / 'download_status.json').write_text(json.dumps(status, indent=2))

    save()
    try:
        print(json.dumps(status), flush=True)
        snapshot_download(manifest['model'], local_dir=str(root / 'model'),
                          allow_patterns=[f['path'] for f in manifest['files']],
                          max_workers=4)
        status['phase'] = 'verify'
        save()
        for entry in manifest['files']:
            path = root / 'model' / entry['path']
            if path.stat().st_size != entry['size']:
                raise ValueError(f'Size mismatch: {path}')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if entry['sha256'] and digest.hexdigest() != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
        status['phase'] = 'ready'
        save()
        print('READY: all manifest sizes and SHA256 hashes verified', flush=True)
    except Exception as error:
        status.update(phase='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
