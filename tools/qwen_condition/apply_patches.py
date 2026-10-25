"""Surgical patches for Qwen-glyph dual-channel conditioning (arms A/B/C) on the 48 U-Net stack.

Every patch is an exact-string replacement; a failed anchor aborts before writing anything.
Backups: <file>.bak_qwencond ; unified diffs saved to tools/qwen_condition/patches_20261025.diff
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/ds/Workspace/DiT')

NEW_FILES = {
    'src/utils/qwen_glyph_store.py': '''"""Qwen-generated glyph condition store: 64x64 white-ink-on-black, sigma-softened at load."""  # noqa: E501
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class QwenGlyphStore:
    """Keyed by (calligrapher_id, script_id, character); matches std_soft normalization."""

    def __init__(self, root, sigma=0.65, size=64):
        from .std_soft_glyph import soften_mask
        self.size = int(size)
        self.sigma = float(sigma)
        index = json.loads((Path(root) / 'index.json').read_text(encoding='utf-8'))
        stack = torch.zeros(len(index), self.size, self.size, dtype=torch.uint8)
        self._keys = {}
        for i, (name, rec) in enumerate(sorted(index.items())):
            img = Image.open(Path(root) / 'glyphs' / name).convert('L')
            if img.size != (self.size, self.size):
                img = img.resize((self.size, self.size), Image.Resampling.BOX)
            stack[i] = torch.from_numpy(np.asarray(img, dtype=np.uint8))
            self._keys[(int(rec['callig_id']), int(rec['script_id']),
                        str(rec['character']))] = i
        soft = torch.zeros(len(stack), 1, self.size, self.size, dtype=torch.float32)
        for i in range(len(stack)):
            s = soften_mask(Image.fromarray(stack[i].numpy(), mode='L'), self.sigma)
            soft[i, 0] = torch.from_numpy(np.asarray(s, dtype=np.float32).copy()) / 255.0
        self._stack = soft

    def __len__(self):
        return len(self._keys)

    def get(self, callig_id, script_id, char):
        key = (int(callig_id), int(script_id), str(char))
        if key not in self._keys:
            raise KeyError(f'qwen glyph missing: {key}')
        return self._stack[self._keys[key]].clone()
''',
}

PATCHES = {
    'src/unet/model.py': [
        ("std_gate_scales=(),writer_film_rank=0,",
         "std_gate_scales=(),writer_film_rank=0,std_channels=1,"),
        ("self.std_stem=nn.Sequential(nn.Conv2d(1,64,3,stride=2,padding=1),act_layer(),nn.Conv2d(64,64,3,padding=1))",
         "self.std_stem=nn.Sequential(nn.Conv2d(std_channels,64,3,stride=2,padding=1),act_layer(),nn.Conv2d(64,64,3,padding=1))"),
        ("if g is None:g=x.new_zeros((len(x),1,64,64))",
         "if g is None:g=x.new_zeros((len(x),self.std_stem[0].in_channels,64,64))"),
        ("if g.shape!=(len(x),1,64,64):raise ValueError('Expected STD [B,1,64,64]')",
         "if g.shape!=(len(x),self.std_stem[0].in_channels,64,64):raise ValueError('Expected STD [B,C,64,64] matching std_stem')"),
    ],
    'src/unet/io.py': [
        ("    std_shape_aug:str='strong'\n",
         "    std_shape_aug:str='strong'\n    std_channels:int=1\n    qwen_mode:str='off'\n    qwen_dropout:float=0.\n    qwen_soft_root:str=''\n"),
        ("act_fn=getattr(config,'act_fn','silu'))",
         "act_fn=getattr(config,'act_fn','silu'),std_channels=getattr(config,'std_channels',1))"),
        ("        eval_vae_path=config.vae_path,",
         "        eval_vae_path=config.vae_path,\n        qwen_soft_root=getattr(config,'qwen_soft_root',''),\n        qwen_mode=getattr(config,'qwen_mode','off'),"),
    ],
    'src/unet/data.py': [
        ("        self.mode=config.std_shape_aug",
         "        self.mode=config.std_shape_aug\n        self.qwen_mode=getattr(config,'qwen_mode','off')\n        self.qwen_dropout=float(getattr(config,'qwen_dropout',0.))"),
        ("        kwargs['g']=self(batch,x0.device,step,clean=kwargs['g'])",
         "        kwargs['g']=self(batch,x0.device,step,clean=kwargs['g'])\n"
         "        if self.qwen_mode!='off' and 'qwen_soft' in batch:\n"
         "            q=batch['qwen_soft'].to(x0.device,non_blocking=True).float()\n"
         "            if q.ndim==3:q=q[:,None]\n"
         "            keep=torch.rand(len(q),device=q.device)>=self.qwen_dropout\n"
         "            q=torch.where(keep[:,None,None,None],q,torch.zeros_like(q))\n"
         "            kwargs['g']=torch.cat([kwargs['g'],q],1) if self.qwen_mode=='dual' else q"),
    ],
    'src/train/data_setup.py': [
        ("                                    std_soft_root=(getattr(args, 'std_soft_root', '')\n                                                   if std_soft_cond else None),",
         "                                    std_soft_root=(getattr(args, 'std_soft_root', '')\n                                                   if std_soft_cond else None),\n"
         "                                    qwen_soft_root=(getattr(args, 'qwen_soft_root', '') or None),"),
    ],
    'src/utils/latent_dataset.py': [
        ("std_soft_eval_sigma=0.65,\n                 std_soft_preload=True, strict_coverage=True):",
         "std_soft_eval_sigma=0.65, qwen_soft_root=None,\n                 std_soft_preload=True, strict_coverage=True):"),
        ("        self.latent_shards_dir = latent_shards_dir",
         "        self._qwen_soft = None\n"
         "        if qwen_soft_root:\n"
         "            from .qwen_glyph_store import QwenGlyphStore\n"
         "            self._qwen_soft = QwenGlyphStore(qwen_soft_root, sigma=0.65)\n"
         "            print(f\"[qwen-soft] {len(self._qwen_soft)} Qwen glyphs from {qwen_soft_root}\", flush=True)\n"
         "        self.latent_shards_dir = latent_shards_dir"),
        ("        else:\n            std_soft_t = torch.zeros(0)",
         "        else:\n            std_soft_t = torch.zeros(0)\n"
         "        if self._qwen_soft is not None:\n"
         "            qwen_soft_t = self._qwen_soft.get(int(row['calligrapher_id']),\n"
         "                                              int(row['script_id']), row.get('character', ''))\n"
         "        else:\n            qwen_soft_t = torch.zeros(0)"),
        ("            'std_soft': std_soft_t,",
         "            'std_soft': std_soft_t,\n            'qwen_soft': qwen_soft_t,"),
    ],
    'src/eval/dataset.py': [
        ("callig_script_map=None, std_soft_root=None,\n                    std_soft_sigma=0.65):",
         "callig_script_map=None, std_soft_root=None,\n                    std_soft_sigma=0.65, qwen_soft_root=None):"),
        ("    skels = torch.zeros(n, 1, image_size, image_size, dtype=torch.float32)",
         "    qwen_store = None\n"
         "    if qwen_soft_root:\n"
         "        from src.utils.qwen_glyph_store import QwenGlyphStore\n"
         "        qwen_store = QwenGlyphStore(qwen_soft_root, sigma=std_soft_sigma)\n"
         "    qwen_latent = None\n"
         "    if qwen_store is not None:\n"
         "        qwen_latent = torch.zeros(n, 1, qwen_store.size, qwen_store.size, dtype=torch.float32)\n"
         "    skels = torch.zeros(n, 1, image_size, image_size, dtype=torch.float32)"),
        ("        if std_soft_store is not None:\n            skels_latent[i] = std_soft_store.get(\n                int(row[\"script_id\"]), row.get(\"character\", \"\"))",
         "        if std_soft_store is not None:\n            skels_latent[i] = std_soft_store.get(\n                int(row[\"script_id\"]), row.get(\"character\", \"\"))\n"
         "        if qwen_store is not None:\n"
         "            qwen_latent[i] = qwen_store.get(\n"
         "                int(row[\"calligrapher_id\"]), int(row[\"script_id\"]),\n"
         "                row.get(\"character\", \"\"))"),
        ('            "skels_latent": skels_latent, "missing_skel": missing_skel,',
         '            "skels_latent": skels_latent, "qwen_latent": qwen_latent, "missing_skel": missing_skel,'),
    ],
    'src/eval/in_mem_eval.py': [
        ("          bool(getattr(args, \"use_script_cond\", False)), _sf_ck, _sf_shift_ck)",
         "          bool(getattr(args, \"use_script_cond\", False)), _sf_ck, _sf_shift_ck,\n"
         "          getattr(args, 'qwen_soft_root', '') or '', getattr(args, 'qwen_mode', 'off'))"),
        ("        _CACHES[ck] = cache",
         "        _CACHES[ck] = cache\n"
         "        _mode = getattr(args, 'qwen_mode', 'off')\n"
         "        if _mode != 'off' and cache.get('qwen_latent') is not None:\n"
         "            _q = cache['qwen_latent']; _s = cache['skels_latent']\n"
         "            cache['skels_latent'] = torch.cat([_s, _q], 1) if _mode == 'dual' else _q"),
        ("        validate_std_soft_condition(cache.get(\"skels_latent\"), cache[\"n\"])",
         "        if getattr(args, 'qwen_mode', 'off') == 'off':\n"
         "            validate_std_soft_condition(cache.get(\"skels_latent\"), cache[\"n\"])"),
        ("    if skels_latent.ndim == 4 and skels_latent.shape[1] == 1:",
         "    if skels_latent.ndim == 4 and skels_latent.shape[1] in (1, 2):"),
        ("            array = ((1.0 - skels_latent[i, 0].float().clamp(0, 1)) * 255.0)",
         "            array = ((1.0 - skels_latent[i, skels_latent.shape[1] - 1].float().clamp(0, 1)) * 255.0)"),
    ],
    'src/unet/variants/transfer.py': [
        ("    for key,value in source.items():\n        if value.shape!=target[key].shape:raise ValueError(f'Checkpoint shape mismatch: {key}')",
         "    if 'std_stem.0.weight' in source and source['std_stem.0.weight'].dim()==4 \\\n"
         "            and source['std_stem.0.weight'].shape[1]==1 and target['std_stem.0.weight'].shape[1]==2:\n"
         "        w=source['std_stem.0.weight']\n"
         "        pad=torch.zeros(w.shape[0],2,*w.shape[2:],dtype=w.dtype)\n"
         "        pad[:,0:1]=w\n"
         "        source['std_stem.0.weight']=pad\n"
         "    for key,value in source.items():\n        if value.shape!=target[key].shape:raise ValueError(f'Checkpoint shape mismatch: {key}')"),
    ],
}


def main():
    report = {}
    diffs = []
    for rel, text in NEW_FILES.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        report[rel] = 'created'
    for rel, pairs in PATCHES.items():
        path = ROOT / rel
        original = path.read_text(encoding='utf-8')
        if (ROOT / (rel + '.bak_qwencond')).exists():
            if pairs[0][1].split('\n')[0] in original:
                report[rel] = 'already patched, skipped'
                continue
            raise SystemExit(f'backup exists but content unexpected: {rel}')
        patched = original
        for old, new in pairs:
            if patched.count(old) != 1:
                raise SystemExit(f'anchor not unique/found in {rel}: {old[:60]!r} count={patched.count(old)}')
            patched = patched.replace(old, new)
        shutil.copy2(path, ROOT / (rel + '.bak_qwencond'))
        path.write_text(patched, encoding='utf-8')
        diffs.append(subprocess.run(['diff', '-u', rel + '.bak_qwencond', rel],
                                    cwd=ROOT, capture_output=True, text=True).stdout)
        report[rel] = f'{len(pairs)} patches applied'
    (ROOT / 'tools/qwen_condition/patches_20261025.diff').write_text(
        '\n'.join(diffs), encoding='utf-8')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
