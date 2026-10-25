"""Precompute prompt+glyph embeddings for every unique train/val key (single-prompt encode)."""
import json
import sys
import time
from pathlib import Path

root = Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025')
sys.path.insert(0, str(root))
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, str(root / 'vendor'))
sys.path.insert(0, '/home/ds/Workspace/DiT')
import os
os.chdir('/home/ds/Workspace/DiT')

import torch  # noqa: E402
from diffusers import QwenImage21Pipeline  # noqa: E402
from src.utils.std_soft_glyph import load_font_config, FontCoverage, render_base_mask, soften_mask  # noqa: E402

OUT = root / 'text_std.pt'
data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
SID = {'楷': 0, '行': 3, '隶': 4}
font_config, script_specs, font_dirs = load_font_config('configs/std_soft_fonts.json', [])
coverage = FontCoverage()


def std_image(script, char):
    face = None
    for f in script_specs[SID[script]]['faces']:
        if coverage.supports(f, char):
            face = f
            break
    if face is None:
        raise KeyError(f'no font for {script} {char!r}')
    mask = render_base_mask(char, face, render_size=512)
    soft = soften_mask(mask, 0.65).resize((256, 256), Image.Resampling.BILINEAR)
    gray = Image.eval(soft, lambda v: 255 - v)
    return Image.merge('RGBA', (gray, gray, gray, Image.new('L', gray.size, 255)))


def prompt_for(row):
    char, script = row['character'], row['script']
    return (f'书法家：{row["calligrapher"]}。书体：{script}书。'
            f'请以参考字形的结构与布局为标准，用{row["calligrapher"]}的{script}书笔法书写汉字「{char}」。'
            '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')


def main():
    keys = {}
    for split in ('train', 'validation'):
        for row in data['datasets'][split]:
            k = (row['character'], row['calligrapher'], row['script'])
            if k not in keys:
                keys[k] = row
    print('UNIQUE', len(keys), flush=True)
    if OUT.exists():
        print('CACHE_EXISTS', flush=True)
        return
    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=None,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    result = {}
    started = time.monotonic()
    for i, (k, row) in enumerate(keys.items()):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            emb, mask, pad = pipe.encode_prompt(prompt=[prompt_for(row)],
                                                image=[std_image(row['script'], row['character'])],
                                                device='cuda')
        result[k] = (emb[0].cpu(), pad[0].cpu() if pad is not None else None)
        if (i + 1) % 500 == 0:
            print(f'{i + 1}/{len(keys)} {time.monotonic() - started:.0f}s', flush=True)
    torch.save(result, OUT)
    print('TEXT_STD_READY', len(result), f'{time.monotonic() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
