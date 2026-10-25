"""Build the Qwen glyph condition store for the U-Net A/B/C conditioning experiments.

For every unique (calligrapher, script, character) in the given CSVs, generate one glyph
with the trained Qwen Turbo LoRA (256, 8 steps, fixed seed) and store it as a 64x64
uint8 white-ink-on-black PNG -- the exact format/convention of the TTF std_soft store.
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

QROOT = Path('/home/ds/Workspace/xy/qwen-lora-top10-20261024')
sys.path.insert(0, str(QROOT))
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, str(QROOT / 'vendor'))

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline


def subset_csv(src, dst, n, seed=90):
    rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
    rng = random.Random(seed)
    picked = rng.sample(rows, n)
    with open(dst, 'w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(picked)
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--eval-csv', required=True)
    parser.add_argument('--subset', type=int, default=5000)
    parser.add_argument('--subset-out', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    train_rows = subset_csv(args.train_csv, args.subset_out, args.subset)
    eval_rows = list(csv.DictReader(open(args.eval_csv, encoding='utf-8-sig')))
    keys = []
    seen = set()
    for row in train_rows + eval_rows:
        k = (row['calligrapher_id'], row['script_id'], row['character'],
             row['calligrapher'], row['script'])
        if k[:3] not in seen:
            seen.add(k[:3])
            keys.append(k)
    print('UNIQUE_KEYS', len(keys), 'train_rows', len(train_rows), 'eval_rows', len(eval_rows), flush=True)

    out = Path(args.out)
    (out / 'glyphs').mkdir(parents=True, exist_ok=False)
    data = json.loads((QROOT / 'data.json').read_text(encoding='utf-8'))
    cfg = json.loads((QROOT / 'config.json').read_text())
    text = torch.load(QROOT / 'text.pt', map_location='cpu', weights_only=True)

    model = QwenImage21Transformer2DModel.from_pretrained(
        QROOT / 'base', subfolder='transformer', torch_dtype=torch.bfloat16,
        local_files_only=True).requires_grad_(False).cuda()
    model.add_adapter(LoraConfig(r=cfg['rank'], lora_alpha=cfg['rank'],
                                 init_lora_weights='gaussian',
                                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    saved = {k.removeprefix('transformer.'): v
             for k, v in load_file(str(QROOT / 'checkpoints/0020000/pytorch_lora_weights.safetensors')).items()}
    set_peft_model_state_dict(model, saved, adapter_name='default')
    model.eval()
    pipe = QwenImage21Pipeline.from_pretrained(QROOT / 'base', transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)

    def prompt_for(calligrapher, script, char):
        return (f'书法家：{calligrapher}。书体：{script}书。'
                f'请用{calligrapher}的{script}书风格书写汉字「{char}」。'
                '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')

    index = {}
    previews = []
    started = time.monotonic()
    for i, (callig_id, script_id, char, calligrapher, script) in enumerate(keys):
        cached = text.get((char, calligrapher, script))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if cached is not None:
                emb = cached[None].cuda()
                mask = torch.ones(emb.shape[:2], device='cuda', dtype=torch.bool)
                img = pipe(prompt_embeds=emb, prompt_embeds_mask=mask, height=256, width=256,
                           output_resolution=256, num_inference_steps=8, true_cfg_scale=1.,
                           use_kv_cache=True,
                           generator=torch.Generator('cpu').manual_seed(90)).images[0]
            else:
                img = pipe(prompt=prompt_for(calligrapher, script, char), height=256, width=256,
                           output_resolution=256, num_inference_steps=8, true_cfg_scale=1.,
                           use_kv_cache=True,
                           generator=torch.Generator('cpu').manual_seed(90)).images[0]
        white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
        gray = white.convert('RGB').convert('L')
        ink_white_on_black = Image.eval(gray, lambda v: 255 - v)  # ink -> white
        glyph = ink_white_on_black.resize((64, 64), Image.Resampling.BOX)
        name = f'g{int(callig_id):05d}_s{int(script_id)}_U{ord(char):05X}.png'
        glyph.save(out / 'glyphs' / name)
        index[name] = dict(callig_id=int(callig_id), script_id=int(script_id),
                           character=char, codepoint=ord(char))
        if len(previews) < 24:
            previews.append(white.convert('RGB').resize((160, 160)))
        if (i + 1) % 200 == 0:
            print(f'{i + 1}/{len(keys)} {time.monotonic() - started:.0f}s', flush=True)

    (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
    montage = Image.new('RGB', (160 * 8 + 20, 160 * 3 + 20), 'white')
    for i, img in enumerate(previews):
        montage.paste(img, (10 + (i % 8) * 160, 10 + (i // 8) * 160))
    montage.save(out / 'preview.jpg', quality=80)
    print('GLYPHS_READY', len(index), flush=True)


if __name__ == '__main__':
    main()
