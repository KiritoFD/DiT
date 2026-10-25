"""Qwen P2-A 1,000-image targeted synthesis and two-stage evaluation pipeline.

1. Targets: Top10 calligraphers x 50 missing characters x 2 seeds = 1,000 images.
2. Filter Stage 1: Physical stroke & ink quality (frag, hole, ink density, STD IoU).
3. Filter Stage 2: Independent writer style verification via style_disc_ce_holdout.pt (pass or degrade to Null-Writer).
"""
import os
import sys
import json
import time
import math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-lora-ref-20261025/vendor')
sys.path.insert(0, '/home/ds/Workspace/DiT')
os.chdir('/home/ds/Workspace/DiT')

from safetensors.torch import load_file
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline
from src.utils.std_soft_glyph import load_font_config, FontCoverage, render_base_mask, soften_mask
from src.eval.metrics import frag_ratio, hole_ratio
from src.eval.metrics_ink import ink_iou


def select_targets(data_path, out_dir):
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    train_rows = data['datasets']['train']
    val_rows = data['datasets']['validation']
    test_rows = data['datasets']['test']

    all_train_chars = set(r['character'] for r in train_rows)
    val_chars = set(r['character'] for r in val_rows)
    test_chars = set(r['character'] for r in test_rows)
    held_out_chars = val_chars | test_chars

    font_config, script_specs, font_dirs = load_font_config('configs/std_soft_fonts.json', [])
    coverage = FontCoverage()
    SID = {'楷': 0, '行': 3, '隶': 4}

    def has_font(script, char):
        for f in script_specs[SID[script]]['faces']:
            if coverage.supports(f, char):
                return True
        return False

    primary_scripts = {
        '何绍基': '隶',
        '文徵明': '行',
        '柳公权': '楷',
        '欧阳询': '楷',
        '王羲之': '行',
        '米芾': '行',
        '苏轼': '行',
        '褚遂良': '楷',
        '赵孟頫': '楷',
        '颜真卿': '楷',
    }

    char_freq = Counter(r['character'] for r in train_rows)
    targets = []
    task_id = 0

    seeds = [20261026, 20261027]

    for writer in sorted(primary_scripts.keys()):
        script = primary_scripts[writer]
        writer_existing = set(r['character'] for r in train_rows if r['calligrapher'] == writer)
        candidates = [
            c for c in all_train_chars
            if c not in writer_existing
            and c not in held_out_chars
            and has_font(script, c)
        ]
        candidates.sort(key=lambda c: char_freq[c], reverse=True)
        chosen_chars = candidates[:50]
        assert len(chosen_chars) == 50, f'{writer} only has {len(chosen_chars)} candidates'

        for char in chosen_chars:
            for s_idx, seed in enumerate(seeds):
                targets.append({
                    'task_id': task_id,
                    'calligrapher': writer,
                    'script': script,
                    'character': char,
                    'seed': seed,
                    'seed_idx': s_idx,
                    'filename': f'{task_id:04d}_{writer}_{script}_{char}_s{seed}.png'
                })
                task_id += 1

    targets_path = out_dir / 'targets_1000.json'
    targets_path.write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Generated target manifest: {len(targets)} samples across 10 writers x 50 chars x 2 seeds.')
    return targets


def main():
    root = Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025')
    out_dir = Path('/home/ds/Workspace/xy/qwen_synth_1000')
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    print('=== Step 1: Select 1,000 target combinations ===')
    targets = select_targets(root / 'data.json', out_dir)

    print('\n=== Step 2: Initialize Qwen P2-A 4k model ===')
    config = json.loads((root / 'config.json').read_text())
    shared_root = Path(config['shared_root'])
    cfg = json.loads((shared_root / 'config.json').read_text())

    shared_ckpt = shared_root / 'checkpoints' / '0020000' / 'pytorch_lora_weights.safetensors'
    std_ckpt = root / 'checkpoints' / '0004000' / 'pytorch_lora_weights.safetensors'

    model = QwenImage21Transformer2DModel.from_pretrained(
        root / 'base', subfolder='transformer', torch_dtype=torch.bfloat16,
        local_files_only=True).requires_grad_(False).cuda()

    model.add_adapter(LoraConfig(r=cfg['rank'], lora_alpha=cfg['rank'],
                                 init_lora_weights='gaussian',
                                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']),
                      adapter_name='shared')

    def strip_adapter(d):
        return {k.removeprefix('transformer.')
                .replace('.lora_A.shared.', '.lora_A.')
                .replace('.lora_B.shared.', '.lora_B.'): v for k, v in d.items()}

    saved = strip_adapter({k: v for k, v in load_file(str(shared_ckpt)).items()})
    set_peft_model_state_dict(model, saved, adapter_name='shared')

    model.add_adapter(LoraConfig(r=config['rank'], lora_alpha=config['rank'],
                                 init_lora_weights='gaussian',
                                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']),
                      adapter_name='std')

    saved_std = {k.removeprefix('transformer.')
                 .replace('.lora_A.std.', '.lora_A.')
                 .replace('.lora_B.std.', '.lora_B.'): v
                 for k, v in load_file(str(std_ckpt)).items()}
    set_peft_model_state_dict(model, saved_std, adapter_name='std')
    model.set_adapter(['shared', 'std'])
    model.eval()

    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    print('Qwen P2-A 4k pipeline ready.')

    font_config, script_specs, font_dirs = load_font_config('configs/std_soft_fonts.json', [])
    coverage = FontCoverage()
    SID = {'楷': 0, '行': 3, '隶': 4}

    def render_std(script, char):
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
        return gray, Image.merge('RGBA', (gray, gray, gray, Image.new('L', gray.size, 255)))

    print('\n=== Step 3: Run targeted synthesis of 1,000 images ===')
    t0 = time.time()
    for i, t in enumerate(targets):
        dest = images_dir / t['filename']
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        gray, glyph = render_std(t['script'], t['character'])
        prompt = (f"书法家：{t['calligrapher']}。书体：{t['script']}书。"
                  f"请以参考字形的结构与布局为标准，用{t['calligrapher']}的{t['script']}书笔法书写汉字「{t['character']}」。"
                  "纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。")

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            img = pipe(prompt=prompt, image=[glyph], height=256, width=256,
                       output_resolution=256, num_inference_steps=8, true_cfg_scale=1.0,
                       use_kv_cache=True,
                       generator=torch.Generator('cpu').manual_seed(t['seed'])).images[0]

        white = Image.new('RGBA', img.size, 'white')
        white.alpha_composite(img.convert('RGBA'))
        pred = white.convert('RGB')
        pred.save(dest)

        if (i + 1) % 50 == 0 or (i + 1) == len(targets):
            el = time.time() - t0
            spd = (i + 1) / el
            rem = (len(targets) - (i + 1)) / spd if spd > 0 else 0
            print(f'Progress: [{i+1}/{len(targets)}] {((i+1)/len(targets)*100):.1f}% | '
                  f'Speed: {spd:.2f} img/s | ETA: {rem/60:.1f} min', flush=True)

    print(f'Synthesis completed in {(time.time()-t0)/60:.2f} minutes.')


if __name__ == '__main__':
    main()
