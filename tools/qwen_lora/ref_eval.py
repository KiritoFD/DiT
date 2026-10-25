"""Test-200 rerun with 2 same-writer different-character reference images; paired vs no-ref baseline."""
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

root = Path('/home/ds/Workspace/xy/qwen-lora-top10-20261024')
sys.path.insert(0, str(root))
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, str(root / 'vendor'))

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline
from src.eval.metrics import ssim, get_lpips
from src.eval.metrics_ink import ink_iou

CKPT = root / 'checkpoints' / '0020000'
OUT = root / 'ref_eval_20k'
BASELINE = root / 'test_eval_20k' / 'metrics.json'
FIELDS = ('ssim', 'lpips', 'ink_iou')


def pick_refs(train, writer, script, target):
    cands = [r for r in train if r['calligrapher'] == writer and r['script'] == script
             and r['character'] != target]
    if len(cands) < 2:
        raise ValueError(f'Not enough refs for {writer}-{script}')
    counts = Counter(r['character'] for r in cands)
    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:2]
    return [next(r for r in cands if r['character'] == c) for c, _ in top]


def main():
    (OUT / 'images').mkdir(parents=True, exist_ok=False)
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    cfg = json.loads((root / 'config.json').read_text())
    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    test_rows = data['datasets']['test']
    train = data['datasets']['train']

    model = QwenImage21Transformer2DModel.from_pretrained(root / 'base', subfolder='transformer',
                                                          torch_dtype=torch.bfloat16,
                                                          local_files_only=True).requires_grad_(False).cuda()
    model.add_adapter(LoraConfig(r=cfg['rank'], lora_alpha=cfg['rank'], init_lora_weights='gaussian',
                                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    saved = {k.removeprefix('transformer.'): v
             for k, v in load_file(str(CKPT / 'pytorch_lora_weights.safetensors')).items()}
    set_peft_model_state_dict(model, saved, adapter_name='default')
    current = get_peft_model_state_dict(model)
    if set(current) != set(saved) or any(not torch.equal(current[k].detach().cpu(), v) for k, v in saved.items()):
        raise RuntimeError('LoRA checkpoint load was not exact')
    model.eval()
    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    lp, error = get_lpips('cuda')
    if lp is None:
        raise RuntimeError(error)

    results = []
    started = time.monotonic()
    for i, row in enumerate(test_rows):
        refs = pick_refs(train, row['calligrapher'], row['script'], row['character'])
        prompt = (f'书法家：{row["calligrapher"]}。书体：{row["script"]}书。'
                  f'请学习全部2张参考真迹的笔画与结体风格，'
                  f'并用{row["calligrapher"]}的{row["script"]}书风格书写汉字「{row["character"]}」。'
                  '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
        images = [Image.open(root / r['file']).convert('RGB') for r in refs]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            img = pipe(prompt=prompt, image=images, height=256, width=256, output_resolution=256,
                       num_inference_steps=8, true_cfg_scale=1., use_kv_cache=True,
                       generator=torch.Generator('cpu').manual_seed(90)).images[0]
        if len(pipe.scheduler.timesteps) != 8:
            raise ValueError('Unexpected sampling steps')
        white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
        pred = white.convert('RGB')
        name = f'{i:03d}_{row["calligrapher"]}_{row["script"]}_{row["character"]}.png'
        pred.save(OUT / 'images' / name)
        gt = Image.open(root / row['file']).convert('RGB')
        p = np.asarray(pred, dtype=np.float32) / 255
        g = np.asarray(gt, dtype=np.float32) / 255
        ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
        with torch.inference_mode():
            d = float(lp(ten(p), ten(g)).item())
        results.append(dict(character=row['character'], calligrapher=row['calligrapher'],
                            script=row['script'], image=name, gt=row['file'],
                            refs=[r['character'] for r in refs],
                            ssim=round(ssim(p, g), 5), lpips=round(d, 5),
                            ink_iou=round(ink_iou(p, g), 5)))
        if (i + 1) % 25 == 0:
            print(f'{i + 1}/200 {time.monotonic() - started:.0f}s', flush=True)
    pipe.vae.to('cpu'); del pipe
    (OUT / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    if len(results) != len(baseline):
        raise ValueError('Baseline/ref length mismatch')
    deltas = {f: [r[f] - b[f] for r, b in zip(results, baseline)] for f in FIELDS}
    summary = dict(
        checkpoint=str(CKPT), protocol='256 / 8 steps / seed90 / 2 same-writer different-char train refs',
        refs='2 most frequent other characters of the same writer+script in TRAIN split',
        n=len(results),
        with_refs={f: round(st.mean(r[f] for r in results), 5) for f in FIELDS},
        no_refs={f: round(st.mean(b[f] for b in baseline), 5) for f in FIELDS},
        mean_delta={f: round(st.mean(v), 5) for f, v in deltas.items()},
        improved={f: sum(1 for v in deltas[f] if (v > 0 if f != 'lpips' else v < 0)) for f in FIELDS},
        by_script={s: dict(with_refs={f: round(st.mean(r[f] for r in results if r['script'] == s), 5) for f in FIELDS},
                           no_refs={f: round(st.mean(b[f] for b, r in zip(baseline, results) if r['script'] == s), 5) for f in FIELDS})
                   for s in sorted({r['script'] for r in results})},
        by_writer={w: dict(delta_ssim=round(st.mean(r['ssim'] - b['ssim'] for r, b in zip(results, baseline)
                                                  if r['calligrapher'] == w), 5),
                           delta_lpips=round(st.mean(r['lpips'] - b['lpips'] for r, b in zip(results, baseline)
                                                     if r['calligrapher'] == w), 5))
                   for w in sorted({r['calligrapher'] for r in results})})
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps({k: summary[k] for k in ('with_refs', 'no_refs', 'mean_delta', 'improved')},
                                ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
