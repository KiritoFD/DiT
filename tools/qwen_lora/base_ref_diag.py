"""Base model (no LoRA) on test200 with and without references; disentangles ref value from LoRA OOD."""
import json
import statistics as st
import sys
import time
from pathlib import Path

root = Path('/home/ds/Workspace/xy/qwen-lora-top10-20261024')
sys.path.insert(0, str(root))
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, str(root / 'vendor'))
sys.path.insert(0, str(root / 'code'))

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline
from src.eval.metrics import ssim, get_lpips
from src.eval.metrics_ink import ink_iou
from ref_eval import pick_refs

OUT = root / 'base_ref_diag'
FIELDS = ('ssim', 'lpips', 'ink_iou')


def main():
    (OUT / 'images').mkdir(parents=True, exist_ok=False)
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    text = torch.load(root / 'text.pt', map_location='cpu', weights_only=True)
    test_rows = data['datasets']['test']
    train = data['datasets']['train']
    pipe = QwenImage21Pipeline.from_pretrained(root / 'base',
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    lp, error = get_lpips('cuda')
    if lp is None:
        raise RuntimeError(error)

    def run(mode):
        results = []
        started = time.monotonic()
        for i, row in enumerate(test_rows):
            if mode == 'refs':
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
            else:
                emb = text[(row['character'], row['calligrapher'], row['script'])][None].cuda()
                mask = torch.ones(emb.shape[:2], device='cuda', dtype=torch.bool)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    img = pipe(prompt_embeds=emb, prompt_embeds_mask=mask, height=256, width=256,
                               output_resolution=256, num_inference_steps=8, true_cfg_scale=1.,
                               use_kv_cache=True,
                               generator=torch.Generator('cpu').manual_seed(90)).images[0]
            white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
            pred = white.convert('RGB')
            pred.save(OUT / 'images' / f'{mode}_{i:03d}_{row["calligrapher"]}_{row["script"]}_{row["character"]}.png')
            gt = Image.open(root / row['file']).convert('RGB')
            p = np.asarray(pred, dtype=np.float32) / 255
            g = np.asarray(gt, dtype=np.float32) / 255
            ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
            with torch.inference_mode():
                d = float(lp(ten(p), ten(g)).item())
            results.append(dict(ssim=round(ssim(p, g), 5), lpips=round(d, 5),
                                ink_iou=round(ink_iou(p, g), 5)))
            if (i + 1) % 50 == 0:
                print(f'{mode} {i + 1}/200 {time.monotonic() - started:.0f}s', flush=True)
        return results

    no_refs = run('noref')
    refs = run('refs')
    pipe.vae.to('cpu'); del pipe
    lora_no = json.loads((root / 'test_eval_20k' / 'metrics.json').read_text(encoding='utf-8'))
    lora_ref = json.loads((root / 'ref_eval_20k' / 'metrics.json').read_text(encoding='utf-8'))

    def mean(rows, f):
        return round(st.mean(r[f] for r in rows), 5)

    summary = {}
    for name, rows in (('base_noref', no_refs), ('base_refs', refs),
                       ('lora_noref', lora_no), ('lora_refs', lora_ref)):
        summary[name] = {f: mean(rows, f) for f in FIELDS}
    summary['paired_delta'] = {
        'base_refs_minus_noref': {f: round(st.mean(r[f] - n[f] for r, n in zip(refs, no_refs)), 5) for f in FIELDS},
        'lora_refs_minus_noref': {f: round(st.mean(r[f] - n[f] for r, n in zip(lora_ref, lora_no)), 5) for f in FIELDS}}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'metrics_noref.json').write_text(json.dumps(no_refs), encoding='utf-8')
    (OUT / 'metrics_refs.json').write_text(json.dumps(refs), encoding='utf-8')
    print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
