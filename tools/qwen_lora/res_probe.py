"""Base-model 256-vs-512 generation probe against GT, using cached text embeddings."""
import json
import statistics as st
import sys
from pathlib import Path

root = Path('/home/ds/Workspace/xy/qwen-lora-top10-20261024')
sys.path.insert(0, str(root))
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, str(root / 'vendor'))

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import QwenImage21Pipeline
from src.eval.metrics import ssim, get_lpips
from src.eval.metrics_ink import ink_iou

data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
text = torch.load(root / 'text.pt', map_location='cpu', weights_only=True)
seen = set(); rows = []
for r in data['datasets']['validation']:
    k = (r['writer_index'], r['script_index'])
    if k in seen:
        continue
    seen.add(k); rows.append(r)
    if len(rows) >= 9:
        break

pipe = QwenImage21Pipeline.from_pretrained(root / 'base', text_encoder=None,
                                           dtype=torch.bfloat16, local_files_only=True).to('cuda')
pipe.set_progress_bar_config(disable=True)
lp, error = get_lpips('cuda')
if lp is None:
    raise RuntimeError(error)

out = root / 'res_probe'; out.mkdir(exist_ok=False)
results = []
for i, row in enumerate(rows):
    emb = text[(row['character'], row['calligrapher'], row['script'])][None].cuda()
    mask = torch.ones(emb.shape[:2], device='cuda', dtype=torch.bool)
    per = []
    for res in (256, 512):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            img = pipe(prompt_embeds=emb, prompt_embeds_mask=mask, height=res, width=res,
                       output_resolution=res, num_inference_steps=8, true_cfg_scale=1.,
                       use_kv_cache=True, generator=torch.Generator('cpu').manual_seed(90)).images[0]
        white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
        pred = white.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
        name = f'{i}_{row["calligrapher"]}_{row["script"]}_{row["character"]}_{res}.png'
        pred.save(out / name)
        gt = Image.open(root / row['file']).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
        p = np.asarray(pred, dtype=np.float32) / 255
        g = np.asarray(gt, dtype=np.float32) / 255
        ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
        with torch.inference_mode():
            d = float(lp(ten(p), ten(g)).item())
        per.append(dict(res=res, ssim=round(ssim(p, g), 5), lpips=round(d, 5),
                        ink=round(ink_iou(p, g), 5)))
    results.append(dict(row={k: row[k] for k in ('character', 'calligrapher', 'script')}, metrics=per))
    print(json.dumps(results[-1], ensure_ascii=False), flush=True)

summary = {f: {res: round(st.mean(m[f] for r in results for m in r['metrics'] if m['res'] == res), 5)
               for res in (256, 512)} for f in ('ssim', 'lpips', 'ink')}
(out / 'summary.json').write_text(json.dumps(dict(summary=summary, rows=results),
                                             ensure_ascii=False, indent=2), encoding='utf-8')
print('SUMMARY', json.dumps(summary))

mont = Image.new('RGB', (256 * 3 + 40, (256 + 14) * len(rows)), 'white')
draw = ImageDraw.Draw(mont)
for i, row in enumerate(rows):
    y = i * (256 + 14)
    for col, res in enumerate((256, 512)):
        img = Image.open(out / f'{i}_{row["calligrapher"]}_{row["script"]}_{row["character"]}_{res}.png')
        mont.paste(img, (col * (256 + 20), y + 14))
    gt = Image.open(root / row['file']).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
    mont.paste(gt, (2 * (256 + 20), y + 14))
    draw.text((4, y + 2), f'{row["calligrapher"]}-{row["script"]}-{row["character"]}', fill='black')
    draw.text((256 + 24, y + 2), '512->256', fill='black')
    draw.text((2 * (256 + 20) + 4, y + 2), 'GT', fill='black')
mont.save(out / 'montage.png')
print('MONTAGE_SAVED', out / 'montage.png')
