"""Full test-200 evaluation of one LoRA checkpoint + per-writer poster, prompt-only conditioning."""
import json
import statistics as st
import sys
import time
from pathlib import Path


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


root = Path(_arg('--root', '/home/ds/Workspace/xy/qwen-lora-top10-20261024'))
CKPT = root / 'checkpoints' / _arg('--ckpt', '0020000')
OUT = root / _arg('--out', 'test_eval_20k')
GEN_STEPS = int(_arg('--steps', 8))

sys.path.insert(0, str(root))
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, str(root / 'vendor'))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline
from src.eval.metrics import ssim, get_lpips
from src.eval.metrics_ink import ink_iou

CKPT = root / 'checkpoints' / '0020000'
OUT = root / 'test_eval_20k'
FIELDS = ('ssim', 'lpips', 'ink_iou')
WRITER_EN = {'颜真卿': 'Yan Zhenqing', '王羲之': 'Wang Xizhi', '苏轼': 'Su Shi', '米芾': 'Mi Fu',
             '赵孟頫': 'Zhao Mengfu', '欧阳询': 'Ouyang Xun', '柳公权': 'Liu Gongquan',
             '褚遂良': 'Chu Suiliang', '文徵明': 'Wen Zhengming', '何绍基': 'He Shaoji'}


def load_font(size):
    for path in ('/home/ds/Workspace/DiT/tools/fonts/NotoSerifSC-VF.ttf',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'):
        try:
            return ImageFont.truetype(path, size), True
        except Exception:
            continue
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size), False


def agg(items):
    return {f: round(st.mean(r[f] for r in items), 5) for f in FIELDS}


def main():
    if '--poster-only' in sys.argv:
        results = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
        data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
        test_rows = data['datasets']['test']
        if len(results) != len(test_rows):
            raise ValueError('metrics/data index alignment broken')
        for r, row in zip(results, test_rows):
            r['gt'] = row['file']
        cfg = json.loads((root / 'config.json').read_text())
        trend = {}
        for step_dir in sorted((root / 'evaluations').glob('0*')):
            trend[int(step_dir.name)] = agg(json.loads((step_dir / 'metrics.json').read_text(encoding='utf-8')))
        by_writer = {w: agg([r for r in results if r['calligrapher'] == w])
                     for w in sorted({r['calligrapher'] for r in results})}
        by_script = {s: agg([r for r in results if r['script'] == s])
                     for s in sorted({r['script'] for r in results})}
        build_poster(results, by_writer, by_script, trend, cfg)
        return
    (OUT / 'images').mkdir(parents=True, exist_ok=False)
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    text = torch.load(root / 'text.pt', map_location='cpu', weights_only=True)
    cfg = json.loads((root / 'config.json').read_text())

    model = QwenImage21Transformer2DModel.from_pretrained(root / 'base', subfolder='transformer',
                                                          torch_dtype=torch.bfloat16,
                                                          local_files_only=True).requires_grad_(False).cuda()
    model.add_adapter(LoraConfig(r=cfg['rank'], lora_alpha=cfg['rank'], init_lora_weights='gaussian',
                                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    # Match train.py: adapter params in fp32 so the saved fp32 checkpoint loads exactly.
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
    print('LORA_LOADED_EXACT', flush=True)

    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model, text_encoder=None,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    lp, error = get_lpips('cuda')
    if lp is None:
        raise RuntimeError(error)

    rows = data['datasets']['test']
    results = []
    started = time.monotonic()
    for i, row in enumerate(rows):
        emb = text[(row['character'], row['calligrapher'], row['script'])][None].cuda()
        mask = torch.ones(emb.shape[:2], device='cuda', dtype=torch.bool)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            img = pipe(prompt_embeds=emb, prompt_embeds_mask=mask, height=256, width=256,
                       output_resolution=256, num_inference_steps=GEN_STEPS, true_cfg_scale=1.,
                       use_kv_cache=True, generator=torch.Generator('cpu').manual_seed(90)).images[0]
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
                            ssim=round(ssim(p, g), 5), lpips=round(d, 5),
                            ink_iou=round(ink_iou(p, g), 5)))
        if (i + 1) % 25 == 0:
            print(f'{i + 1}/{len(rows)} {time.monotonic() - started:.0f}s', flush=True)
    pipe.vae.to('cpu'); del pipe
    (OUT / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    by_writer = {w: agg([r for r in results if r['calligrapher'] == w])
                 for w in sorted({r['calligrapher'] for r in results})}
    by_script = {s: agg([r for r in results if r['script'] == s])
                 for s in sorted({r['script'] for r in results})}
    by_ws = {f'{w}-{s}': agg(sub)
             for w in by_writer for s in by_script
             if (sub := [r for r in results if r['calligrapher'] == w and r['script'] == s])}
    trend = {}
    for step_dir in sorted((root / 'evaluations').glob('0*')):
        val = json.loads((step_dir / 'metrics.json').read_text(encoding='utf-8'))
        trend[int(step_dir.name)] = agg(val)
    summary = dict(checkpoint=str(CKPT), resolution=256, steps=GEN_STEPS, seed=90, n=len(results),
                   overall=agg(results), by_writer=by_writer, by_script=by_script,
                   by_writer_script=by_ws, validation_trend=trend)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps(dict(overall=summary['overall'], by_script=by_script),
                                ensure_ascii=False), flush=True)

    build_poster(results, by_writer, by_script, trend, cfg)


def build_poster(results, by_writer, by_script, trend, cfg):
    font, cjk = load_font(20)
    small, _ = load_font(16)
    tiny, _ = load_font(14)
    title_f, _ = load_font(26)
    writers = sorted(by_writer)
    W = 800
    header = 320
    block_h = 34 + 2 * (256 + 46)
    H = header + len(writers) * block_h + 60
    canvas = Image.new('RGB', (W, H), '#eef2f6')
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), 'Top10 Qwen-Image-2.1-Turbo LoRA - prompt-only conditioning - test200',
              font=title_f, fill='black')
    draw.text((14, 46), f"256 / rank{cfg['rank']} / LR {cfg['lr']} cosine / 20k steps / "
                        f"B{cfg['batch']} / Turbo 8-step sampling / seed 90 / checkpoint 0020000",
              font=font, fill='#333')
    line = 78
    overall = agg(results)
    draw.text((14, line), 'OVERALL  ' + ''.join(f'{f.upper()} {overall[f]:.4f}  ' for f in FIELDS),
              font=font, fill='black')
    line += 30
    draw.text((14, line), '   ' + '   '.join(f'{s}: S {m["ssim"]:.3f} L {m["lpips"]:.3f} I {m["ink_iou"]:.3f}'
                                             for s, m in by_script.items()), font=font, fill='#333')
    line += 30
    draw.text((14, line), 'val trend:  ' + '  '.join(
        f'{k//1000}k S {v["ssim"]:.3f} L {v["lpips"]:.3f} I {v["ink_iou"]:.3f}'
        for k, v in sorted(trend.items())), font=small, fill='#555')
    line += 26
    draw.text((14, line), 'rows: writer header = test mean over all scripts; below = two median-SSIM '
                          'test samples (GT left, generated right)', font=small, fill='#555')

    y = header
    for w in writers:
        m = by_writer[w]
        label = f'{w} ({WRITER_EN[w]})' if cjk else WRITER_EN[w]
        draw.text((14, y), f'{label}   S {m["ssim"]:.3f}  L {m["lpips"]:.3f}  I {m["ink_iou"]:.3f}',
                  font=font, fill='black')
        y += 34
        picks = []
        for s in ('楷', '行', '隶'):
            sub = sorted([r for r in results if r['calligrapher'] == w and r['script'] == s],
                         key=lambda r: r['ssim'])
            if sub:
                picks.append(sub[len(sub) // 2])
            if len(picks) == 2:
                break
        for r in picks:
            pred = Image.open(OUT / 'images' / r['image']).convert('RGB')
            gt = Image.open(root / r['gt']).convert('RGB')
            canvas.paste(gt, (14, y))
            canvas.paste(pred, (280, y))
            draw.rectangle((280, y, 535, y + 255), outline='#c0392b', width=3)
            draw.text((14, y + 258), 'GT', font=font, fill='black')
            draw.text((280, y + 258), '生成 (GEN)', font=font, fill='#c0392b')
            char = r['character'] if cjk else f'U+{ord(r["character"]):04X}'
            draw.text((546, y + 4), f'{char} ({r["script"]})', font=font, fill='black')
            draw.text((546, y + 32), f'S {r["ssim"]:.3f}', font=small, fill='#333')
            draw.text((546, y + 54), f'L {r["lpips"]:.3f}', font=small, fill='#333')
            draw.text((546, y + 76), f'I {r["ink_iou"]:.3f}', font=small, fill='#333')
            y += 256 + 46
    canvas.save(OUT / 'poster.png')
    canvas.save(OUT / 'poster.jpg', quality=72, optimize=True)
    print('POSTER_READY', OUT / 'poster.jpg', 'cjk_font', cjk, flush=True)


if __name__ == '__main__':
    main()
