"""D1 TTF-vs-Qwen-vs-GT, D2 writer-swap multi-seed stability, D3 train/test quality gap."""
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

root = Path('/home/ds/Workspace/xy/qwen-lora-top10-20261024')
store = Path('/home/ds/Workspace/DiT/data/std_soft_glyphs_top10_v1')
sys.path.insert(0, str(root))
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, str(root / 'vendor'))
sys.path.insert(0, '/home/ds/Workspace/DiT')

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline
from src.eval.metrics import ssim, get_lpips, skel_iou
from src.eval.metrics_ink import ink_iou
from src.utils.std_soft_glyph import soften_mask

CKPT = root / 'checkpoints' / '0020000'
FIELDS = ('ssim', 'lpips', 'ink_iou', 'skel_iou')
SCRIPT_ID = {'楷': 0, '行': 3, '隶': 4}


def metrics_all(pred_img, gt_img, lp):
    p = np.asarray(pred_img.convert('RGB'), dtype=np.float32) / 255
    g = np.asarray(gt_img.convert('RGB'), dtype=np.float32) / 255
    ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
    with torch.inference_mode():
        d = float(lp(ten(p), ten(g)).item())
    return dict(ssim=round(ssim(p, g), 5), lpips=round(d, 5),
                ink_iou=round(ink_iou(p, g), 5), skel_iou=round(skel_iou(p, g), 5))


def load_ttf(script, char):
    index = {}
    with (store / 'index.csv').open(encoding='utf-8') as stream:
        import csv as _csv
        for row in _csv.DictReader(stream):
            index[(int(row['script_id']), row['character'])] = row['path']
    return index[(SCRIPT_ID[script], char)]


def ttf_display256(script, char, sigma=0.65):
    mask = Image.open(store / load_ttf(script, char)).convert('L')
    soft = soften_mask(mask, sigma)
    black_on_white = Image.eval(soft, lambda v: 255 - v).resize((256, 256), Image.Resampling.BILINEAR)
    return black_on_white.convert('RGB')


def main():
    out = root / 'p1_diagnostics'
    out.mkdir(exist_ok=False)
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    cfg = json.loads((root / 'config.json').read_text())
    test = data['datasets']['test']
    qwen_imgs = {int(p.name[:3]): p for p in (root / 'test_eval_20k' / 'images').glob('*.png')}

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
    model.eval()
    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    lp, error = get_lpips('cuda')
    if lp is None:
        raise RuntimeError(error)

    # ---------------- D1 ----------------
    d1 = {'ttf_vs_gt': [], 'qwen_vs_gt': [], 'qwen_vs_ttf': []}
    started = time.monotonic()
    for i, row in enumerate(test):
        gt = Image.open(root / row['file']).convert('RGB')
        qwen = Image.open(qwen_imgs[i]).convert('RGB')
        ttf = ttf_display256(row['script'], row['character'])
        d1['ttf_vs_gt'].append(metrics_all(ttf, gt, lp))
        d1['qwen_vs_gt'].append(metrics_all(qwen, gt, lp))
        d1['qwen_vs_ttf'].append(metrics_all(qwen, ttf, lp))
    d1_summary = {k: {f: round(st.mean(r[f] for r in v), 5) for f in FIELDS} for k, v in d1.items()}
    (out / 'd1_summary.json').write_text(json.dumps(
        dict(summary=d1_summary, n=len(test), ttf='std_soft store 64->256, sigma0.65, kai/xing/li cascade'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('D1', json.dumps(d1_summary, ensure_ascii=False), f'{time.monotonic() - started:.0f}s', flush=True)

    picks = [test[i] for i in range(0, len(test), max(1, len(test) // 20))][:20]
    font = ImageFont.truetype('/home/ds/Workspace/DiT/tools/fonts/NotoSerifSC-VF.ttf', 20)
    poster = Image.new('RGB', (3 * 256 + 80, 60 + len(picks) * (256 + 28)), 'white')
    draw = ImageDraw.Draw(poster)
    for col, label in enumerate(('TTF(STD)', 'Qwen', 'GT')):
        draw.text((10 + col * 266, 16), label, font=font, fill='black')
    for r_i, row in enumerate(picks):
        y = 60 + r_i * (256 + 28)
        i = test.index(row)
        poster.paste(ttf_display256(row['script'], row['character']), (10, y))
        poster.paste(Image.open(qwen_imgs[i]).convert('RGB'), (276, y))
        poster.paste(Image.open(root / row['file']).convert('RGB'), (542, y))
        draw.text((10, y + 258), f'{row["calligrapher"]} {row["script"]} {row["character"]}', font=font, fill='black')
    poster.save(out / 'd1_poster.jpg', quality=75, optimize=True)

    # ---------------- D2 ----------------
    writers = data['writers']
    trio = [writers.index(w) for w in ('颜真卿', '王羲之', '赵孟頫')]
    d2_chars = []
    for script in ('楷', '行', '隶'):
        d2_chars.extend([r for r in test if r['script'] == script][:2])
    d2 = {}
    for row in d2_chars:
        for wi in trio:
            for seed in (90, 91, 92):
                writer = writers[wi]
                prompt = (f'书法家：{writer}。书体：{row["script"]}书。'
                          f'请用{writer}的{row["script"]}书风格书写汉字「{row["character"]}」。'
                          '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    img = pipe(prompt=prompt, height=256, width=256, output_resolution=256,
                               num_inference_steps=8, true_cfg_scale=1., use_kv_cache=True,
                               generator=torch.Generator('cpu').manual_seed(seed)).images[0]
                white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
                d2[(row['character'], wi, seed)] = white.convert('RGB')

    def pair_ssim(a, b):
        x = np.asarray(a, dtype=np.float32) / 255
        y = np.asarray(b, dtype=np.float32) / 255
        return ssim(x, y)

    seed_stability, writer_separation = [], []
    for row in d2_chars:
        c = row['character']
        for wi in trio:
            imgs = [d2[(c, wi, s)] for s in (90, 91, 92)]
            seed_stability.extend(pair_ssim(imgs[i], imgs[j]) for i in range(3) for j in range(i + 1, 3))
        for seed in (90, 91, 92):
            imgs = [d2[(c, wi, seed)] for wi in trio]
            writer_separation.extend(pair_ssim(imgs[i], imgs[j]) for i in range(3) for j in range(i + 1, 3))
    d2_summary = dict(seed_stability_ssim=round(st.mean(seed_stability), 5),
                      writer_separation_ssim=round(st.mean(writer_separation), 5),
                      n_chars=len(d2_chars), writers=[writers[w] for w in trio], seeds=[90, 91, 92],
                      note='stability高=同书家换seed稳；separation低=换书家字形差异大（低是好事，但需配合内容不变）')
    (out / 'd2_summary.json').write_text(json.dumps(d2_summary, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    grid = Image.new('RGB', (len(d2_chars) * 130 + 10, 3 * 3 * 130 + 60), 'white')
    draw = ImageDraw.Draw(grid)
    for ci, row in enumerate(d2_chars):
        for ri, wi in enumerate(trio):
            for si, seed in enumerate((90, 91, 92)):
                img = d2[(row['character'], wi, seed)].resize((124, 124))
                x = 10 + ci * 130
                y = 50 + (ri * 3 + si) * 130
                grid.paste(img, (x, y))
                if si == 0:
                    draw.text((x, y - 20), f'{writers[wi]} {row["character"]}', font=font, fill='black')
    grid.save(out / 'd2_grid.jpg', quality=75, optimize=True)
    print('D2', json.dumps(d2_summary, ensure_ascii=False), flush=True)

    # ---------------- D3 ----------------
    train = data['datasets']['train']
    rng = random.Random(20261024)
    pools = {}
    for r in train:
        pools.setdefault((r['writer_index'], r['script_index']), []).append(r)
    slots = sorted(pools)
    picked = []
    while len(picked) < 200:
        for slot in slots:
            if pools[slot]:
                picked.append(pools[slot].pop(rng.randrange(len(pools[slot]))))
            if len(picked) == 200:
                break
    d3_train = []
    started = time.monotonic()
    for i, row in enumerate(picked):
        emb_key = (row['character'], row['calligrapher'], row['script'])
        # train rows share text cache with all rows; fall back to raw prompt if absent
        from cache import text_prompt
        prompt = text_prompt(row['character'], row['calligrapher'], row['script'])
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            img = pipe(prompt=prompt, height=256, width=256, output_resolution=256,
                       num_inference_steps=8, true_cfg_scale=1., use_kv_cache=True,
                       generator=torch.Generator('cpu').manual_seed(90)).images[0]
        white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
        pred = white.convert('RGB')
        gt = Image.open(root / row['file']).convert('RGB')
        d3_train.append(metrics_all(pred, gt, lp))
        if (i + 1) % 50 == 0:
            print(f'd3 {i + 1}/200 {time.monotonic() - started:.0f}s', flush=True)
    d3_test = json.loads((root / 'test_eval_20k' / 'metrics.json').read_text(encoding='utf-8'))
    common = ('ssim', 'lpips', 'ink_iou')  # test metrics.json predates skel_iou
    d3_summary = dict(
        train200={f: round(st.mean(r[f] for r in d3_train), 5) for f in common},
        test200={f: round(st.mean(r[f] for r in d3_test), 5) for f in common},
        gap={f: round(st.mean(r[f] for r in d3_train) - st.mean(r[f] for r in d3_test), 5) for f in common},
        note='same protocol 256/8step/seed90; train chars seen by LoRA 20k training')
    (out / 'd3_summary.json').write_text(json.dumps(
        dict(summary=d3_summary, per_image_train=d3_train), ensure_ascii=False, indent=2), encoding='utf-8')
    print('D3', json.dumps(d3_summary, ensure_ascii=False), flush=True)
    pipe.vae.to('cpu'); del pipe
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
