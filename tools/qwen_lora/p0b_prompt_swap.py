"""P0b: brushwork-description prompts + real-combo writer-swap + multi-seed. No training."""
import json
import statistics as st
import sys
import time
from collections import defaultdict
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
OUT = root / 'p0b_prompt_swap'
BRUSH = {
    '颜真卿': '用笔雄浑厚重，横画细竖画粗，起笔藏锋，收笔回锋，结体外紧内松、端庄宽博',
    '褚遂良': '用笔瘦劲灵动，提按分明，笔画粗细对比强烈，结体宽绰秀逸、笔意流动',
    '王羲之': '笔势连贯流畅，牵丝映带，结体欹侧变化自然，遒媚劲健',
    '欧阳询': '用笔方硬险劲，笔画挺拔锐利，结体内敛修长、法度森严',
    '柳公权': '骨力遒劲，方圆并施，结体中宫收紧、四面开张',
    '赵孟頫': '用笔圆润流畅，楷中带行意，结体端正匀称、秀美典雅',
    '苏轼': '用笔丰腴跌宕，轻重相间，结体扁平左低右高、率意天真',
    '米芾': '用笔八面出锋，节奏强烈，结体跌宕欹侧、沉着痛快',
    '何绍基': '用笔圆劲婉转，颤笔涩行，回腕含蓄，结体宽博纵逸',
    '文徵明': '用笔清劲秀丽，提按分明，结体端整紧密、温雅精熟',
}


def metrics_pair(pred, gt_img, lp):
    p = np.asarray(pred, dtype=np.float32) / 255
    g = np.asarray(gt_img, dtype=np.float32) / 255
    ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
    with torch.inference_mode():
        d = float(lp(ten(p), ten(g)).item())
    return dict(ssim=round(ssim(p, g), 5), lpips=round(d, 5), ink_iou=round(ink_iou(p, g), 5))


def main():
    (OUT / 'images').mkdir(parents=True, exist_ok=False)
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    cfg = json.loads((root / 'config.json').read_text())
    text = torch.load(root / 'text.pt', map_location='cpu', weights_only=True)
    writers, scripts = data['writers'], data['scripts']
    train, val = data['datasets']['train'], data['datasets']['validation']

    real = {(r['writer_index'], r['script_index']) for r in train}
    model = QwenImage21Transformer2DModel.from_pretrained(
        root / 'base', subfolder='transformer', torch_dtype=torch.bfloat16,
        local_files_only=True).requires_grad_(False).cuda()
    model.add_adapter(LoraConfig(r=cfg['rank'], lora_alpha=cfg['rank'],
                                 init_lora_weights='gaussian',
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

    def gen(writer, script, char, seed, brush=False):
        if brush:
            prompt = (f'书法家：{writer}。书体：{script}书。{BRUSH[writer]}。'
                      f'请严格按上述笔法特征书写汉字「{char}」。'
                      '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
        else:
            prompt = (f'书法家：{writer}。书体：{script}书。'
                      f'请用{writer}的{script}书风格书写汉字「{char}」。'
                      '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            img = pipe(prompt=prompt, height=256, width=256, output_resolution=256,
                       num_inference_steps=8, true_cfg_scale=1., use_kv_cache=True,
                       generator=torch.Generator('cpu').manual_seed(seed)).images[0]
        white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
        return white.convert('RGB')

    # ---------- 1) brushwork prompt vs current template (paired, same seed) ----------
    started = time.monotonic()
    picked = []
    for (wi, si) in sorted(real):
        rows = [r for r in val if r['writer_index'] == wi and r['script_index'] == si]
        picked.extend(rows[:2])
    cond = {'base': defaultdict(list), 'brush': defaultdict(list)}
    for k, r in enumerate(picked):
        for tag, brush in (('base', False), ('brush', True)):
            pred = gen(r['calligrapher'], r['script'], r['character'], 90, brush=brush)
            gt = Image.open(root / r['file']).convert('RGB')
            m = metrics_pair(pred, gt, lp)
            cond[tag][r['script']].append(m)
            cond[tag]['all'].append(m)
            pred.save(OUT / 'images' / f'{tag}_{k:02d}_{r["calligrapher"]}_{r["script"]}_{r["character"]}.png')
        if (k + 1) % 10 == 0:
            print(f'prompt {k + 1}/{len(picked)} {time.monotonic() - started:.0f}s', flush=True)
    prompt_summary = {
        tag: {s: {f: round(st.mean(x[f] for x in v), 5) for f in ('ssim', 'lpips', 'ink_iou')}
              for s, v in d.items()}
        for tag, d in cond.items()}
    (OUT / 'prompt_summary.json').write_text(json.dumps(
        dict(summary=prompt_summary, n_pairs=len(picked), brushes=BRUSH), ensure_ascii=False, indent=2),
        encoding='utf-8')
    print('PROMPT', json.dumps({t: prompt_summary[t]['all'] for t in ('base', 'brush')}), flush=True)

    # ---------- 2) real-combo writer-swap + multi-seed ----------
    by_script = defaultdict(list)
    for (wi, si) in sorted(real):
        by_script[si].append(wi)
    swap = defaultdict(list)
    seed_stab, writer_sep = [], []
    for si, wis in sorted(by_script.items()):
        for wi in wis:
            rows = [r for r in val if r['writer_index'] == wi and r['script_index'] == si]
            if not rows:
                continue
            r0 = rows[0]
            imgs = {w: gen(writers[w], scripts[si], r0['character'], 90) for w in wis}
            seeds = {sd: gen(writers[wi], scripts[si], r0['character'], sd) for sd in (91, 92)}
            pil = lambda im: im
            def pssim(a, b):
                x = np.asarray(a, dtype=np.float32) / 255
                y = np.asarray(b, dtype=np.float32) / 255
                return ssim(x, y)
            ws = list(imgs.values())
            for a in range(len(ws)):
                for b in range(a + 1, len(ws)):
                    writer_sep.append(pssim(ws[a], ws[b]))
            st3 = [imgs[wi], seeds[91], seeds[92]]
            seed_stab.extend((pssim(st3[0], st3[1]), pssim(st3[0], st3[2]), pssim(st3[1], st3[2])))
            swap[scripts[si]].append(dict(writer=writers[wi], char=r0['character'],
                                          n_candidates=len(wis)))
            for w, im in imgs.items():
                im.save(OUT / 'images' / f'swap_s{si}_w{w}_{r0["character"]}.png')
    swap_summary = dict(seed_stability_ssim=round(st.mean(seed_stab), 5),
                        writer_separation_ssim=round(st.mean(writer_sep), 5),
                        n_combos=sum(len(v) for v in swap.values()),
                        per_script={k: len(v) for k, v in swap.items()},
                        note='real training combos only; same char+seed across writers, 3 seeds for stability')
    (OUT / 'swap_summary.json').write_text(json.dumps(swap_summary, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    print('SWAP', json.dumps(swap_summary, ensure_ascii=False), flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
