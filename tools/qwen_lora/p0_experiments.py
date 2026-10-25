"""P0: writer-swap cross-char recognition + LoRA strength scan, no training."""
import json
import random
import statistics as st
import sys
import time
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


def set_lora_scale(model, weight):
    """Absolute scale on every lora layer: delta*weight; 0.0 == exact base model."""
    hits = 0
    for m in model.modules():
        if hasattr(m, 'lora_A') and hasattr(m, 'set_scale'):
            m.set_scale('default', weight)
            hits += 1
    if hits == 0:
        raise RuntimeError('no lora layers found for scaling')
from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline
from src.eval.metrics import ssim, get_lpips
from src.eval.metrics_ink import ink_iou

CKPT = root / 'checkpoints' / '0020000'
FIELDS = ('ssim', 'lpips', 'ink_iou')
N_CHARS_PER_SCRIPT = 10
N_REF_IMAGES = 20


def main():
    out = root / 'p0_writer_swap'
    (out / 'images').mkdir(parents=True, exist_ok=False)
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    cfg = json.loads((root / 'config.json').read_text())
    text = torch.load(root / 'text.pt', map_location='cpu', weights_only=True)
    writers = data['writers']
    scripts = data['scripts']

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

    # ---- deterministic 30 chars: 10 per script, evenly spaced over sorted test chars ----
    test = data['datasets']['test']
    chosen = []
    for si, script in enumerate(scripts):
        chars = sorted({r['character'] for r in test if r['script'] == script})
        if len(chars) < N_CHARS_PER_SCRIPT:
            raise ValueError(f'not enough {script} chars')
        picks = sorted({chars[round(i * (len(chars) - 1) / (N_CHARS_PER_SCRIPT - 1))]
                        for i in range(N_CHARS_PER_SCRIPT)})
        for c in picks:
            row = next(r for r in test if r['character'] == c and r['script'] == script)
            chosen.append((c, si, row))
    print('CHARS', [(c, scripts[si]) for c, si, _ in chosen], flush=True)

    # ---- writer-swap generation: same seed, only writer name varies ----
    swap = {}
    started = time.monotonic()
    for tag, scale in (('base', 0.0), ('lora', 1.0)):
        if scale == 0.0:
            set_lora_scale(model, 0.0)
        for ci, (char, si, row) in enumerate(chosen):
            for wi, writer in enumerate(writers):
                prompt = (f'书法家：{writer}。书体：{scripts[si]}书。'
                          f'请用{writer}的{scripts[si]}书风格书写汉字「{char}」。'
                          '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    img = pipe(prompt=prompt, height=256, width=256, output_resolution=256,
                               num_inference_steps=8, true_cfg_scale=1., use_kv_cache=True,
                               generator=torch.Generator('cpu').manual_seed(90)).images[0]
                white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
                name = f'{tag}_{ci:02d}_w{wi}s{si}_{char}.png'
                white.convert('RGB').save(out / 'images' / name)
                swap[(tag, ci, wi, si)] = str(Path('images') / name)
        if scale == 0.0:
            set_lora_scale(model, 1.0)
        print(f'swap[{tag}] done {time.monotonic() - started:.0f}s', flush=True)

    # ---- independent writer classifier: mean-LPIPS to same-script train refs, argmin ----
    rng = random.Random(20261024)
    refs = {}
    train = data['datasets']['train']
    for wi, writer in enumerate(writers):
        for si, script in enumerate(scripts):
            pool = sorted(r['file'] for r in train
                          if r['calligrapher'] == writer and r['script'] == script)
            refs[(wi, si)] = rng.sample(pool, min(N_REF_IMAGES, len(pool)))
    ref_tensors = {}
    for key, files in refs.items():
        if not files:
            continue  # writer has no training data for this script
        batch = torch.stack([
            (torch.from_numpy(np.asarray(Image.open(root / f).convert('RGB'), dtype=np.float32) / 255)
             .permute(2, 0, 1) * 2 - 1) for f in files]).cuda()
        ref_tensors[key] = batch
    candidates = {si: [wi for wi in range(len(writers)) if (wi, si) in ref_tensors]
                  for si in range(len(scripts))}

    def classify(image, si, true_char):
        tensor = (torch.from_numpy(np.asarray(image, dtype=np.float32) / 255)
                  .permute(2, 0, 1)[None].cuda() * 2 - 1)
        scores = []
        for wi in candidates[si]:
            batch = ref_tensors[(wi, si)]
            repeated = tensor.repeat(len(batch), 1, 1, 1)
            with torch.inference_mode():
                scores.append(float(lp(repeated, batch).mean().item()))
        return candidates[si][int(np.argmin(scores))], scores

    classifier = dict(refs_per_writer_script={str(k): v for k, v in refs.items()},
                      metric='mean-LPIPS to 20 same-writer same-script train GT, argmin',
                      candidates_per_script={scripts[si]: len(v) for si, v in candidates.items()})
    for tag in ('base', 'lora'):
        correct = 0
        per_script = {s: [0, 0] for s in scripts}
        # true writer of each chosen char = the writer of its test row
        for ci, (char, si, row) in enumerate(chosen):
            true_wi = row['writer_index']
            image = Image.open(out / swap[(tag, ci, true_wi, si)]).convert('RGB')
            pred_wi, _ = classify(image, si, char)
            ok = pred_wi == true_wi
            correct += ok
            per_script[scripts[si]][0] += ok
            per_script[scripts[si]][1] += 1
        classifier[tag] = dict(top1=correct, n=len(chosen),
                               per_script={k: f'{v[0]}/{v[1]}' for k, v in per_script.items()})
        print(f'classifier[{tag}] {correct}/{len(chosen)}', flush=True)
    gt_ceiling = 0
    for ci, (char, si, row) in enumerate(chosen):
        image = Image.open(root / row['file']).convert('RGB')
        pred_wi, _ = classify(image, si, char)
        gt_ceiling += pred_wi == row['writer_index']
    classifier['gt_ceiling'] = dict(top1=gt_ceiling, n=len(chosen))

    # ---- LoRA strength scan on full validation 100 ----
    val = data['datasets']['validation']
    scan = {}
    for scale in (0.0, 0.5, 0.75, 1.0, 1.25):
        if scale != 1.0:
            set_lora_scale(model, scale)
        rows = []
        for i, row in enumerate(val):
            emb = text[(row['character'], row['calligrapher'], row['script'])][None].cuda()
            mask = torch.ones(emb.shape[:2], device='cuda', dtype=torch.bool)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                img = pipe(prompt_embeds=emb, prompt_embeds_mask=mask, height=256, width=256,
                           output_resolution=256, num_inference_steps=8, true_cfg_scale=1.,
                           use_kv_cache=True, generator=torch.Generator('cpu').manual_seed(90)).images[0]
            white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
            pred = white.convert('RGB')
            gt = Image.open(root / row['file']).convert('RGB')
            p = np.asarray(pred, dtype=np.float32) / 255
            g = np.asarray(gt, dtype=np.float32) / 255
            ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
            with torch.inference_mode():
                d = float(lp(ten(p), ten(g)).item())
            rows.append(dict(ssim=ssim(p, g), lpips=d, ink_iou=ink_iou(p, g)))
            if scale == 1.25 and i < 20:
                pred.save(out / 'images' / f'scan{scale}_{i:03d}_{row["calligrapher"]}_{row["script"]}.png')
        scan[str(scale)] = {f: round(st.mean(r[f] for r in rows), 5) for f in FIELDS}
        if scale != 1.0:
            set_lora_scale(model, 1.0)
        print('scan', scale, scan[str(scale)], flush=True)

    summary = dict(checkpoint=str(CKPT), chars=[(c, scripts[si]) for c, si, _ in chosen],
                   writer_classifier=classifier, strength_scan=scan,
                   note='swap: fixed seed90, only writer name varies; scan: validation100, prompt_embeds path')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps(dict(classifier=classifier, scan=scan), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
