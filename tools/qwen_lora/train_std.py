"""P2-A: STD-conditioned LoRA. Native image conditioning with the TARGET char's TTF glyph.

base + shared(20k, frozen) + std-adapter(rank16, trained). Condition image = fresh font-cascade
render of the target character (256, white bg black ink, sigma .65) -> Qwen3-VL image tokens.
Loss on target only, native img_mask protocol (same as pipeline).
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from cache import setup


def main(root, resume=None):
    setup(root)
    sys.path.insert(0, '/home/ds/Workspace/DiT')
    os.chdir('/home/ds/Workspace/DiT')
    import fcntl
    gpu_lock = (root / 'gpu.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    import numpy as np
    import torch
    from PIL import Image
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler
    from src.utils.std_soft_glyph import load_font_config, FontCoverage, render_base_mask, soften_mask
    torch.set_num_threads(8); torch.manual_seed(20261023); random.seed(20261023)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads((root / 'config.json').read_text())
    shared_root = Path(config['shared_root'])
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    cfg = json.loads((shared_root / 'config.json').read_text())
    shared_ckpt = shared_root / 'checkpoints' / '0020000' / 'pytorch_lora_weights.safetensors'

    font_config, script_specs, font_dirs = load_font_config('configs/std_soft_fonts.json', [])
    coverage = FontCoverage()
    SID = {'楷': 0, '行': 3, '隶': 4}

    def std_image(script, char):
        face = None
        for f in script_specs[SID[script]].faces:
            if coverage.supports(f, char):
                face = f
                break
        if face is None:
            raise KeyError(f'no font for {script} {char!r}')
        mask = render_base_mask(char, face, render_size=512)
        soft = soften_mask(mask, 0.65).resize((256, 256), Image.Resampling.BILINEAR)
        return Image.eval(soft, lambda v: 255 - v).convert('RGB')

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

    from safetensors.torch import load_file
    saved = strip_adapter({k: v for k, v in load_file(str(shared_ckpt)).items()})
    for n, pp in model.named_parameters():
        if pp.requires_grad:
            pp.data = pp.data.float()
    set_peft_model_state_dict(model, saved, adapter_name='shared')
    current = strip_adapter(get_peft_model_state_dict(model, adapter_name='shared'))
    if set(current) != set(saved) or any(not torch.equal(current[k].detach().cpu(), v.float())
                                         for k, v in saved.items()):
        raise RuntimeError('Shared LoRA load was not exact')
    for n, p in model.named_parameters():
        p.requires_grad_(False)
        if 'lora_A.shared' in n or 'lora_B.shared' in n:
            p.data = p.data.to(torch.bfloat16)
    model.add_adapter(LoraConfig(r=config['rank'], lora_alpha=config['rank'],
                                 init_lora_weights='gaussian',
                                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']),
                      adapter_name='std')
    model.set_adapter(['shared', 'std'])
    model.train()
    lora = [p for p in model.parameters() if p.requires_grad]
    for p in lora:
        p.data = p.data.float()
    print('DUAL_ADAPTER trainable', sum(p.numel() for p in lora), flush=True)
    optimizer = torch.optim.AdamW([{'params': lora, 'lr': config['lr']}], weight_decay=.01)

    latent = torch.load(root / 'latents.pt', map_location='cpu', weights_only=True)
    indices = {name: i for i, name in enumerate(latent['names'])}
    train_rows = [r for r in data['datasets']['train'] if r['file'] in indices]
    val_rows = data['datasets']['validation']

    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        root / 'base', subfolder='scheduler', local_files_only=True)
    sigmas = scheduler.sigmas.to('cuda'); timesteps = scheduler.timesteps.to('cuda')
    mean = torch.tensor(latent['latents_mean'], device='cuda').view(1, 64, 1, 1, 1)
    std = torch.tensor(latent['latents_std'], device='cuda').view(1, 64, 1, 1, 1)

    start_step = 0
    if resume is not None:
        state = torch.load(resume / 'state.pt', map_location='cpu', weights_only=False)
        for key in ('rank', 'batch', 'accumulation'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume changed {key}')
        savedr = {k.removeprefix('transformer.')
                  .replace('.lora_A.std.', '.lora_A.')
                  .replace('.lora_B.std.', '.lora_B.'): v
                  for k, v in load_file(str(resume / 'pytorch_lora_weights.safetensors')).items()}
        set_peft_model_state_dict(model, savedr, adapter_name='std')
        optimizer.load_state_dict(state['optimizer'])
        start_step = int(state['step'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng'])
        print('RESUME_VERIFIED', start_step, flush=True)

    status = root / 'train_status.json'
    st = dict(pid=os.getpid(), phase='training', step=start_step, max_steps=config['max_steps'],
              lora_parameters=sum(p.numel() for p in lora), train_rows=len(train_rows),
              conditioning='prompt+std_glyph_image')

    def save_status():
        status.write_text(json.dumps(st, indent=2))

    save_status()
    started = time.monotonic()

    def evaluate(step):
        from src.eval.metrics import ssim, get_lpips
        from src.eval.metrics_ink import ink_iou
        lp, error = get_lpips('cuda')
        if lp is None:
            raise RuntimeError(error)
        out = root / 'evaluations' / f'{step:07d}'
        out.mkdir(parents=True, exist_ok=False)
        metrics = []
        model.eval()
        for row in val_rows:
            char, script = row['character'], row['script']
            prompt = (f'书法家：{row["calligrapher"]}。书体：{script}书。'
                      f'请以参考字形的结构与布局为标准，用{row["calligrapher"]}的{script}书笔法书写汉字「{char}」。'
                      '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
            glyph = std_image(script, char)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                img = pipe(prompt=prompt, image=[glyph], height=256, width=256,
                           output_resolution=256, num_inference_steps=8, true_cfg_scale=1.,
                           use_kv_cache=True,
                           generator=torch.Generator('cpu').manual_seed(90)).images[0]
            white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
            pred = white.convert('RGB')
            name = f'{len(metrics):03d}_{row["calligrapher"]}_{script}_{char}.png'
            pred.save(out / name)
            gt = Image.open(root / row['file']).convert('RGB')
            p = np.asarray(pred, dtype=np.float32) / 255
            g = np.asarray(gt, dtype=np.float32) / 255
            ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
            with torch.inference_mode():
                d = float(lp(ten(p), ten(g)).item())
            metrics.append(dict(character=char, calligrapher=row['calligrapher'], script=script,
                                ssim=round(ssim(p, g), 5), lpips=round(d, 5),
                                ink_iou=round(ink_iou(p, g), 5)))
        model.train()
        agg = {f: round(sum(m[f] for m in metrics) / len(metrics), 5)
               for f in ('ssim', 'lpips', 'ink_iou')}
        (out / 'metrics.json').write_text(json.dumps(dict(agg=agg, per_char=metrics),
                                                    ensure_ascii=False, indent=2), encoding='utf-8')
        del lp
        print('EVAL', step, json.dumps(agg), flush=True)

    def checkpoint(step):
        dest = root / 'checkpoints' / f'{step:07d}'
        dest.mkdir(parents=True, exist_ok=False)
        QwenImage21Pipeline.save_lora_weights(
            str(dest), transformer_lora_layers=get_peft_model_state_dict(model, adapter_name='std'))
        torch.save(dict(optimizer=optimizer.state_dict(), step=step, config=config,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        python_rng=random.getstate(),
                        data_sha256=hashlib.sha256((root / 'data.json').read_bytes()).hexdigest(),
                        shared_sha256=hashlib.sha256(shared_ckpt.read_bytes()).hexdigest()),
                   dest / 'state.pt')

    if start_step == 0 and not (root / 'evaluations' / '0000000' / 'metrics.json').exists():
        (root / 'evaluations').mkdir(exist_ok=True)
        evaluate(0)
    for step in range(start_step + 1, config['max_steps'] + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for _ in range(config['accumulation']):
            chosen = [random.choice(train_rows) for _ in range(config['batch'])]
            prompts = []
            images = []
            for row in chosen:
                char, script = row['character'], row['script']
                prompts.append(f'书法家：{row["calligrapher"]}。书体：{script}书。'
                               f'请以参考字形的结构与布局为标准，用{row["calligrapher"]}的{script}书笔法书写汉字「{char}」。'
                               '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
                images.append(std_image(script, char))
            selected = torch.tensor([indices[r['file']] for r in chosen])
            mu = latent['mean'][selected].cuda().float(); sd = latent['std'][selected].cuda().float()
            clean = ((mu + sd * torch.randn_like(mu) - mean) / std).to(torch.bfloat16)
            noise = torch.randn_like(clean)
            index = torch.randint(len(timesteps), (config['batch'],), device='cuda')
            sigma = sigmas[index].view(config['batch'], 1, 1, 1, 1)
            z = (1 - sigma) * clean + sigma * noise
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                emb, mask, pad = pipe.encode_prompt(prompt=prompts, image=images, device='cuda')
            h, w = z.shape[-2:]
            packed = QwenImage21Pipeline._pack_latents(z, config['batch'], 64, h, w)
            img_mask = torch.cat([pad, torch.ones(config['batch'], h * w // 4, device='cuda',
                                                  dtype=torch.bool)], 1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(hidden_states=packed, encoder_hidden_states=emb,
                                   encoder_hidden_states_mask=mask,
                                   timestep=timesteps[index] / 1000,
                                   img_shapes=[[(1, h, w)]] * config['batch'], img_mask=img_mask,
                                   return_dict=False)[0][:, -packed.shape[1]:]
                prediction = QwenImage21Pipeline._unpack_latents(
                    prediction, config['resolution'], config['resolution'], 16)
                loss = (prediction.float() - (noise - clean).float()).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite FM loss')
            (loss / config['accumulation']).backward()
            total += loss.item() / config['accumulation']
        if step == start_step + 1:
            norm = sum(float(p.grad.float().square().sum()) for p in lora if p.grad is not None) ** .5
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError(f'Invalid gradient {norm}')
            st['initial_grad_norm'] = norm
        torch.nn.utils.clip_grad_norm_(lora, 1., error_if_nonfinite=True)
        optimizer.step()
        frac = min(1., step / config['max_steps'])
        if step <= config['warmup']:
            lr = config['lr'] * min(1., step / config['warmup']) * \
                (.1 + .9 * .5 * (1 + math.cos(math.pi * step / config['max_steps'])))
        else:
            lr = config['min_lr'] + (config['lr'] - config['min_lr']) * .5 * \
                (1 + math.cos(math.pi * (step - config['warmup']) / (config['max_steps'] - config['warmup'])))
        optimizer.param_groups[0]['lr'] = lr
        if step % 10 == 0 or step == config['max_steps']:
            torch.cuda.synchronize()
            st.update(step=step, loss=total,
                      seconds_per_step=(time.monotonic() - started) / (step - start_step),
                      lr=lr, peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)
            save_status(); print(json.dumps(st), flush=True)
        if step % config['save_every'] == 0 or step == config['max_steps']:
            checkpoint(step); evaluate(step)
    st['phase'] = 'completed'
    save_status()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    try:
        main(a.root.resolve(), a.resume.resolve() if a.resume else None)
    except Exception as error:
        path = a.root / 'train_status.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(phase='failed', error=repr(error))
        path.write_text(json.dumps(state, indent=2))
        raise
