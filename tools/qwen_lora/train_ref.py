"""Turbo LoRA with native reference-image conditioning: prompt + same-writer different-char ref -> target GT.

Mirrors pipeline internals: encode_prompt(prompt, image) -> (embeds, mask, image_pad_mask);
transformer img_mask = [image_pad_mask, ones(target latent slots)]. Loss only on target latents.
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


def main(root, smoke=False, resume=None):
    setup(root)
    import fcntl
    gpu_lock = (root / 'gpu.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    import numpy as np
    import torch
    from PIL import Image
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler
    torch.set_num_threads(8); torch.manual_seed(20261023); random.seed(20261023)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads((root / 'config.json').read_text())
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    model_path = root / 'base'
    status_latents = json.loads((root / 'cache_latents_status.json').read_text())
    if Path(status_latents['model']).resolve() != Path(model_path).resolve() \
            or status_latents['phase'] != 'ready':
        raise ValueError('Wrong/not-ready latents cache')
    identity = json.loads((root / 'latents_identity.json').read_text())
    if identity['data_sha256'] != hashlib.sha256((root / 'data.json').read_bytes()).hexdigest():
        raise ValueError('Latents cache dataset identity changed')
    latent = torch.load(root / 'latents.pt', map_location='cpu', weights_only=True)
    if latent['mean'].shape[-2:] != (config['resolution'] // 16,) * 2:
        raise ValueError('Wrong latent cache resolution')
    indices = {name: i for i, name in enumerate(latent['names'])}
    train_rows = [r for r in data['datasets']['train'] if r['file'] in indices]
    writer_pools = {wid: [r for r in train_rows if r['writer_index'] == wid]
                    for wid in sorted({r['writer_index'] for r in train_rows})}
    ref_pools = {}
    for row0 in train_rows:
        ref_pools.setdefault((row0['writer_index'], row0['script_index']), []).append(row0)

    model = QwenImage21Transformer2DModel.from_pretrained(model_path, subfolder='transformer',
                                                          torch_dtype=torch.bfloat16,
                                                          local_files_only=True).requires_grad_(False).cuda()
    model.add_adapter(LoraConfig(r=config['rank'], lora_alpha=config['rank'],
                                 init_lora_weights='gaussian',
                                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    if config.get('gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    model.train()
    lora = [p for p in model.parameters() if p.requires_grad]
    for p in lora:
        p.data = p.data.float()
    optimizer = torch.optim.AdamW([{'params': lora, 'lr': config['lr']}], weight_decay=.01)
    start_step = 0
    if resume is not None:
        from safetensors.torch import load_file
        state = torch.load(resume / 'state.pt', map_location='cpu', weights_only=False)
        if state['data_sha256'] != hashlib.sha256((root / 'data.json').read_bytes()).hexdigest():
            raise ValueError('Resume dataset identity mismatch')
        for key in ('rank', 'resolution', 'batch', 'accumulation'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume changed {key}')
        saved = {k.removeprefix('transformer.'): v
                 for k, v in load_file(str(resume / 'pytorch_lora_weights.safetensors')).items()}
        set_peft_model_state_dict(model, saved, adapter_name='default')
        optimizer.load_state_dict(state['optimizer'])
        start_step = int(state['step'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng'])
        print('RESUME_VERIFIED', resume, 'step', start_step, flush=True)

    pipe = QwenImage21Pipeline.from_pretrained(model_path, transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler',
                                                                local_files_only=True)
    sigmas = scheduler.sigmas.to('cuda'); timesteps = scheduler.timesteps.to('cuda')
    mean = torch.tensor(latent['latents_mean'], device='cuda').view(1, 64, 1, 1, 1)
    std = torch.tensor(latent['latents_std'], device='cuda').view(1, 64, 1, 1, 1)
    target_steps = 30 if smoke else config['max_steps']
    accumulation = 1 if smoke else config['accumulation']
    batch = 1 if smoke else config['batch']
    state = dict(pid=os.getpid(), phase='training', smoke=smoke, step=start_step,
                 max_steps=target_steps, lora_parameters=sum(p.numel() for p in lora),
                 conditioning='prompt+reference_image', microbatch=batch, accumulation=accumulation)

    def save_status():
        (root / ('smoke_status.json' if smoke else 'train_ref_status.json')).write_text(
            json.dumps(state, indent=2))

    save_status()
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()

    def checkpoint(step):
        dest = root / 'checkpoints' / f'{step:07d}'
        dest.mkdir(parents=True, exist_ok=False)
        QwenImage21Pipeline.save_lora_weights(str(dest),
                                              transformer_lora_layers=get_peft_model_state_dict(model))
        torch.save(dict(optimizer=optimizer.state_dict(), step=step,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        python_rng=random.getstate(), config=config,
                        data_sha256=hashlib.sha256((root / 'data.json').read_bytes()).hexdigest()),
                   dest / 'state.pt')

    def evaluate(step):
        from src.eval.metrics import ssim, get_lpips
        from src.eval.metrics_ink import ink_iou
        lp, error = get_lpips('cuda')
        if lp is None:
            raise RuntimeError(error)
        out = root / 'evaluations' / f'{step:07d}'
        out.mkdir(parents=True, exist_ok=False)
        metrics = []
        for wi in range(len(data['writers'])):
            for si in range(len(data['scripts'])):
                pool = [r for r in data['datasets']['validation']
                        if r['writer_index'] == wi and r['script_index'] == si]
                if not pool:
                    continue
                for row in pool[:config.get('eval_per_slot', 1)]:
                    same = [r for r in train_rows
                            if r['writer_index'] == wi and r['script_index'] == si
                            and r['character'] != row['character']]
                    same.sort(key=lambda r: r['character'])
                    ref = Image.open(root / same[len(same) // 2]['file']).convert('RGB')
                    prompt = (f'书法家：{row["calligrapher"]}。书体：{row["script"]}书。'
                              f'请学习参考真迹的笔画与结体风格，'
                              f'并用{row["calligrapher"]}的{row["script"]}书风格书写汉字「{row["character"]}」。'
                              '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
                    model.eval()
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        img = pipe(prompt=prompt, image=[ref], height=config['resolution'],
                                   width=config['resolution'], output_resolution=config['resolution'],
                                   num_inference_steps=config.get('eval_steps', 8),
                                   true_cfg_scale=1., use_kv_cache=True,
                                   generator=torch.Generator('cpu').manual_seed(90)).images[0]
                    model.train()
                    if len(pipe.scheduler.timesteps) != config.get('eval_steps', 8):
                        raise ValueError('Unexpected sampling steps')
                    white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
                    pred = white.convert('RGB')
                    name = f'U{ord(row["character"]):04X}_w{wi}_s{si}_{len(metrics)}.png'
                    pred.save(out / name)
                    gt = Image.open(root / row['file']).convert('RGB')
                    p = np.asarray(pred, dtype=np.float32) / 255
                    g = np.asarray(gt, dtype=np.float32) / 255
                    ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
                    with torch.inference_mode():
                        d = float(lp(ten(p), ten(g)).item())
                    metrics.append(dict(character=row['character'], writer=row['calligrapher'],
                                        script=row['script'], ssim=round(ssim(p, g), 5),
                                        lpips=round(d, 5), ink_iou=round(ink_iou(p, g), 5)))
        (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
        del lp

    for step in range(start_step + 1, target_steps + 1):
        optimizer.zero_grad(set_to_none=True); total = 0.
        group = optimizer.param_groups[0]; peak = config['lr']
        if step <= config['warmup']:
            group['lr'] = peak * min(1., step / config['warmup']) * \
                (.1 + .9 * .5 * (1 + math.cos(math.pi * step / target_steps)))
        else:
            frac = (step - config['warmup']) / (target_steps - config['warmup'])
            group['lr'] = config['min_lr'] + (peak - config['min_lr']) * .5 * (1 + math.cos(math.pi * frac))
        for _ in range(accumulation):
            chosen = [random.choice(writer_pools[random.choice(list(writer_pools))]) for _ in range(batch)]
            prompts = []
            images = []
            for row in chosen:
                pool = [r for r in ref_pools[(row['writer_index'], row['script_index'])]
                        if r['character'] != row['character']]
                ref = random.choice(pool)
                prompts.append(f'书法家：{row["calligrapher"]}。书体：{row["script"]}书。'
                               f'请学习参考真迹的笔画与结体风格，'
                               f'并用{row["calligrapher"]}的{row["script"]}书风格书写汉字「{row["character"]}」。'
                               '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
                images.append(Image.open(root / ref['file']).convert('RGB'))
            selected = torch.tensor([indices[row['file']] for row in chosen])
            mu = latent['mean'][selected].cuda().float(); sd = latent['std'][selected].cuda().float()
            clean = ((mu + sd * torch.randn_like(mu) - mean) / std).to(torch.bfloat16)
            noise = torch.randn_like(clean)
            index = torch.randint(len(timesteps), (batch,), device='cuda')
            sigma = sigmas[index].view(batch, 1, 1, 1, 1)
            z = (1 - sigma) * clean + sigma * noise
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                emb, mask, pad = pipe.encode_prompt(prompt=prompts, image=images, device='cuda')
            h, w = z.shape[-2:]
            packed = QwenImage21Pipeline._pack_latents(z, batch, 64, h, w)
            img_mask = torch.cat([pad, torch.ones(batch, h * w // 4, device='cuda', dtype=torch.bool)], 1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(hidden_states=packed, encoder_hidden_states=emb,
                                   encoder_hidden_states_mask=mask, timestep=timesteps[index] / 1000,
                                   img_shapes=[[(1, h, w)]] * batch, img_mask=img_mask,
                                   return_dict=False)[0][:, -packed.shape[1]:]
                prediction = QwenImage21Pipeline._unpack_latents(
                    prediction, config['resolution'], config['resolution'], 16)
                loss = (prediction.float() - (noise - clean).float()).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite FM loss')
            (loss / accumulation).backward(); total += loss.item() / accumulation
        if step == start_step + 1:
            norm = sum(float(p.grad.float().square().sum()) for p in lora if p.grad is not None) ** .5
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError(f'Invalid lora gradient: {norm}')
            state['lora_initial_grad_norm'] = norm
        torch.nn.utils.clip_grad_norm_(lora, 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0 or step == start_step + 1 or step == target_steps:
            torch.cuda.synchronize()
            state.update(step=step, loss=total,
                         seconds_per_step=(time.monotonic() - started) / (step - start_step),
                         learning_rates=[g['lr'] for g in optimizer.param_groups],
                         peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)
            save_status(); print(json.dumps(state), flush=True)
        if not smoke and (step % config['save_every'] == 0 or step == target_steps):
            checkpoint(step); evaluate(step)
    state['phase'] = 'completed' if not smoke else 'smoke_done'
    save_status()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--resume', type=Path)
    a = p.parse_args()
    try:
        main(a.root.resolve(), a.smoke, a.resume.resolve() if a.resume else None)
    except Exception as error:
        path = a.root / ('smoke_status.json' if a.smoke else 'train_ref_status.json')
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(phase='failed', error=repr(error)); path.write_text(json.dumps(state, indent=2))
        raise
