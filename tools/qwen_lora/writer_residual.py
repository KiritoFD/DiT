"""P1: writer-residual LoRA. Merge frozen shared-20k into base, add fresh rank8, train on one writer.

Deployment math: base + shared(20k) + residual(rank8) == merged-base + residual (exact).
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
    import fcntl
    gpu_lock = (root / 'gpu.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    import numpy as np
    import torch
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler
    torch.set_num_threads(8); torch.manual_seed(20261023); random.seed(20261023)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads((root / 'config.json').read_text())
    shared_root = Path(config['shared_root'])
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    cfg = json.loads((shared_root / 'config.json').read_text())
    shared_ckpt = shared_root / 'checkpoints' / '0020000' / 'pytorch_lora_weights.safetensors'

    model = QwenImage21Transformer2DModel.from_pretrained(
        root / 'base', subfolder='transformer', torch_dtype=torch.bfloat16,
        local_files_only=True).requires_grad_(False).cuda()
    model.add_adapter(LoraConfig(r=cfg['rank'], lora_alpha=cfg['rank'],
                                 init_lora_weights='gaussian',
                                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    saved = {k.removeprefix('transformer.'): v for k, v in load_file_safe(shared_ckpt).items()}
    set_peft_model_state_dict(model, saved, adapter_name='default')
    current = get_peft_model_state_dict(model)
    if set(current) != set(saved) or any(not torch.equal(current[k].detach().cpu(), v)
                                         for k, v in saved.items()):
        raise RuntimeError('Shared LoRA load was not exact')
    model.merge_and_unload(safe_merge=True)
    print('SHARED_MERGED', flush=True)
    model.add_adapter(LoraConfig(r=config['rank'], lora_alpha=config['rank'],
                                 init_lora_weights='gaussian',
                                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    model.enable_gradient_checkpointing() if config.get('gradient_checkpointing') else None
    model.train()
    lora = [p for p in model.parameters() if p.requires_grad]
    for p in lora:
        p.data = p.data.float()
    optimizer = torch.optim.AdamW([{'params': lora, 'lr': config['lr']}], weight_decay=.01)

    latent = torch.load(root / 'latents.pt', map_location='cpu', weights_only=True)
    text = torch.load(root / 'text.pt', map_location='cpu', weights_only=True)
    indices = {name: i for i, name in enumerate(latent['names'])}
    wi = int(config['writer_index'])
    train_rows = [r for r in data['datasets']['train']
                  if r['writer_index'] == wi and r['file'] in indices]
    val_rows = [r for r in data['datasets']['validation'] if r['writer_index'] == wi]
    if not train_rows or not val_rows:
        raise ValueError(f'empty rows for writer {wi}')
    print('WRITER', data['writers'][wi], 'train', len(train_rows), 'val', len(val_rows), flush=True)

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
        from safetensors.torch import load_file as lf
        state = torch.load(resume / 'state.pt', map_location='cpu', weights_only=False)
        for key in ('rank', 'batch'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume changed {key}')
        savedr = {k.removeprefix('transformer.'): v
                  for k, v in lf(str(resume / 'pytorch_lora_weights.safetensors')).items()}
        set_peft_model_state_dict(model, savedr, adapter_name='default')
        optimizer.load_state_dict(state['optimizer'])
        start_step = int(state['step'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng'])
        print('RESUME_VERIFIED', start_step, flush=True)

    status = root / 'train_status.json'
    st = dict(pid=os.getpid(), phase='training', step=start_step, max_steps=config['max_steps'],
              writer=data['writers'][wi], lora_parameters=sum(p.numel() for p in lora),
              train_rows=len(train_rows))
    st['shared_merged'] = True

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
            key = (row['character'], row['calligrapher'], row['script'])
            emb = text[key][None].cuda()
            mask = torch.ones(emb.shape[:2], device='cuda', dtype=torch.bool)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                img = pipe(prompt_embeds=emb, prompt_embeds_mask=mask, height=256, width=256,
                           output_resolution=256, num_inference_steps=8, true_cfg_scale=1.,
                           use_kv_cache=True,
                           generator=torch.Generator('cpu').manual_seed(90)).images[0]
            white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
            pred = white.convert('RGB')
            name = f'U{ord(row["character"]):04X}_{len(metrics)}.png'
            pred.save(out / name)
            gt = Image.open(root / row['file']).convert('RGB')
            p = np.asarray(pred, dtype=np.float32) / 255
            g = np.asarray(gt, dtype=np.float32) / 255
            ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
            with torch.inference_mode():
                d = float(lp(ten(p), ten(g)).item())
            metrics.append(dict(character=row['character'], script=row['script'],
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
        QwenImage21Pipeline.save_lora_weights(str(dest),
                                              transformer_lora_layers=get_peft_model_state_dict(model))
        torch.save(dict(optimizer=optimizer.state_dict(), step=step, config=config,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        python_rng=random.getstate(), writer_index=wi,
                        data_sha256=hashlib.sha256((root / 'data.json').read_bytes()).hexdigest(),
                        merged_shared_sha256=hashlib.sha256(shared_ckpt.read_bytes()).hexdigest()),
                   dest / 'state.pt')

    for step in range(start_step + 1, config['max_steps'] + 1):
        optimizer.zero_grad(set_to_none=True)
        peak = config['lr']
        if step <= config['warmup']:
            group_lr = peak * min(1., step / config['warmup']) * \
                (.1 + .9 * .5 * (1 + math.cos(math.pi * step / config['max_steps'])))
        else:
            frac = (step - config['warmup']) / (config['max_steps'] - config['warmup'])
            group_lr = config['min_lr'] + (peak - config['min_lr']) * .5 * (1 + math.cos(math.pi * frac))
        optimizer.param_groups[0]['lr'] = group_lr
        chosen = [random.choice(train_rows) for _ in range(config['batch'])]
        selected = torch.tensor([indices[r['file']] for r in chosen])
        mu = latent['mean'][selected].cuda().float(); sd = latent['std'][selected].cuda().float()
        clean = ((mu + sd * torch.randn_like(mu) - mean) / std).to(torch.bfloat16)
        noise = torch.randn_like(clean)
        index = torch.randint(len(timesteps), (config['batch'],), device='cuda')
        sigma = sigmas[index].view(config['batch'], 1, 1, 1, 1)
        z = (1 - sigma) * clean + sigma * noise
        sequences = [text[(r['character'], r['calligrapher'], r['script'])] for r in chosen]
        emb = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).cuda()
        lengths = torch.tensor([len(v) for v in sequences], device='cuda')
        text_mask = torch.arange(emb.shape[1], device='cuda')[None] < lengths[:, None]
        h, w = z.shape[-2:]
        packed = QwenImage21Pipeline._pack_latents(z, config['batch'], 64, h, w)
        img_mask = torch.cat([torch.zeros(config['batch'], emb.shape[1], device='cuda', dtype=torch.bool),
                              torch.ones(config['batch'], h * w // 4, device='cuda', dtype=torch.bool)], 1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(hidden_states=packed, encoder_hidden_states=emb,
                               encoder_hidden_states_mask=text_mask,
                               timestep=timesteps[index] / 1000,
                               img_shapes=[[(1, h, w)]] * config['batch'], img_mask=img_mask,
                               return_dict=False)[0][:, -packed.shape[1]:]
            prediction = QwenImage21Pipeline._unpack_latents(
                prediction, config['resolution'], config['resolution'], 16)
            loss = (prediction.float() - (noise - clean).float()).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite FM loss')
        loss.backward()
        if step == 1:
            norm = sum(float(p.grad.float().square().sum()) for p in lora if p.grad is not None) ** .5
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError(f'Invalid gradient {norm}')
            st['initial_grad_norm'] = norm
        torch.nn.utils.clip_grad_norm_(lora, 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0 or step == config['max_steps']:
            torch.cuda.synchronize()
            st.update(step=step, loss=float(loss.detach()),
                      seconds_per_step=(time.monotonic() - started) / step,
                      lr=group_lr, peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)
            save_status(); print(json.dumps(st), flush=True)
        if step % config['save_every'] == 0 or step == config['max_steps']:
            checkpoint(step); evaluate(step)
    st['phase'] = 'completed'
    save_status()


def load_file_safe(path):
    from safetensors.torch import load_file
    return load_file(str(path))


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
