"""train_std.py: full native conditioning protocol in the training loop.

hidden = cat([std_glyph_vae_latents(256 tok), noised_target(256 tok)]);
img_mask = cat([encode pad (VL slots), ones(target//4=64)]); img_shapes = [[(1,16,16),(1,16,16)]];
loss on prediction[:, -256:] only. Matches pipeline __call__ exactly.
"""
import ast
from pathlib import Path

p = Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025/code/train_std.py')
s = p.read_text(encoding='utf-8')

old = """            selected = torch.tensor([indices[r['file']] for r in chosen])
            mu = latent['mean'][selected].cuda().float(); sd = latent['std'][selected].cuda().float()
            clean = ((mu + sd * torch.randn_like(mu) - mean) / std).to(torch.bfloat16)
            noise = torch.randn_like(clean)
            index = torch.randint(len(timesteps), (config['batch'],), device='cuda')
            sigma = sigmas[index].view(config['batch'], 1, 1, 1, 1)
            z = (1 - sigma) * clean + sigma * noise
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                per = [pipe.encode_prompt(prompt=[prompts[i]], image=[images[i]], device='cuda')
                       for i in range(config['batch'])]
            emb = torch.nn.utils.rnn.pad_sequence([e[0][0] for e in per], batch_first=True)
            enc_mask = torch.nn.utils.rnn.pad_sequence(
                [e[1][0] if e[1] is not None else torch.ones(e[0].shape[1], dtype=torch.bool,
                                                             device=e[0].device) for e in per],
                batch_first=True)
            img_pad = torch.nn.utils.rnn.pad_sequence(
                [e[2][0] if e[2] is not None else torch.zeros(e[0].shape[1], dtype=torch.bool,
                                                              device=e[0].device) for e in per],
                batch_first=True)
            h, w = z.shape[-2:]
            packed = QwenImage21Pipeline._pack_latents(z, config['batch'], 64, h, w)
            img_mask = torch.cat([img_pad, torch.ones(config['batch'], h * w // 4, device='cuda',
                                                      dtype=torch.bool)], 1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(hidden_states=packed, encoder_hidden_states=emb,
                                   encoder_hidden_states_mask=enc_mask,
                                   timestep=timesteps[index] / 1000,
                                   img_shapes=[[(1, h, w)]] * config['batch'], img_mask=img_mask,
                                   return_dict=False)[0][:, -packed.shape[1]:]
                prediction = QwenImage21Pipeline._unpack_latents(
                    prediction, config['resolution'], config['resolution'], 16)
                loss = (prediction.float() - (noise - clean).float()).square().mean()"""

new = """            selected = torch.tensor([indices[r['file']] for r in chosen])
            mu = latent['mean'][selected].cuda().float(); sd = latent['std'][selected].cuda().float()
            clean = ((mu + sd * torch.randn_like(mu) - mean) / std).to(torch.bfloat16)
            noise = torch.randn_like(clean)
            index = torch.randint(len(timesteps), (config['batch'],), device='cuda')
            sigma = sigmas[index].view(config['batch'], 1, 1, 1, 1)
            z = (1 - sigma) * clean + sigma * noise
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                per = [pipe.encode_prompt(prompt=[prompts[i]], image=[images[i]], device='cuda')
                       for i in range(config['batch'])]
                emb = torch.nn.utils.rnn.pad_sequence([e[0][0] for e in per], batch_first=True)
                enc_mask = torch.nn.utils.rnn.pad_sequence(
                    [e[1][0] if e[1] is not None else torch.ones(e[0].shape[1], dtype=torch.bool,
                                                                 device=e[0].device) for e in per],
                    batch_first=True)
                img_pad = torch.nn.utils.rnn.pad_sequence(
                    [e[2][0] if e[2] is not None else torch.zeros(e[0].shape[1], dtype=torch.bool,
                                                                  device=e[0].device) for e in per],
                    batch_first=True)
                pixels = torch.stack([
                    torch.from_numpy(np.asarray(im.copy(), dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1
                    for im in images]).cuda().to(torch.bfloat16).unsqueeze(2)
                cond_lat = vae.encode(pixels).latent_dist.mean
                h, w = cond_lat.shape[-2:]
                cond_tok = QwenImage21Pipeline._pack_latents(cond_lat, config['batch'], 64, h, w)
            packed = QwenImage21Pipeline._pack_latents(
                z.squeeze(2) if z.dim() == 5 else z, config['batch'], 64, z.shape[-2], z.shape[-1])
            target_tokens = packed.shape[1]
            hidden = torch.cat([cond_tok, packed], 1)
            img_mask = torch.cat([img_pad,
                                  torch.ones(config['batch'], target_tokens // 4, device='cuda',
                                             dtype=torch.bool)], 1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(hidden_states=hidden, encoder_hidden_states=emb,
                                   encoder_hidden_states_mask=enc_mask,
                                   timestep=timesteps[index] / 1000,
                                   img_shapes=[[(1, h, w), (1, h, w)]] * config['batch'],
                                   img_mask=img_mask, return_dict=False)[0]
                prediction = prediction[:, -target_tokens:]
                clean4 = clean.squeeze(2) if clean.dim() == 5 else clean
                noise4 = noise.squeeze(2) if noise.dim() == 5 else noise
                pred_img = QwenImage21Pipeline._unpack_latents(
                    prediction, config['resolution'], config['resolution'], 16)
                loss = (pred_img.float() - (noise4 - clean4).float()).square().mean()"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)

old2 = """    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')"""
new2 = """    from diffusers import AutoencoderKLQwenImage21
    vae = AutoencoderKLQwenImage21.from_pretrained(root / 'base', subfolder='vae',
                                                   torch_dtype=torch.bfloat16,
                                                   local_files_only=True).eval().requires_grad_(False).cuda()
    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')"""
assert s.count(old2) == 1
s = s.replace(old2, new2)
p.write_text(s, encoding='utf-8')
ast.parse(s)
print('native protocol patched')
