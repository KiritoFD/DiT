"""train_std.py: replace batched encode_prompt with per-sample encode + manual pad."""
import ast
from pathlib import Path

p = Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025/code/train_std.py')
s = p.read_text(encoding='utf-8')
old = """            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                emb, mask, pad = pipe.encode_prompt(prompt=prompts, image=images, device='cuda')
            h, w = z.shape[-2:]
            packed = QwenImage21Pipeline._pack_latents(z, config['batch'], 64, h, w)
            img_mask = torch.cat([pad, torch.ones(config['batch'], h * w // 4, device='cuda',
                                                  dtype=torch.bool)], 1)"""
new = """            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                per = [pipe.encode_prompt(prompt=[prompts[i]], image=[images[i]], device='cuda')
                       for i in range(config['batch'])]
            emb = torch.nn.utils.rnn.pad_sequence([e[0][0] for e in per], batch_first=True)
            enc_mask = torch.nn.utils.rnn.pad_sequence([e[1][0] for e in per], batch_first=True)
            img_pad = torch.nn.utils.rnn.pad_sequence([e[2][0] for e in per], batch_first=True)
            h, w = z.shape[-2:]
            packed = QwenImage21Pipeline._pack_latents(z, config['batch'], 64, h, w)
            img_mask = torch.cat([img_pad, torch.ones(config['batch'], h * w // 4, device='cuda',
                                                      dtype=torch.bool)], 1)"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
old2 = """                prediction = model(hidden_states=packed, encoder_hidden_states=emb,
                                   encoder_hidden_states_mask=mask,"""
new2 = """                prediction = model(hidden_states=packed, encoder_hidden_states=emb,
                                   encoder_hidden_states_mask=enc_mask,"""
assert s.count(old2) == 1
s = s.replace(old2, new2)
p.write_text(s, encoding='utf-8')
ast.parse(s)
print('per-sample encode patched')
