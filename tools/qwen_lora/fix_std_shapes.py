"""Print step-1 shapes to pinpoint the loss broadcast failure."""
import ast
from pathlib import Path

p = Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025/code/train_std.py')
s = p.read_text(encoding='utf-8')
old = """                pred_img = QwenImage21Pipeline._unpack_latents(
                    prediction, config['resolution'], config['resolution'], 16)
                loss = (pred_img.float() - (noise4 - clean4).float()).square().mean()"""
new = """                pred_img = QwenImage21Pipeline._unpack_latents(
                    prediction, config['resolution'], config['resolution'], 16)
                if step == 1:
                    print('SHAPES prediction', tuple(prediction.shape), 'pred_img',
                          tuple(pred_img.shape), 'clean', tuple(clean.shape),
                          'clean4', tuple(clean4.shape), 'noise4', tuple(noise4.shape), flush=True)
                loss = (pred_img.float() - (noise4 - clean4).float()).square().mean()"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
ast.parse(s)
print('shape probe added')
