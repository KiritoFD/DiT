"""std_image must return RGBA (Qwen VAE takes 4 channels, same as GT cache path)."""
import ast
from pathlib import Path

p = Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025/code/train_std.py')
s = p.read_text(encoding='utf-8')
old = """        mask = render_base_mask(char, face, render_size=512)
        soft = soften_mask(mask, 0.65).resize((256, 256), Image.Resampling.BILINEAR)
        return Image.eval(soft, lambda v: 255 - v).convert('RGB')"""
new = """        mask = render_base_mask(char, face, render_size=512)
        soft = soften_mask(mask, 0.65).resize((256, 256), Image.Resampling.BILINEAR)
        gray = Image.eval(soft, lambda v: 255 - v)
        return Image.merge('RGBA', (gray, gray, gray, Image.new('L', gray.size, 255)))"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
ast.parse(s)
print('std_image -> RGBA')
