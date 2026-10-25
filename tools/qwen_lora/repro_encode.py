"""Repro: encode_prompt with mixed-length prompts + images; print returned shapes."""
import sys

sys.path.insert(0, '/home/ds/Workspace/xy/qwen-lora-ref-20261025')
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-lora-ref-20261025/vendor')
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers import QwenImage21Pipeline  # noqa: E402

pipe = QwenImage21Pipeline.from_pretrained('/home/ds/Workspace/xy/qwen-lora-ref-20261025/base',
                                           transformer=None, dtype=torch.bfloat16,
                                           local_files_only=True).to('cuda')
img = Image.new('RGB', (256, 256), 'white')
prompts = ['短提示', '书法家：颜真卿。书体：楷书。请以参考字形的结构与布局为标准书写汉字「深」。纯白背景黑色毛笔单字']
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    emb, mask, pad = pipe.encode_prompt(prompt=prompts, image=[img, img], device='cuda')
print('emb', tuple(emb.shape), 'mask', tuple(mask.shape), 'pad', tuple(pad.shape))
print('pad dtype', pad.dtype, 'pad row sums', pad.sum(1).tolist())
