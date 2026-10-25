"""Instrument the real pipeline call: print transformer arg shapes for an image-conditioned run."""
import sys

sys.path.insert(0, '/home/ds/Workspace/xy/qwen-lora-ref-20261025')
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-lora-ref-20261025/vendor')
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers import QwenImage21Pipeline  # noqa: E402

pipe = QwenImage21Pipeline.from_pretrained('/home/ds/Workspace/xy/qwen-lora-ref-20261025/base',
                                           dtype=torch.bfloat16, local_files_only=True).to('cuda')
pipe.set_progress_bar_config(disable=True)
img = Image.new('RGB', (256, 256), 'white')

orig = pipe.transformer.forward


def spy(*args, **kwargs):
    hs = kwargs.get('hidden_states')
    im = kwargs.get('img_mask')
    es = kwargs.get('encoder_hidden_states')
    esm = kwargs.get('encoder_hidden_states_mask')
    sh = kwargs.get('img_shapes')
    print('SPY hidden', tuple(hs.shape), 'enc', tuple(es.shape), 'enc_mask_sum',
          int(esm.sum()) if esm is not None else None,
          'img_mask', tuple(im.shape), 'true_per_row', im.sum(1).tolist()[:2],
          'img_shapes', sh[0] if sh else None, flush=True)
    out = orig(*args, **kwargs)
    print('SPY out', tuple(out[0].shape) if isinstance(out, tuple) else 'dict', flush=True)
    return out


pipe.transformer.forward = spy
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    pipe(prompt='书法家：颜真卿。书体：楷书。请以参考字形为结构标准书写汉字「深」。',
         image=[img], height=256, width=256, output_resolution=256,
         num_inference_steps=2, true_cfg_scale=1., use_kv_cache=False,
         generator=torch.Generator('cpu').manual_seed(90))
print('PIPE_OK')
