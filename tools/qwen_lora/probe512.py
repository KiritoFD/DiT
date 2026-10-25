"""P1-A: 20k LoRA at 512 inference (downscaled to 256 for metrics) + GT VAE roundtrip comparison."""
import json
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
from diffusers import QwenImage21Transformer2DModel, QwenImage21Pipeline, AutoencoderKLQwenImage21
from src.eval.metrics import ssim, get_lpips
from src.eval.metrics_ink import ink_iou

CKPT = root / 'checkpoints' / '0020000'
FIELDS = ('ssim', 'lpips', 'ink_iou')


def main():
    out = root / 'p1a_probe512'
    (out / 'images').mkdir(parents=True, exist_ok=False)
    data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
    cfg = json.loads((root / 'config.json').read_text())
    text = torch.load(root / 'text.pt', map_location='cpu', weights_only=True)
    test = data['datasets']['test']

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
    model.eval()
    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model, text_encoder=None,
                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    lp, error = get_lpips('cuda')
    if lp is None:
        raise RuntimeError(error)

    rows = []
    started = time.monotonic()
    for i, row in enumerate(test):
        emb = text[(row['character'], row['calligrapher'], row['script'])][None].cuda()
        mask = torch.ones(emb.shape[:2], device='cuda', dtype=torch.bool)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            img = pipe(prompt_embeds=emb, prompt_embeds_mask=mask, height=512, width=512,
                       output_resolution=512, num_inference_steps=8, true_cfg_scale=1.,
                       use_kv_cache=True, generator=torch.Generator('cpu').manual_seed(90)).images[0]
        white = Image.new('RGBA', img.size, 'white'); white.alpha_composite(img.convert('RGBA'))
        pred = white.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
        if i < 24:
            pred.save(out / 'images' / f'{i:03d}_{row["calligrapher"]}_{row["script"]}_{row["character"]}.png')
        gt = Image.open(root / row['file']).convert('RGB')
        p = np.asarray(pred, dtype=np.float32) / 255
        g = np.asarray(gt, dtype=np.float32) / 255
        ten = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda() * 2 - 1
        with torch.inference_mode():
            d = float(lp(ten(p), ten(g)).item())
        rows.append(dict(ssim=round(ssim(p, g), 5), lpips=round(d, 5),
                         ink_iou=round(ink_iou(p, g), 5)))
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/200 {time.monotonic() - started:.0f}s', flush=True)
    gen512 = {f: round(st.mean(r[f] for r in rows), 5) for f in FIELDS}
    gen256 = {f: round(st.mean(r[f] for r in json.loads(
        (root / 'test_eval_20k' / 'metrics.json').read_text(encoding='utf-8'))), 5) for f in FIELDS}
    pipe.vae.to('cpu'); del pipe
    torch.cuda.empty_cache()

    vae = AutoencoderKLQwenImage21.from_pretrained(root / 'base', subfolder='vae',
                                                   torch_dtype=torch.bfloat16,
                                                   local_files_only=True).eval().requires_grad_(False).cuda()
    rt = {'256': [], '512': []}
    for i, row in enumerate(test[:200]):
        im = Image.open(root / row['file']).convert('RGBA')
        for res in ('256', '512'):
            sized = im if res == '256' else im.resize((512, 512), Image.Resampling.LANCZOS)
            pixels = (torch.from_numpy(np.array(sized).copy()).permute(2, 0, 1).float() / 127.5 - 1)
            pixels = pixels.cuda().to(torch.bfloat16)[None, :, None]
            with torch.no_grad():
                decoded = vae.decode(vae.encode(pixels).latent_dist.mean).sample
            back = ((decoded[0].float().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5)
            back = np.clip(back, 0, 255).astype(np.uint8)
            back_img = Image.fromarray(back, 'RGBA').convert('RGB')
            if res == '512':
                back_img = back_img.resize((256, 256), Image.Resampling.LANCZOS)
            gt = np.asarray(Image.open(root / row['file']).convert('RGB'), dtype=np.float32) / 255
            b = np.asarray(back_img, dtype=np.float32) / 255
            rt[res].append(ssim(b, gt))
    roundtrip = {k: round(st.mean(v), 5) for k, v in rt.items()}

    summary = dict(gen512_then256=gen512, gen256_direct=gen256,
                   gt_vae_roundtrip_ssim=roundtrip, note='test200, seed90, 8 steps, LoRA 20k')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
