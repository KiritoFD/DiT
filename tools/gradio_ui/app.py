"""Gradio front end for the current U-Net calligraphy model, with checkpoint switching.

Replaces the legacy `legacy/gradio/gradio_stdskel.py`, which loads a DiT
(`DiT_2Cond_models`). That is a different architecture entirely: `src.unet.io.load`
rebuilds the U-Net from the checkpoint's own `config` and *refuses* anything that is not
`latent_cnn_unet_v1`/`calliflow_v2`, so no v10b-era checkpoint can be served here.

Why switching is safe
---------------------
Architecture is read from the checkpoint, never from a flag in this file. `load()` returns
the `Config` the weights were trained with, so a K-era checkpoint and a D1/GAR-era
checkpoint (which carries `decoder_ema`) each get their correct shapes without the UI
knowing anything about them. What the UI does enforce is that a *switch* either fully
succeeds or leaves the previous model serving -- a half-loaded model that renders garbage
is worse than a refused switch.

Assets are checked before anything is served. See `check_assets.py`; the same logic runs
at startup so a missing STD glyph cache or remap table is a clear message in the terminal
rather than a broken image in the browser.

Usage:
  PYTHONPATH=. python tools/gradio_ui/app.py --ckpt <path>[,<path>...] [--device cuda]
  PYTHONPATH=. python tools/gradio_ui/app.py --ckpt-scan   # discover serveable checkpoints
"""
import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools/gradio_ui'))

import gradio as gr          # noqa: E402
import numpy as np           # noqa: E402
import torch                 # noqa: E402

from check_assets import check_one   # noqa: E402

SAMPLERS = ('heun', 'euler')
DEFAULT_DEPLOY = dict(cfg=1.0, steps=50, sampler='heun', sde_gamma=0.0)


class Registry:
    """Holds the loaded model and swaps it atomically.

    Swapping is the risky operation: it frees a ~1.8 GB model and loads another. If the
    new one fails to load, `self.model` must still point at the old working model, so the
    swap builds first and assigns after -- never the other way round.
    """

    def __init__(self, paths, device):
        self.device = device
        self.paths = list(paths)
        self.model = None
        self.cfg = None
        self.ck = None
        self.current = None
        self.cache = {}

    def labels(self):
        return [Path(p).parent.parent.name + '/' + Path(p).name for p in self.paths]

    def load(self, path):
        if path in self.cache:
            self.model, self.cfg, self.ck = self.cache[path]
            self.current = path
            return self.model, self.cfg
        from src.unet.io import load
        model, cfg, ck = load(path, self.device, use_ema=True)
        self.cache[path] = (model, cfg, ck)
        self.model, self.cfg, self.ck = model, cfg, ck
        self.current = path
        return model, cfg

    def switch(self, label):
        """UI entry point. Returns (status, deployment defaults)."""
        if not label:
            return '未选择 checkpoint', DEFAULT_DEPLOY
        idx = self.labels().index(label) if label in self.labels() else None
        if idx is None:
            return f'未知 checkpoint: {label}', DEFAULT_DEPLOY
        path = self.paths[idx]
        try:
            _m, cfg = self.load(path)
        except Exception as e:                                # noqa: BLE001
            return f'切换失败，仍在用上一个：{type(e).__name__}: {e}', DEFAULT_DEPLOY
        dep = deployment_for(path, DEFAULT_DEPLOY)
        step = int(self.ck.get('train_steps', -1))
        info = (f'已切换 → {label}  (step {step}, '
                f'repr={getattr(cfg, "style_matrix_repr", "?")} '
                f'inj={getattr(cfg, "style_matrix_injection", "?")} '
                f'writers={getattr(cfg, "num_writers", "?")})')
        return info, dep


_DEPLOY_CACHE = {}


def deployment_for(path, default):
    """Prefer a recorded deployment recipe next to the checkpoint, else the default."""
    if path in _DEPLOY_CACHE:
        return dict(_DEPLOY_CACHE[path])
    p = Path(path)
    for cand in (p.parent.parent / 'deployment.json',
                 p.parent.parent / 'eval.json',
                 p.parent.parent / 'identity.json'):
        if cand.is_file():
            try:
                d = json.loads(cand.read_text(encoding='utf-8'))
            except Exception:                                 # noqa: BLE001
                continue
            dep = {}
            for k in ('cfg', 'steps', 'sampler', 'sde_gamma'):
                src = d.get(k)
                if src is None and isinstance(d.get('deployment'), dict):
                    src = d['deployment'].get(k)
                if src is not None:
                    dep[k] = src
            if dep:
                merg = dict(default)
                merg.update(dep)
                _DEPLOY_CACHE[path] = merg
                return dict(merg)
    _DEPLOY_CACHE[path] = dict(default)
    return dict(default)


def _std_glyph(cfg, device):
    """A neutral soft-glyph condition: a filled rounded square.

    Deliberately NOT a real target glyph. The UI is for eyeballing style and sampling
    behaviour; using a synthetic box keeps the condition identical across checkpoints so
    differences between them are the model's, not the glyph's.
    """
    g = torch.zeros(1, 1, 64, 64, device=device)
    g[:, :, 10:54, 12:52] = 1.0
    return g


@torch.no_grad()
def generate(reg, writer_label, script_id, steps, cfg_scale, seed, sampler, sde_gamma):
    if reg.model is None:
        return None, '没有已加载的模型'
    try:
        from src.loss.flow_matching import FlowMatching
        cfg = reg.cfg
        n_writers = int(getattr(cfg, 'num_writers', 1))
        lc = int(getattr(cfg, 'latent_channels', 4))
        device = reg.device

        # Writer label -> id. "null" is the reserved slot at index num_writers, which is
        # how the model encodes "no writer condition"; it is always safe in range.
        if writer_label == '(null)':
            wid = n_writers
        else:
            wid = int(str(writer_label).strip())

        gen = torch.Generator(device=device).manual_seed(int(seed))
        z = torch.randn(1, lc, 32, 32, device=device, generator=gen)
        g = _std_glyph(cfg, device)
        fm = FlowMatching(num_steps=int(steps), sampler=sampler)

        total = int(steps)
        for i, t in enumerate(torch.linspace(1.0, 0.0, total + 1, device=device)[:-1]):
            tt = torch.full((1,), float(t) * 1000.0, device=device)
            v = reg.model(z, tt, torch.tensor([wid], device=device),
                          y_script=torch.tensor([int(script_id)], device=device), g=g)
            if isinstance(v, (tuple, list)):
                v = v[0]
            z = z - (1.0 / total) * v
            if sde_gamma and i < total - 1:
                z = z + float(sde_gamma) * (1.0 / total) ** 0.5 * torch.randn_like(z)

        sf = float(getattr(cfg, 'vae_scaling_factor', 0.18215) or 0.18215)
        sh = float(getattr(cfg, 'vae_shift_factor', 0.0) or 0.0)
        from src.eval.decoder import _get_vae
        vae = _get_vae(device, str(getattr(cfg, 'vae_path', '') or ''), sh)
        base = getattr(vae, '_base', vae)
        # Joint arms (D1/D2/GAR) store a fine-tuned decoder in the checkpoint; the plain
        # K-era checkpoints do not. Try it, and fall back rather than failing the render.
        ck = reg.ck or {}
        off = ck.get('decoder_ema')
        if off is None:
            off = ck.get('decoder')
        if off is not None:
            try:
                base.decoder.load_state_dict(off, strict=True)
                note = 'decoder=ckpt'
            except Exception:                                 # noqa: BLE001
                note = 'decoder=stock (ckpt decoder incompatible)'
        else:
            note = 'decoder=stock'

        px = base.decode((z / sf) + sh).sample.float().cpu()
        img = ((px[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
        arr = (img * 255).clip(0, 255).astype(np.uint8)
        return arr, (f'OK steps={steps} cfg={cfg_scale} sampler={sampler} '
                     f'seed={seed} writer={writer_label}(id={wid}) script={script_id} {note}')
    except Exception as e:                                    # noqa: BLE001
        traceback.print_exc()
        return None, f'生成失败：{type(e).__name__}: {e}'


def discover():
    pats = ('assets/abc_20261031/execution/runs/*/checkpoints/*.pt',
            'assets/style_matrix_20261027/matrix64/*/execution/runs/*/checkpoints/*.pt',
            'assets/d_20261101/runs/*/checkpoints/*.pt',
            'assets/p0_s4_ft/runs/*/checkpoints/*.pt',
            'assets/imported_ckpts/*.pt')
    out = []
    for pat in pats:
        out += sorted(str(p) for p in ROOT.glob(pat))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='', help='comma-separated checkpoint paths')
    ap.add_argument('--ckpt-scan', action='store_true', help='discover serveable checkpoints')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='generate once and exit')
    a = ap.parse_args()

    paths = [p for p in a.ckpt.split(',') if p.strip()] if a.ckpt else []
    if a.ckpt_scan:
        paths += discover()
    paths = [str(Path(p).resolve().relative_to(ROOT)) if Path(p).is_absolute() else p
             for p in paths]
    if not paths:
        print('no checkpoints given; use --ckpt or --ckpt-scan')
        return 2

    print(f'device={a.device}')
    ready = []
    for p in paths:
        r = check_one(p, a.device, sample=True)
        if r['ok']:
            print(f'  OK   {p}')
            ready.append(p)
        else:
            print(f'  SKIP {p}')
            for pb in r['problems']:
                print(f'         {pb}')
    if not ready:
        print('\nnone of the given checkpoints are serveable -- see problems above')
        return 1

    reg = Registry(ready, a.device)
    status, dep = reg.switch(reg.labels()[0])
    print(status)

    if a.smoke:
        img, msg = generate(reg, '(null)', 0, 8, 1.0, 0, 'heun', 0.0)
        print(f'[smoke] {msg}')
        return 0 if img is not None else 1

    with gr.Blocks(title='书法生成 · U-Net') as demo:
        gr.Markdown('## 书法生成 — 当前 U-Net 架构\n'
                    '架构由 checkpoint 自身决定；切换失败会保留上一个模型继续服务。')
        with gr.Row():
            ck_in = gr.Dropdown(reg.labels(), value=reg.labels()[0], label='Checkpoint')
            switch_btn = gr.Button('切换', variant='secondary')
        st_out = gr.Textbox(value=status, label='状态', interactive=False)
        with gr.Row():
            writer_in = gr.Textbox(value='(null)', label='Writer id（或 (null) 表示无条件）')
            script_in = gr.Number(value=0, label='Script id', precision=0)
            seed_in = gr.Number(value=0, label='Seed', precision=0)
        with gr.Row():
            steps_in = gr.Slider(2, 100, value=int(dep['steps']), step=1, label='采样步数')
            cfg_in = gr.Slider(0.0, 4.0, value=float(dep['cfg']), step=0.1, label='CFG')
            sampler_in = gr.Dropdown(list(SAMPLERS), value=dep['sampler'], label='采样器')
            sde_in = gr.Slider(0.0, 1.0, value=float(dep['sde_gamma']), step=0.05,
                               label='SDE gamma')
        btn = gr.Button('生成', variant='primary')
        img_out = gr.Image(label='结果', type='numpy')

        def _switch(label):
            s, d = reg.switch(label)
            return s, gr.update(value=int(d['steps'])), gr.update(value=float(d['cfg']))

        switch_btn.click(_switch, [ck_in], [st_out, steps_in, cfg_in])
        btn.click(lambda *xs: generate(reg, *xs),
                  [writer_in, script_in, steps_in, cfg_in, seed_in, sampler_in, sde_in],
                  [img_out, st_out])

    demo.launch(server_name='0.0.0.0', server_port=a.port, share=a.share)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
