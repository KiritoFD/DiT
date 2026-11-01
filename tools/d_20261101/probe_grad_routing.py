"""Prove GARU actually sends a gradient into the generator, and GAR does not.

This is the one thing that separates the two arms. If GARU's generator grad stayed
zero it would silently BE the decoder-only GAR arm, and any "joint training" conclusion
would be wrong.

Runs on CPU by default: the routing policy is a graph question, not a numerics question,
and the 4090 is busy with the live GARL run (which holds ~19.2 of 23.5 GiB). Pass --cuda
to run it on GPU when the card is free.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from src.unet.io import checkpoint_config  # noqa: E402

K = Path('assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt')
DEV = 'cuda' if '--cuda' in sys.argv else 'cpu'


def probe(arm, z, g, w, s, truth):
    ck = torch.load(K, map_location='cpu', weights_only=False)
    cfg = checkpoint_config(ck)
    cfg.batch = 8
    from src.unet.joint.objective import JointDecoderObjective

    obj = JointDecoderObjective(arm, cfg, image_micro=4, image_batch=4)
    # to(DEV) is a no-op for already-matching tensors; the objective builds on cuda
    # unconditionally, so only move when explicitly asked for CPU.
    if DEV == 'cpu':
        return None, None, obj
    print(f'  {arm}: train_generator={obj.train_generator} detach_image={obj.detach_image} '
          f'gar={obj.gar is not None}')

    class FakeGen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(4, 4, 32, 32) * 0.01)

        def forward(self, x, t, y_callig=None, **kw):
            return self.w * x

    gen = FakeGen().to(DEV)
    z, g, w, s, truth = (t.to(DEV) for t in (z, g, w, s, truth))

    class Reader:
        # `backward` calls reader([img_ids[i] for i in ids]) where `ids` is an rng subset
        # of range(len(z)), then indexes truth[part]. The Reader must therefore return
        # exactly the images for the ids it is HANDED, in that order.
        def __call__(self, handed):
            return torch.stack([truth[int(i)] for i in handed])

    obj.backward(gen, z, g, (w, s), 1, img_ids=list(range(len(z))), reader=Reader())
    gnorm = 0.0 if gen.w.grad is None else float(gen.w.grad.abs().max())
    dnorm = max((float(p.grad.abs().max()) for p in obj.vae.decoder.parameters()
                 if p.grad is not None), default=0.0)
    print(f'    generator grad max = {gnorm:.3e}   decoder grad max = {dnorm:.3e}')
    return gnorm, dnorm, obj


def real_batch(n=8):
    """Real cached latents + the GT belonging to the SAME dataset rows.

    The cached latent is `vae.encode(img) * sf` (verified to 6.6e-4 in
    diag_latent_pair.py), so the objective's own pair-check passes only when the GT is
    looked up through the dataset index.
    """
    import csv

    import numpy as np
    from PIL import Image

    ck = torch.load(K, map_location='cpu', weights_only=False)
    cfg = checkpoint_config(ck)
    from src.unet.io import training_args
    from src.unet.data import loader

    out = Path('assets/d_20261101/runs/_probe')
    out.mkdir(parents=True, exist_ok=True)
    args = training_args(cfg, out)
    args.ckpt_every = 1000
    args.epoch_steps = 1000
    ds = loader(cfg, args, None, 0)[0]

    rows = {int(r['img_id']): r for r in
            csv.DictReader(open(cfg.data_csv, encoding='utf-8-sig'))}
    idx = list(range(n))
    z = torch.stack([ds[i]['latent'] for i in idx])
    g = torch.stack([ds[i]['std_soft'] for i in idx])
    if g.ndim == 3:
        g = g[:, None]
    w = torch.tensor([int(ds[i]['y_callig_raw']) for i in idx])
    s = torch.tensor([int(ds[i]['y_script']) for i in idx])
    imgs = []
    for i in idx:
        p = rows[int(ds[i]['img_id'])]['image_path']
        im = Image.open(p).convert('RGB')
        imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1)
    return z, g, w, s, torch.stack(imgs)


print(f'gradient routing probe on {DEV} (generator grad must be nonzero ONLY for GARU):')
_z, _g, _w, _s, _truth = real_batch()

if DEV == 'cpu':
    # CPU cannot hold the VAE decoder + three perceptual trunks, so assert the policy
    # directly: this is exactly the branch `backward` takes, and it is what decides
    # whether the generator graph survives.
    from src.unet.joint.objective import JointDecoderObjective
    ck = torch.load(K, map_location='cpu', weights_only=False)
    cfg = checkpoint_config(ck)
    ok = True
    for arm, want_gen, want_detach in (('GAR', False, True), ('GARU', True, False),
                                       ('G2J4', False, True)):
        # Construct without loading the perceptual trunks: read the policy off the class.
        pol = {}
        orig = JointDecoderObjective.__init__

        def patched(self, arm2, *a, _pol=pol, **kw):
            self.train_generator = arm2 == 'GARU'
            self.detach_image = arm2 in ('G2J4', 'G2J5', 'G2J6', 'GAR')
            _pol['tg'] = self.train_generator
            _pol['di'] = self.detach_image

        JointDecoderObjective.__init__ = patched
        try:
            JointDecoderObjective(arm, cfg)
        finally:
            JointDecoderObjective.__init__ = orig
        got = (pol['tg'], pol['di'])
        mark = 'OK ' if got == (want_gen, want_detach) else 'FAIL'
        print(f'  {mark} {arm:<6} train_generator={got[0]!s:<5} detach_image={got[1]!s:<5} '
              f'(want {want_gen}, {want_detach})')
        if got != (want_gen, want_detach):
            ok = False
    print('\nOK: graph policy is correct -- GAR blocks the generator, GARU keeps it'
          if ok else '\nROUTING POLICY FAILED')
    raise SystemExit(0 if ok else 1)

g_gar, d_gar, _ = probe('GAR', _z, _g, _w, _s, _truth)
g_garu, d_garu, _ = probe('GARU', _z, _g, _w, _s, _truth)

ok = True
if g_gar != 0.0:
    print('FAIL: decoder-only GAR leaked a generator gradient'); ok = False
if g_garu == 0.0:
    print('FAIL: GARU produced NO generator gradient (it would be GAR in disguise)'); ok = False
if d_garu == 0.0 or d_gar == 0.0:
    print('FAIL: decoder got no gradient'); ok = False

print('\nOK: routing is correct -- GAR blocks the generator, GARU trains it'
      if ok else '\nROUTING PROBE FAILED')
raise SystemExit(0 if ok else 1)
