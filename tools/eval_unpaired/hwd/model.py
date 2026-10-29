"""Minimal HWD VGG16 port, matching upstream non-streaming square-image path."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.models import vgg16

UPSTREAM = 'eabb5b5ede5899ac7f8b716429cbd1a1c97c7485'
URL = 'https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth'


def load(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    net = vgg16(weights=None, num_classes=10400)
    net.load_state_dict(state, strict=True)
    return net.features.eval().requires_grad_(False)


def preprocess(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        w, h = image.size
        if w < h:
            left = (h-w)//2
            image = ImageOps.expand(image, (left, 0, h-w-left, 0), fill='white')
        w, h = image.size
        # Current study uses square single-character tiles, hence one VGG patch/image.
        if w != h:
            raise ValueError('This audited HWD path requires square tiles: '+str(path))
        array = np.asarray(image.resize((32, 32), Image.Resampling.NEAREST)).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float()/255


@torch.inference_mode()
def extract(net, paths):
    # Upstream default batch=1. No ImageNet normalization or feature L2 normalization.
    features = [net(preprocess(p)[None]).reshape(512).numpy() for p in paths]
    return np.stack(features)


def distance(x, y, rows):
    writers = sorted({r['calligrapher'] for r in rows})
    per_writer = {}
    for writer in writers:
        mask = np.array([r['calligrapher'] == writer for r in rows])
        per_writer[writer] = float(np.linalg.norm(x[mask].mean(0)-y[mask].mean(0)))
    return dict(hwd=float(np.mean(list(per_writer.values()))), per_writer=per_writer,
                n=len(rows), writers=len(writers))
