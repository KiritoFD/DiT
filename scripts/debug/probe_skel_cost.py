import time, os
import numpy as np
from PIL import Image

os.chdir('/root/Workspace/xy/DiT')
a = np.asarray(Image.open('final_imgs_256/100560.png').convert('L'))
print('img shape', a.shape, flush=True)

t0 = time.time()
for _ in range(20):
    a.copy()
print('np copy: %.2f ms' % ((time.time() - t0) / 20 * 1000), flush=True)

try:
    import cv2
    import cv2.ximgproc
    print('cv2', cv2.__version__, flush=True)
    bw = (a < 127).astype('uint8') * 255
    t0 = time.time()
    for _ in range(5):
        th = cv2.ximgproc.thinning(bw, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    print('cv2 thinning per img: %.1f ms' % ((time.time() - t0) / 5 * 1000), flush=True)
except Exception as e:
    print('cv2 err:', repr(e), flush=True)

from skimage.morphology import skeletonize
t0 = time.time()
for _ in range(3):
    sk = skeletonize(a < 127)
print('skimage skeletonize per img: %.1f ms' % ((time.time() - t0) / 3 * 1000), flush=True)

from scipy.ndimage import binary_dilation, generate_binary_structure
st = generate_binary_structure(2, 2)
t0 = time.time()
for _ in range(3):
    d = binary_dilation(sk, structure=st, iterations=3)
print('dil3 per img: %.1f ms' % ((time.time() - t0) / 3 * 1000), flush=True)

# VAE encode reference cost at 4 threads
import torch
torch.set_num_threads(4)
import torchvision.transforms as T
from diffusers.models import AutoencoderKL
tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
vae = AutoencoderKL.from_pretrained('pretrained_models/sd-vae-ft-ema').eval()
xb = torch.stack([tf(Image.open('final_imgs_256/100560.png').convert('RGB'))] * 16)
with torch.no_grad():
    vae.encode(xb).latent_dist.mode()
    t0 = time.time()
    for _ in range(10):
        vae.encode(xb).latent_dist.mode()
dt = (time.time() - t0) / 10
print('vae encode bs16 thr4: %.1f ms/batch -> %.2f img/s' % (dt * 1000, 16 / dt), flush=True)
print('DONE', flush=True)
