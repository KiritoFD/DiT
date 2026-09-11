import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.chdir('./')
import torch
import random
import numpy as np
# 由于版本不兼容问题，需要这个代码补丁
# 在导入diffusers之前添加以下代码
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download

from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from moyun.moyun_2 import moyun_12channel
from utils.diffusion import create_diffusion
from utils.config import text2label
from utils.stroke import get_ch_strokes_tensor
from utils.download import find_model
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 256
latent_size = image_size // 8
vae = AutoencoderKL.from_pretrained("/mnt/ssd/model/image/sd-vae-ft-ema").to(device)

def get_condition(calligraphy):
    class_labels = [text2label(*item) for item in calligraphy]
    # class_labels = [load_features(item) for item in calligraphy]
    stroke = [get_ch_strokes_tensor(item[2]).numpy() for item in calligraphy]
    stroke = np.array(stroke)
    return class_labels, stroke

def generate_image(calligrapher, font, charactors, model_path:str, seed = -1, cfg_scale = 1.5, steps = 50, samples_per_row = 5):
    
    seed = int(seed) if seed != -1 else random.randint(1, 65536)
    torch.manual_seed(seed)
    model = moyun_12channel(input_size=latent_size,if_rope = False, if_rope_residual = False, device = device, num_classes=4793).to(device)
    state_dict = find_model(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    class_labels, stroke = get_condition((calligrapher, font, charactor) for charactor in charactors)
    diffusion = create_diffusion(str(steps))
    n = len(class_labels)
    z = torch.randn(n, 12, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([(4793,4793,4793)] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, stroke = stroke, cfg_scale=cfg_scale)
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, 
        model_kwargs=model_kwargs, progress=False, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = samples[:,0:4,:,:]
    samples = vae.decode(samples / 0.18215).sample

    save_image(samples, "./sample_250.png", nrow=int(samples_per_row), 
           normalize=True, value_range=(-1, 1))
    samples = Image.open("./sample_250.png")
    return samples



if __name__ == "__main__":
    generate_image(model_path= "model_train_ful/000-moyun-12channel/checkpoints/0243000.pt", calligrapher='王羲之', font='楷书',charactors=('舍','白','初','高'))
