import sys
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

packages = ["torchvision", "diffusers", "timm", "einops", "wandb", "cv2", "PIL", "scipy", "h5py"]
for p in packages:
    try:
        m = __import__(p)
        print(f"  {p}: {getattr(m, '__version__', 'ok')}")
    except ImportError as e:
        print(f"  {p}: MISSING ({e})")
