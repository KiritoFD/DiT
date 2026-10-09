import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free: {free / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB")
