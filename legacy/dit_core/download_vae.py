import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    print("Downloading VAE using hf-mirror.com...")
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-ema",
        local_dir="pretrained_models/sd-vae-ft-ema",
        local_dir_use_symlinks=False
    )
    print("Download complete!")
