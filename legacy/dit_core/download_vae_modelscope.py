import os
import urllib.request

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully downloaded {dest}")
    except Exception as e:
        print(f"Failed to download {dest}: {e}")

if __name__ == "__main__":
    out_dir = "pretrained_models/sd-vae-ft-ema"
    os.makedirs(out_dir, exist_ok=True)
    
    config_url = "https://www.modelscope.cn/api/v1/models/AI-ModelScope/sd-vae-ft-ema/repo?Revision=master&FilePath=config.json"
    safe_url = "https://www.modelscope.cn/api/v1/models/AI-ModelScope/sd-vae-ft-ema/repo?Revision=master&FilePath=diffusion_pytorch_model.safetensors"
    
    download_file(config_url, os.path.join(out_dir, "config.json"))
    download_file(safe_url, os.path.join(out_dir, "diffusion_pytorch_model.safetensors"))
    
    print("\nAll downloads finished! You can now run the training script without MockVAE.")
