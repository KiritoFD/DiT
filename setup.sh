#!/usr/bin/env bash
set -e

echo "=== Setting up pretrained models for DiT Calligraphy ==="

# Create directory
mkdir -p pretrained_models

# 1. Download VAE model from ModelScope into relative path
echo "[1/2] Downloading VAE (stabilityai/sd-vae-ft-ema) via ModelScope..."
modelscope download --model stabilityai/sd-vae-ft-ema --local_dir pretrained_models/sd-vae-ft-ema

# 2. Download DiT pretrained weights (DiT-XL-2-256x256.pt)
echo "[2/2] Downloading DiT base model weights..."
python download.py

echo "=== Setup complete! All weights saved to pretrained_models/ ==="
