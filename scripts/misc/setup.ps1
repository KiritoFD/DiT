# PowerShell setup script for DiT Calligraphy Project
$ErrorActionPreference = "Stop"

Write-Host "=== Setting up pretrained models for DiT Calligraphy ===" -ForegroundColor Green

# Ensure relative folder exists
if (-not (Test-Path -Path "pretrained_models")) {
    New-Item -ItemType Directory -Path "pretrained_models" | Out-Null
}

# 1. Download VAE model via ModelScope into relative path
Write-Host "[1/2] Downloading VAE (stabilityai/sd-vae-ft-ema) via ModelScope..." -ForegroundColor Cyan
modelscope download --model stabilityai/sd-vae-ft-ema --local_dir pretrained_models/sd-vae-ft-ema

# 2. Download DiT pretrained weights
Write-Host "[2/2] Downloading DiT base model weights..." -ForegroundColor Cyan
python download.py

Write-Host "=== Setup complete! All weights saved to pretrained_models/ ===" -ForegroundColor Green
