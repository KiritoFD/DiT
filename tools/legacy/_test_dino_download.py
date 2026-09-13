#!/usr/bin/env python3
"""Test: can we download DINOv2 model on remote?"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModel
print("Downloading model...")
m = AutoModel.from_pretrained("facebook/dinov2-base")
print(f"OK: {type(m).__name__}, hidden={m.config.hidden_size}")
