# -*- coding: utf-8 -*-
"""训练入口 (launcher): 实际实现移至 src/train/train.py.

用法不变:
    python train.py --config exp_s18_s_flow.json [--diffusion-type flow]
"""
from src.train.train import main

if __name__ == "__main__":
    main()
