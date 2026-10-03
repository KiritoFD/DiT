import sys

sys.path.insert(0, "/root/Workspace/xy/DiT")
import os

os.chdir("/root/Workspace/xy/DiT")
sys.argv = ["x", "--config", "src/train/configs/v15_fs_沈周.json",
            "--train-only-new-callig", "--init-new-callig", "mean_scaled"]
from src.train.cli import parse_args

a = parse_args()
print("  train_only_new_callig =", a.train_only_new_callig)
print("  init_new_callig       =", a.init_new_callig)
print("  num_calligraphers     =", a.num_calligraphers)
print("  OK")
