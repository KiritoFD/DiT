import sys
sys.path.insert(0, "/root/Workspace/xy/DiT")
from models import DiT_2Cond_models
print("B/4 registered:", "DiT-2Cond-B/4" in DiT_2Cond_models)
print("all models:", list(DiT_2Cond_models.keys()))
