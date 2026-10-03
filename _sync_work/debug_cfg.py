import sys, os, json
sys.path.insert(0, "/root/Workspace/xy/DiT")
import src.train.train as T

cfg = "src/train/configs/v12_pretrain_S_cat_fame_kxl_tj_px60.json"
print("cwd:", os.getcwd())
print("cfg exists:", os.path.isfile(cfg), os.path.abspath(cfg) if os.path.isfile(cfg) else "")
try:
    d = json.load(open(cfg, encoding="utf-8"))
    print("json keys:", len(d))
    print("  condition_fusion =", d.get("condition_fusion"))
    print("  model            =", d.get("model"))
    print("  data_csv         =", d.get("data_csv"))
    print("  max_steps        =", d.get("max_steps"))
except Exception as e:
    print("json load FAILED:", type(e).__name__, e)

# how many parsed actions have dest in config?
print("\n--model argparse default:")
for a in T.__dict__.get("_PARSER", []):
    pass
